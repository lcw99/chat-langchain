"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse

from urllib.request import urlopen
import json
from datetime import datetime
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None

corpText = ""

@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)

    DART_KEY = "0eb9d7eb5c3e5d1cc03806a54a23d30d621459bd"
    coprCode = "01160363"
    quaterDict = {"11013": "Q1", "11012": "Q2", "11014": "Q3", "11011": "Final"}
    yearQColumns = []
    years = []
    #dataArrayYear = {}
    dataArrayQuarter = {}
    accountAccumulation = {}
    for bsnsYear in range(2015, datetime.now().year + 1):
        for reportCode in ["11013", "11012", "11014", "11011"]: # 1분기보고서 : 11013, 반기보고서 : 11012, 3분기보고서 : 11014, 사업보고서 : 11011
            url = f"https://opendart.fss.or.kr/api/fnlttSinglAcnt.json?crtfc_key={DART_KEY}&corp_code={coprCode}&bsns_year={bsnsYear}&reprt_code={reportCode}&fs_div=CFS"
            # url = f"https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json?crtfc_key={DART_KEY}&corp_code={coprCode}&bsns_year={bsnsYear}&reprt_code={reportCode}&fs_div=CFS"
            response = urlopen(url)
            data = json.loads(response.read())
            print(f"{data['status']=}")
            if (data['status'] == '000'):
                year = str(bsnsYear)
                years.append(year)
                quater = quaterDict[reportCode]
                yearQ = f"{year}.{quater}"
                print(f"{yearQ=}")
                if yearQ not in yearQColumns:
                    if quater == "Final":
                        yearQColumns.append(f"{year}.Q4") 
                    else:
                        yearQColumns.append(f"{year}.{quater}") 
                accountData = data['list']
                for d in accountData:
                    if d['fs_div'] != 'OFS':    # CFS: 연결재무제표, OFS: 재무제표
                        continue
                    isIS = d['sj_div'] == 'IS'
                    accountName = d['account_nm']
                    accountValue = d['thstrm_amount']
                    accKey = f"{accountName}.{year}"
                    if accountValue != '':
                        accountValueMil = int(int(accountValue.replace(",", "")) / 1000000)
                        if accKey not in accountAccumulation.keys():
                            accountAccumulation[accKey] = 0
                        if isIS and quater != "Final":
                            accountAccumulation[accKey] += accountValueMil
                    else:
                        accountValueMil = "NA"
                    if accountName not in dataArrayQuarter.keys():
                        dataArrayQuarter[accountName] = {}
                    dataArrayQuarter[accountName][f"{year}.{quater}"] = accountValueMil
                    if quater == 'Final':
                        if f"{year}.Q1" in yearQColumns and f"{year}.Q2" in yearQColumns and f"{year}.Q3" in yearQColumns:
                            if isIS:
                                dataArrayQuarter[accountName][f"{year}.Q4"] = accountValueMil - accountAccumulation[accKey]
                            else:
                                dataArrayQuarter[accountName][f"{year}.Q4"] = dataArrayQuarter[accountName][f"{year}.Final"]
                    # print(f"{yearQ} {accountName} {d['thstrm_amount']}") 
    global corpText
    corpText = f"요약재무제표(단위: 백만원)\n"
    corpText += "계정과목\t" + "\t".join(yearQColumns) + "\n"
    for an in dataArrayQuarter.keys():
        line = ""
        for yq in yearQColumns:
            if yq in dataArrayQuarter[an].keys():
                line += str(dataArrayQuarter[an][yq]) + "\t"
            else:
                line += "NA\t"
        corpText += f"{an}\t{line.strip()}\n"
    print(corpText)

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
