"""Create a ChatVectorDBChain for question/answering."""
from __future__ import annotations

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager

# from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.tracers.langchain_v1 import LangChainTracerV1
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore

from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question in Korean.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """기업의 재무제표를 놓고 Human과 AI가 대화 중이다. AI는 기업분석 전문가의 입장에서 친절하고 상세하게 답변한다.

{history}
Human:{input}
AI:"""
CONV_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["history", "input"]
)


def get_chain(
    vectorstore: VectorStore,
    question_handler: AsyncCallbackHandler,
    stream_handler: AsyncCallbackHandler,
    tracing: bool = False,
) -> ConversationChain:
    """Create a ConversationalRetrievalChain for question/answering."""
    # Construct a ConversationalRetrievalChain with a streaming llm for
    # combine docs and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        # tracer = LangChainTracer()
        tracer = LangChainTracerV1()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = ChatOpenAI(
        temperature=0,
        verbose=True,
        callbacks=[question_handler],
    )  # type: ignore

    streaming_llm = ChatOpenAI(
        streaming=True,
        callbacks=[stream_handler],
        verbose=True,
        temperature=0,
    )  # type: ignore

    question_generator = LLMChain(
        llm=question_gen_llm,
        prompt=CONDENSE_QUESTION_PROMPT,
        callbacks=[question_handler],
        verbose=True,
    )

    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )

    # qa = ConversationalRetrievalChain(
    #     retriever=vectorstore.as_retriever(),
    #     question_generator=question_generator,
    #     combine_docs_chain=doc_chain,
    #     callbacks=manager.handlers,
    #     verbose=True,
    # )
    
    qa = ConversationChain(
        memory=ConversationBufferMemory(),
        llm=streaming_llm,
        prompt=CONV_PROMPT,
        callbacks=manager.handlers,
        verbose=True,
    )
    
    return qa
