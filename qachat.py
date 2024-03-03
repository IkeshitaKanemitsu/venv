import os
from urllib import response
import faiss
import openai

from dotenv import load_dotenv
from operator import itemgetter

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationEntityMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel

# ###############################################################
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
# FAISS_DB_DIR = os.environ["FAISS_DB_DIR"]
FAISS_DB_DIR = "venv/apps/faiss_db"
REDIS_URL = os.environ["REDIS_URL"] 
# ###############################################################
def qa(query):
    embedding = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )
    db = FAISS.load_local("/home/ubuntu/venv/venv/faiss_db", embedding)
    retriever = db.as_retriever()
    
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    
    template = """Your name is "Dorami-chan".Basically, we converse in Japanese. We will answer in English when specified by the customer.Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use six sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" in Japanese at the end of the answer.:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
        ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    _inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
    | StrOutputParser(),
)
    _context = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()
    
    response = conversational_qa_chain.invoke(
    {
        "question": query,
        "chat_history": [
            HumanMessage(content="私はは日本人の優しい女性もみかるスタッフです"),
            AIMessage(content='？\nAI: あなたは優秀なAIアシスタントです。'),
            ],
        }
    )
    res = response.content
    return res

# print(qa("もみかるについて教えて"))


