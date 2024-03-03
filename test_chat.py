import os
import faiss
import openai
import getpass

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

import os
import faiss
import openai
from dotenv import load_dotenv
# from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import faiss
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import ConversationChain
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchainhub import Client
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


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
    

    return 
retriever = VectorStoreRetriever(vectorstore=FAISS(...))
retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=retriever)



# print(qa("もみかるについて教えて"))