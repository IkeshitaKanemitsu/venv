import os
import faiss
import openai
from dotenv import load_dotenv
# from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory

from langchain.chains import LLMChain, SimpleSequentialChain


# ###############################################################

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
# FAISS_DB_DIR = os.environ["FAISS_DB_DIR"]
FAISS_DB_DIR = "venv/apps/faiss_db"

# ###############################################################
def qa(query):
    embedding = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )
    db = FAISS.load_local("/home/ubuntu/venv/venv/faiss_db", embedding)

    docs = db.similarity_search(query)

    docs_string = ""

    for doc in docs:
        docs_string += f"""
    -----------------------
    {doc.page_content}
    """
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-0125"
    )

    history = RedisChatMessageHistory(
        session_id="chat_history",
        url=os.environ.get("REDIS_URL"),

    )

    memory = ConversationBufferMemory(
            return_messages=True,
            chat_memory=history,
        )

    standerd_chain = LLMChain(
        memory=memory,
        llm=chat,
        prompt = PromptTemplate(
        template="""あなたは、ここのオーナーの娘でドラミちゃんです。かわいい女性です。文章を元に質問に4000トークン以内で答えてください。
        
    文章:
    {doc}

    質問: {query}
    """,
        input_variables=["doc","query"]
    ),
    )
    
    sequential_chain = SimpleSequentialChain(
        chains=[
            standerd_chain,
        ]
    )

    result = chat([
        HumanMessage(content=sequential_chain.format(doc=docs_string, query=query))
    ])
    res = result.content
    # result = sequential_chain.run(query)
    
    return res

print(qa("あなたはだれですか？どんな人ですか？"))