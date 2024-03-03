import os
import openai
from dotenv import load_dotenv

from langchain.chains import ConversationChain
from langchain.memory import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# ###############################################################

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
# FAISS_DB_DIR = os.environ["FAISS_DB_DIR"]
FAISS_DB_DIR = "venv/apps/faiss_db"

# ###############################################################

chat = ChatOpenAI(
    model="gpt-4-0125-preview"
)

history = RedisChatMessageHistory(
    session_id="chat_history",
    url=os.environ.get("REDIS_URL"),
)

memory = ConversationBufferMemory(
    return_messages=True,
    chat_memory=history,
)

chain = ConversationChain(
    memory=memory,
    llm=chat,
)

message = "茶碗蒸しを作るのに必要な食材を教えて"

result = chain(
    message
)

print(result)