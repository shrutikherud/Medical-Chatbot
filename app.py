from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import ChatCohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from src.prompt import *
import os

# Flask app initialization
app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# Load Cohere embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone Index
index_name = "medical-bot"
docSearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retriever setup
retriever = docSearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Load Cohere LLM
llm = ChatCohere(model="command-r", temperature=0.4, max_tokens=500)

# Memory setup
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# RAG Chain with memory
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False,
    verbose=False
)

# Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    response = rag_chain.invoke({"question": msg})
    print("Response:", response["answer"])
    
    return str(response["answer"])

# Main
if __name__ == '__main__':
    app.run(debug=True)
