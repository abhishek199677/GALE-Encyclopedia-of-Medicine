import os
# MUST be before other imports to prevent the libiomp5md.dll crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from pinecone import Pinecone, ServerlessSpec

# Initialize Flask
app = Flask(__name__)

# Load Environment Variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 1. Setup Embeddings
embeddings = download_hugging_face_embeddings()

# 2. Setup Pinecone Index
index_name = "encyclopedia-medical-chatbot"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Prevent 404: Check if index exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384, # Must match all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# 3. Connect to Vector Store
docsearch = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

# 4. Define Retrieval Chain
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chatModel = ChatOpenAI(model="gpt-4o-mini") # Use gpt-4o-mini for better speed/cost

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- ROUTES ---

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"]) # Standardized to POST for messages
def chat():
    msg = request.form["msg"]
    print(f"User Input: {msg}")
    
    # RAG process: Search index -> Combine with prompt -> Generate Answer
    response = rag_chain.invoke({"input": msg})
    
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)