# rag.py (Improved RAG with Better Splitting and Embeddings)

from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Better splitter for natural chunks

def load_menu_vectorstore():
    loader = JSONLoader(file_path='menu.json', jq_schema='.[] | .category + " " + (.items[] | .name + " " + .description + " $" + (.price | tostring))', text_content=False)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)  # Optimized for menu items
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")  # Better model for semantics
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = load_menu_vectorstore()

def retrieve_relevant_menu(query):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})  # Increased k, similarity for better relevance
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])