# rag.py (REPLACE your existing rag.py with this)
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
_MENU_JSON = os.path.join(os.getcwd(), "menu.json")

def load_menu_vectorstore():
    # Load documents (menu.json expected to be an array of categories)
    loader = JSONLoader(file_path=_MENU_JSON,
                        jq_schema='.[] | .category + " " + (.items[] | .name + " " + .description + " $" + (.price | tostring))',
                        text_content=False)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Build once at import time (like before)
vectorstore = load_menu_vectorstore()

def retrieve_relevant_menu(query: str, k: int = 10) -> str:
    """
    Return newline-joined page_content for the top-k matches to query.
    """
    if not query:
        return ""
    # Use the vectorstore similarity search (FAISS object supports similarity_search)
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        # Defensive fallback: don't crash the system, return readable error
        return f"Error retrieving menu: {str(e)}"
