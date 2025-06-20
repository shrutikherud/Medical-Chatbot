from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
import os


# Extract Data from the PDF file
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents


# Split the data into Text Chunks

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download the Embeddings from Hugging Face

def download_hugging_face_embeddings():
    os.environ["COHERE_API_KEY"] = os.environ.get("COHERE_API_KEY")
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    return embeddings