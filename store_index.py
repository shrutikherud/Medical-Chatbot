from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import time


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-bot"

pc.create_index(
    name = index_name,
    dimension = 1024,
    metric = "cosine",
    spec = ServerlessSpec(
        cloud = "aws",
        region = "us-east-1"
    )
)


# Batching logic to avoid hitting Cohere rate limits
batch_size = 25
total_batches = (len(text_chunks) + batch_size - 1) // batch_size

print(f"🔁 Total Chunks: {len(text_chunks)} | Batch Size: {batch_size} | Total Batches: {total_batches}")

for i in range(0, len(text_chunks), batch_size):
    batch = text_chunks[i:i + batch_size]
    try:
        PineconeVectorStore.from_documents(
            documents=batch,
            index_name=index_name,
            embedding=embeddings,
        )
        print(f"✅ Uploaded batch {i // batch_size + 1}/{total_batches}")
    except Exception as e:
        print(f"❌ Error in batch {i // batch_size + 1}: {e}")
    time.sleep(6)  # Delay to avoid exceeding token rate limit

print("🎉 All batches processed. Check Pinecone for updated index.")
