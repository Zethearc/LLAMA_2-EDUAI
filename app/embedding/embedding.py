import os
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"] 

# Init
pinecone = PineconeClient(api_key=PINECONE_API_KEY,
                         environment=PINECONE_ENVIRONMENT)

embeddings = model
vectorstore = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever()