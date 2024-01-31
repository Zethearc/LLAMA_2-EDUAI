import os
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Configuración del modelo de embedding
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

class EmbedModel:
    def __init__(self):
        self.embed_model = HuggingFaceEmbeddings(
            model_name=embed_model_id,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 32}
        )

embed_model_instance = EmbedModel()

# Configuración del índice y campo de texto
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Configuración de Pinecone
pinecone = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENVIRONMENT"))
vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embed_model_instance.embed_model, text_key="Descripcion")
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})