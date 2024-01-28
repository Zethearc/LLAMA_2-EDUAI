from fastapi import APIRouter
from embedding.embedding import embed_model

router = APIRouter()

# Ruta para realizar embedding de texto
@router.post("/embed")
async def embed(text: str):
    embeddings = embed_model.encode([text])
    return {"text": text, "embeddings": embeddings.tolist()}
