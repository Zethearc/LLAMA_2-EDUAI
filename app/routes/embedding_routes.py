from fastapi import APIRouter
from pydantic import BaseModel
from app.embedding.embedding import retriever

router = APIRouter()

# Definición del modelo de datos para la consulta
class EmbeddingQuery(BaseModel):
    sentence: str

# Ruta para obtener el embedding de una oración
@router.post("/embed")
async def get_embedding(query: EmbeddingQuery):
    try:
        embedding = retriever.encode([query.sentence])[0].tolist()
        return {"embedding": embedding}
    except Exception as e:
        return {"error": f"Error processing embedding: {e}"}