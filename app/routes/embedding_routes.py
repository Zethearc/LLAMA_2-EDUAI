# app/routes/embedding_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.embedding.embedding import vectorstore, perform_similarity_search

router = APIRouter()

# Definición del modelo de datos para la consulta
class EmbeddingSentence(BaseModel):
    sentence: str

# Ruta para obtener el embedding de una oración
@router.post("/embed")
def get_embedding(sentence_data: EmbeddingSentence):
    try:
        # Realizar la búsqueda de similitud utilizando la función
        responses = perform_similarity_search(sentence_data.sentence, vectorstore, top_k=2)

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
