# my_app/app/routes/llm_routes.py
from fastapi import APIRouter, HTTPException
from app.utils.llm_utils import Query, question, config  # Asegúrate de importar HTTPException

router = APIRouter()

# Ruta para obtener la versión y la ruta del modelo
@router.get("/")
async def version():
    return {"version": config['version'], "model_path": config['model_path']}
    
# Ruta para realizar consultas al modelo
@router.post("/query")
async def query(query: Query):
    try:
        result = question(query.query)  # Cambia query.question a query.query
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")