# my_app/app/routes/llm_routes.py
from fastapi import APIRouter
from utils.llm_utils import Query, question, config

router = APIRouter()

# Ruta para obtener la versión y la ruta del modelo
@router.get("/")
async def version():
    return {"version": config['version'], "model_path": config['model_path']}

# Ruta para realizar consultas al modelo
@router.post("/query")
async def query(query: Query):
    result = question(query.question)
    return {"result": result}