import uvicorn
from fastapi import FastAPI
from app.routes import llm_routes

app = FastAPI()

# Agregar las rutas de LLM y embedding
app.include_router(llm_routes.router, prefix="/llm")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)