import uvicorn
from fastapi import FastAPI
from routes import llm_routes, embedding_routes

app = FastAPI()

# Agregar las rutas de LLM y embedding
app.include_router(llm_routes.router, prefix="/llm")
app.include_router(embedding_routes.router, prefix="/embedding")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)