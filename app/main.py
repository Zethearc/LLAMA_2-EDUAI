# main.py
import uvicorn
from fastapi import FastAPI
from app.routes import llm_routes, embedding_routes
import warnings

# Desactivar el warning específico relacionado con TypedStorage
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

app = FastAPI()

# Agregar las rutas de LLM y embedding
app.include_router(llm_routes.router, prefix="/llm")
app.include_router(embedding_routes.router, prefix="/embedding")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
