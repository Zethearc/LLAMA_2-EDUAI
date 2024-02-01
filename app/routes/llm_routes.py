from fastapi import APIRouter
from app.llm_chain.llm_chain import llm_only_chain, llm_prompt_chain, llm_prompt_memory_chain, llm_prompt_memory_rag_chain

router = APIRouter()

# Ruta para obtener la versión y la ruta del modelo
@router.get("/")
async def version():
    return {"version": config['version'], "model_path": config['model_path']}
    
# Ruta para obtener resultados de la cadena LLM Only
@router.post("/llm_only")
async def llm_only_route(query: str):
    result = llm_only_chain.invoke(input=query)
    return {"result": result}

# Ruta para obtener resultados de la cadena LLM with Prompt
@router.post("/llm_prompt")
async def llm_prompt_route(query: str):
    result = llm_prompt_chain.invoke(input=query)
    return {"result": result}

# Ruta para obtener resultados de la cadena LLM with Prompt and Memory
@router.post("/llm_prompt_memory")
async def llm_prompt_memory_route(query: str):
    result = llm_prompt_memory_chain.invoke(input=query)
    return {"result": result}

# Ruta para obtener resultados de la cadena LLM with Prompt, Memory, and RAG
@router.post("/llm_prompt_memory_rag")
async def llm_prompt_memory_rag_route(query: str):
    result = llm_prompt_memory_rag_chain.invoke(input=query)
    return {"result": result}