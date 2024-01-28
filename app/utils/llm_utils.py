# my_app/app/utils/llm_utils.py
import os
import json
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Pinecone
from pydantic import BaseModel
from pinecone import Pinecone

# Cargar configuraciones desde config.json
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Inicializar Pinecone con try-except
try:
    pinecone = Pinecone(api_key=os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY',
                        environment=os.environ.get('PINECONE_ENVIRONMENT') or 'PINECONE_ENV')
except Exception as pinecone_init_error:
    print(f"Failed to initialize Pinecone: {pinecone_init_error}")
    exit(1)

# Intenta cargar el modelo con Langchain LlamaCpp para GPU
try:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    LLM = LlamaCpp(
        model_path=config["model_path"],
        n_gpu_layers=config["n_gpu_layers"],
        n_batch=config["n_batch"],
        f16_kv=config["f16_kv"],
        callback_manager=callback_manager,
        verbose=True
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# Definición del modelo de datos para la consulta
class Query(BaseModel):
    question: str

# Definición del modelo de datos para la respuesta
class ModelResponse(BaseModel):
    result: str

def question(prompt):
    try:
        # Consulta Pinecone para obtener posibles respuestas
        search_results = pinecone.query(index_name="tu_indice_pinecone", query_vector=prompt)

        # Si hay resultados, selecciona el mejor y utiliza el modelo para generar la respuesta
        if search_results["total"] > 0:
            best_result = search_results["results"][0]
            full_prompt = f"{config['init_prompt']}\n{config['q_prompt']} {best_result['text']}\n{config['a_prompt']}"
            output = LLM.invoke(full_prompt)
            result = output["text"]
            return result
        else:
            # Manejar el caso en que no haya resultados de Pinecone
            return "No se encontraron resultados relevantes."
    except Exception as e:
        return f"Error: {e}"
