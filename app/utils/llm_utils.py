from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from app.embedding.embedding import retriever
import json
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

# Cargar configuraciones desde config.json
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Intentar cargar el modelo LLM con Langchain LlamaCpp para GPU
try:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=config["model_path"],
        n_gpu_layers=config["n_gpu_layers"],
        n_batch=config["n_batch"],
        f16_kv=config["f16_kv"],
        callback_manager=callback_manager,
        verbose=False
    )
except Exception as llm_init_error:
    print(f"Failed to load LLM model: {llm_init_error}")
    exit(1)

# Define the prompt
template = """
Tu nombre es EDUAI, creado por Dario Cabezas, estudiante de Yachay Tech como proyecto de grado. 
Eres un asistente virtual desarrollado por las universidades Yachay Tech y UIDE en Ecuador. 
Responde siempre en español para mantener la coherencia.
Tu propósito es brindar ayuda a estudiantes en matemáticas, tanto de colegios como de universidades. 
Actúas como un asistente, no como un usuario. Responde desde tu función específica. 
Mantén respuestas amables, concisas y rápidas para una mejor experiencia. 
Evita saludar en cada respuesta; responde directamente a la pregunta del usuario. 
Anima a los estudiantes a seguir aprendiendo de manera constante. 
{question}
"""
prompt = PromptTemplate.from_template(template)

# Define the LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

# Define the Query model
class Query(BaseModel):
    question: str

def question(query):
    try:
        result = llm_chain.run(question=query)
        return result
    except Exception as e:
        return f"Error: {e}"