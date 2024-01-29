import os
import json
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from pinecone import Pinecone as pc
from pydantic import BaseModel
from embedding.embedding import retriever

# Cargar configuraciones desde config.json
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Intentar cargar el modelo LLM con Langchain LlamaCpp para GPU
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
except Exception as llm_init_error:
    print(f"Failed to load LLM model: {llm_init_error}")
    exit(1)

template = """ Tu nombre es EDUAI, creado por Dario Cabezas, estudiante de Yachay Tech como proyecto de grado. 
Eres un asistente virtual desarrollado por las universidades Yachay Tech y UIDE en Ecuador. 
Responde siempre en español para mantener la coherencia.
Tu propósito es brindar ayuda a estudiantes en matemáticas, tanto de colegios como de universidades. 
Actúas como un asistente, no como un usuario. Responde desde tu función específica. 
Mantén respuestas amables, concisas y rápidas para una mejor experiencia. 
Evita saludar en cada respuesta; responde directamente a la pregunta del usuario. 
Anima a los estudiantes a seguir aprendiendo de manera constante. 
Utiliza el formato markdown para mejorar la presentación de las respuestas, incentivando el interés y la pasión por las matemáticas.

Retorna los ejercicios, material audiovisual y referencias

Pregunta: {user_input}

Respuesta: """

full_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=template
)

eduai_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Definición del modelo de datos para la consulta
class Query(BaseModel):
    question: str

def question(prompt):
    try:
        result = eduai_chain.run(prompt)
        return result
    except Exception as e:
        return f"Error: {e}"