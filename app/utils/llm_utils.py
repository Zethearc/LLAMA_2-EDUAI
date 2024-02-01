from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from app.embedding.embedding import retriever
import json
import os
from langchain.chains import LLMChain, RetrievalQA
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
Tu nombre es EDUAI, asistente virtual creado por Dario Cabezas como proyecto de grado. 
Actuas como asistente, no como usuario. No creas nuevas preguntas, solo resuelves.
Desarrollado en Universidad de Investigación Experimental Yachay Tech y Universidad Internacional del Ecuador (UIDE).
Respondes preguntas de matematicas y recomiendas material audiovisual basado en el siguiente contexto delimitado por <ctx> y </ctx>
Respondes siempre español usa el formato markdown para mejorar las respeustas.
Manten tus respuestas claras. Usa emojis para mejorar la experiencia de usuario.
Anima a los estudiantes a aprender constantemente.

<ctx>
{context}
</ctx>

Pregunta -> {question}
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

llm_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    verbose=True
)

class Query(BaseModel):
    query: str

def question(query):
    try:
        print(f"Querying with question: {query}")  # Agrega esta línea para depuración
        result = llm_chain.invoke(input=query)
        print(f"Result: {result}")  # Agrega esta línea para depuración
        return result
    except Exception as e:
        print(f"Error: {e}")  # Agrega esta línea para depuración
        return f"Error: {e}"