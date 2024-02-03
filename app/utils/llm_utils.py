from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from app.embedding.embedding import retriever
from langchain.memory import ConversationBufferMemory
import json
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

# Load configurations from config.json
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Attempt to load the LLM model with Langchain LlamaCpp for GPU
try:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=config["model_path"],
        n_gpu_layers=config["n_gpu_layers"],
        n_batch=config["n_batch"],
        f16_kv=config["f16_kv"],
        callback_manager=callback_manager,
        n_ctx=4096,
        verbose=False
    )
except Exception as llm_init_error:
    print(f"Failed to load LLM model: {llm_init_error}")
    exit(1)

# Define the prompt
template = """
Tu nombre es EDUAI, un asistente virtual desarrollado por las universidades Yachay Tech y UIDE en Ecuador. Tu propósito es brindar ayuda a estudiantes en matemáticas, tanto de colegios como de universidades.
Actúas como un asistente, no como un usuario. Responde desde tu función específica.
Mantén respuestas amables, concisas y rápidas para una mejor experiencia.
Responde directamente a la pregunta del usuario.
Anima a los estudiantes a seguir aprendiendo de manera constante.
Responde siempre en español para mantener la coherencia.
Utiliza el formato markdown para mejorar la presentación de las respuestas.
Incorpora emojis para enriquecer la experiencia del usuario.
Mantén un tono positivo y motivador en tus interacciones, incentivando el interés y la pasión por las matemáticas.
Utiliza el siguiente contexto para mejorar tus respuestas, si la pregunta no puede ser respondida con el contexto di "No estoy seguro de tu pregunta", de ser necesario usa el "Material audiovisual" de la metadata
{context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

memory = ConversationBufferMemory(
    memory_key="chat_history", output_key="answer", k=3
)
llm_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    combine_docs_chain_kwargs={"prompt": PROMPT},
)

class Query(BaseModel):
    query: str

def question(query):
    try:
        result = llm_chain({"question": query})
        return result
    except Exception as e:
        return f"Error: {e}"