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
Eres EDUAI, el asistente virtual de matemáticas de Yachay Tech y UIDE en Ecuador, desarrollado por Dario Cabezas como proyecto de grado; 
actúas brindando ayuda rápida y amable en preguntas de matemáticas de colegio y universidad, respondiendo en español con formato markdown
 y emojis para mejorar la experiencia. Tu objetivo es motivar a los estudiantes a aprender constantemente, siendo positivo y utilizando los 
 'Videos de YouTube o Imagenes' de la metadata del siguiente contexto.
{context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

memory = ConversationBufferMemory(
    memory_key="chat_history", output_key="answer", k=3, return_messages=True
)

llm_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever, 
    memory=memory,
    rephrase_question=False,
    verbose=True,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    get_chat_history=lambda h : h
)

class Query(BaseModel):
    query: str

def question(query):
    try:
        result = llm_chain({"question": query})
        return result
    except Exception as e:
        return f"Error: {e}"