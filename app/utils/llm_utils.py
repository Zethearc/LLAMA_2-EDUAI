import json
import os

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
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

class ChainManager:
    def __init__(self, llm, retriever, prompt, memory=None, use_rag=False):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        self.memory = memory
        self.use_rag = use_rag

    def create_chain(self):
        if self.use_rag:
            return self._create_llm_prompt_memory_rag_chain()
        elif self.memory:
            return self._create_llm_prompt_memory_chain()
        elif self.prompt:
            return self._create_llm_prompt_chain()
        else:
            return self._create_llm_only_chain()

    def _create_llm_only_chain(self):
        return LLMChain(llm=self.llm)

    def _create_llm_prompt_chain(self):
        return LLMChain(llm=self.llm, prompt=self.prompt)

    def _create_llm_prompt_rag_chain(self):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt},
            verbose=True
        )

    def _create_llm_prompt_memory_chain(self):
        return LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    def _create_llm_prompt_memory_rag_chain(self):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt, "memory": self.memory},
            verbose=True
        )

