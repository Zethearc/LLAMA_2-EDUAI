
from app.utils.llm_utils import ChainManager
from app.embedding.embedding import retriever
from langchain.memory import ConversationBufferMemory
import os

template_file_path = os.path.join(os.path.dirname(__file__), "prompts", "rag_template.txt")
with open(template_file_path, "r") as template_file:
    template = template_file.read()

# Crear instancias de cadenas
llm_only_chain = ChainManager(llm, retriever=None, prompt=None, memory=None, use_rag=False).create_chain()
llm_prompt_chain = ChainManager(llm, retriever=None, prompt=prompt, memory=None, use_rag=False).create_chain()
llm_prompt_memory_chain = ChainManager(llm, retriever=None, prompt=prompt, memory=ConversationBufferMemory(), use_rag=False).create_chain()
llm_prompt_memory_rag_chain = ChainManager(llm, retriever=retriever, prompt=prompt, memory=ConversationBufferMemory(), use_rag=True).create_chain()