import pathlib

from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory


def init_memory():
    return ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

MEMORY = init_memory()

def load_document(temp_filepath):
    ext = pathlib.Path(temp_filepath).suffix
    loaded = PyPDFLoader(temp_filepath)
    docs = loaded.load()
    return docs