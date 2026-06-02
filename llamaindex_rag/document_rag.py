"""LlamaIndex document RAG over Prodapt policy TXT files."""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "data", "documents")
INDEX_DIR = os.path.join(BASE_DIR, "data", "vector_index")

_index = None


def _build_or_load_index():
    global _index
    if _index is not None:
        return _index

    from llama_index.core import (
        SimpleDirectoryReader,
        VectorStoreIndex,
        StorageContext,
        load_index_from_storage,
        Settings,
    )
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        _index = load_index_from_storage(storage_ctx)
    else:
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        _index = VectorStoreIndex.from_documents(documents)
        os.makedirs(INDEX_DIR, exist_ok=True)
        _index.storage_context.persist(persist_dir=INDEX_DIR)

    return _index


def query_policy(question: str) -> str:
    """Query the policy document RAG and return a synthesized answer."""
    try:
        index = _build_or_load_index()
        engine = index.as_query_engine(similarity_top_k=3)
        response = engine.query(question)
        return str(response)
    except BrokenPipeError as e:
        return f"[PolicyRAG] Connection error - the embedding or LLM service disconnected. Try again."
    except TimeoutError as e:
        return f"[PolicyRAG] Request timed out - the LLM service took too long to respond."
    except Exception as e:
        error_msg = str(e)[:100] if str(e) else type(e).__name__
        return f"[PolicyRAG] Error: {error_msg}"
