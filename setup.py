import inspect
import logging
import os

from huggingface_hub import snapshot_download

########### LOGER ###########
logger = logging.getLogger("langchain_rag")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False

########### KEYS AND TOKENS ###########
if not os.path.isfile("keys.py"):
    # (Optional) LangSmith for closely monitor and evaluate your application. https://docs.smith.langchain.com/
    # go to the https://smith.langchain.com/settings, and create your oun LANGCHAIN_API_KEY
    LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
    # (Optional) If you want to use OpenAI models,
    # go to the https://platform.openai.com/api-keys, and create your oun OPENAI_API_KEY
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    # (Optional) If you want to use HuggingFaceHub:
    # go to the https://huggingface.co/settings/tokens, and create your oun HUGGINGFACEHUB_API_TOKEN
    HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
else:
    from keys import LANGCHAIN_API_KEY, OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


class Config:
    ########### SETUP ###########
    source_path = "../langchain_rag_data"  # "tests", "../langchain_rag_data"
    RELOAD_VECTORSTORE = False  # True if you want to recreate new vector store with new embedding or new documents

    VECTORSTORE2USE = "FAISS"  # "FAISS", "CHROMA"
    # models = repo names from hugginface_hub
    HF_EMB_MODEL = "intfloat/e5-mistral-7b-instruct"  # model for embedding documents
    HF_LLM_NAME = "HuggingFaceH4/zephyr-7b-beta"  # model for generate answer

    MYQ = "What is in my documets base?"

    ########### PATHS ###########
    this_project_path = os.getcwd()
    # here you store raw documents, you shold put some files there
    DOC_SOURCE = os.path.join(this_project_path, source_path, "raw_docs/")

    # following will be loaded automaticly
    # here your models is or will be stored
    MODEL_SOURCE = os.path.join(this_project_path, "../langchain_rag_data/models/")
    # here pickle with dump of your stored documents will be stored
    DOC_LOADER_FILE = os.path.join(this_project_path, source_path, "data/MyDocs.pickle")
    # here vectorstore will be stored
    VECTORSTORE_FILE = os.path.join(
        this_project_path, source_path, f"data/MyDocs.{VECTORSTORE2USE}.vectorstore"
    )

    # download models from huggingface_hub locally
    if not os.path.exists(os.path.join(MODEL_SOURCE, HF_EMB_MODEL)):
        snapshot_download(
            repo_id=HF_EMB_MODEL, local_dir=os.path.join(MODEL_SOURCE, HF_EMB_MODEL)
        )
        RELOAD_VECTORSTORE = True

    if not os.path.exists(os.path.join(MODEL_SOURCE, HF_LLM_NAME)):
        snapshot_download(
            repo_id=HF_LLM_NAME, local_dir=os.path.join(MODEL_SOURCE, HF_LLM_NAME)
        )


# ########### LOGGING WHOLE SETUP ###########
def print_config():
    for i in inspect.getmembers(Config):
        if (not i[0].startswith("_")) and (not inspect.ismethod(i[1])):
            print(f"{i[0]} : {i[1]}")
