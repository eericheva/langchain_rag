import os
import pickle

from langchain_community.document_loaders import pdf
from langchain_community.vectorstores import Chroma, FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from setup import Config, logger
from vectorstores import create_vectorstore


def get_vectorstore(llm_emb):
    vectorstore = []
    if Config.RELOAD_VECTORSTORE:
        # if need to create new vectorstore
        logger.info("#### RELOAD_VECTORSTORE ####")

        # Load Documents
        docs = collect_documents()
        # split documents to chunks, retriever will search through embedded chunks, not whole documents
        logger.info("Split")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200  # num of characters in single chunk
        )
        splits = text_splitter.split_documents(docs)
        del docs  # for gc
        logger.info(f"Num of splits : {len(splits)}")

        if Config.VECTORSTORE2USE == "CHROMA":
            # https://python.langchain.com/v0.1/docs/integrations/vectorstores/chroma/
            # for Num of splits : 700 will take Time : ~60min
            logger.info("vectorstore Chroma")
            create_vectorstore.create_vectorstore_chroma(splits, llm_emb)

        if Config.VECTORSTORE2USE == "FAISS":
            # https://python.langchain.com/v0.1/docs/integrations/vectorstores/faiss/
            # for Num of splits : 700 will take Time : ~60min
            logger.info("vectorstore FAISS")
            create_vectorstore.create_vectorstore_faiss(splits, llm_emb)

    if Config.VECTORSTORE2USE == "CHROMA":
        logger.info("vectorstore Chroma from dump")
        # load from disk
        vectorstore = Chroma(
            persist_directory=Config.VECTORSTORE_FILE, embedding_function=llm_emb
        )

    if Config.VECTORSTORE2USE == "FAISS":
        logger.info("vectorstore FAISS from dump")
        # load from disk
        vectorstore = FAISS.load_local(
            Config.VECTORSTORE_FILE,
            embeddings=llm_emb,
            allow_dangerous_deserialization=True,  # True for data with loading a pickle file.
        )
    return vectorstore


def collect_documents():
    docs = []
    if not os.path.exists(Config.DOC_LOADER_FILE):
        # if there is no dump pickle file with docs
        logger.info("#### LOAD RAW DOCS ####")
        for file_name in os.listdir(Config.DOC_SOURCE):
            fp = os.path.join(Config.DOC_SOURCE, file_name)

            docs += pdf.PyPDFLoader(fp).load()

        logger.info(f"dump raw docs to {Config.DOC_LOADER_FILE} file")
        pickle.dump(docs, open(Config.DOC_LOADER_FILE, "wb"))

    logger.info(f"load raw docs from {Config.DOC_LOADER_FILE} file")
    docs = pickle.load(open(Config.DOC_LOADER_FILE, "rb"))
    return docs
