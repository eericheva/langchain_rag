import os
import pickle

from langchain_community.document_loaders import pdf
from langchain_community.vectorstores import Chroma, FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from setup import Config, logger


def create_vectorstore(llm_emb):
    # Load Documents
    if Config.RELOAD_VECTORSTORE:
        logger.info("#### RELOAD_VECTORSTORE ####")
        if not os.path.exists(Config.DOC_LOADER_FILE):
            logger.info("#### LOAD RAW DOCS ####")
            docs = []
            for file_name in os.listdir(Config.DOC_SOURCE):
                fp = os.path.join(Config.DOC_SOURCE, file_name)

                docs += pdf.PyPDFLoader(fp).load()

            logger.info(f"dump raw docs to {Config.DOC_LOADER_FILE} file")
            pickle.dump(docs, open(Config.DOC_LOADER_FILE, "wb"))

        logger.info(f"load raw docs from {Config.DOC_LOADER_FILE} file")
        docs = pickle.load(open(Config.DOC_LOADER_FILE, "rb"))

        logger.info("Split")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        del docs
        logger.info(f"Split : {len(splits)}")

        if Config.VECTORSTORE2USE == "CHROMA":
            logger.info("vectorstore Chroma : Splits : 700 Time : ~60min")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=llm_emb,
                persist_directory=Config.VECTORSTORE_FILE,
            )
            del splits
            # save to disk
            vectorstore.persist()

        if Config.VECTORSTORE2USE == "FAISS":
            logger.info("vectorstore FAISS : Splits : 700 Time : ~60min")
            # vectorstore = FAISS.from_documents(documents=splits,
            #                                    embedding=llm_emb)  # # 11 min Split : 148
            vectorstore = FAISS.from_documents(documents=[splits[0]], embedding=llm_emb)
            splits = splits[1:]
            for i, d in tqdm(enumerate(splits), desc="vectorstore FAISS documents"):
                vectorstore.add_documents([d])
            del splits
            # save to disk
            vectorstore.save_local(Config.VECTORSTORE_FILE)

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
            allow_dangerous_deserialization=True,
        )
    return vectorstore
