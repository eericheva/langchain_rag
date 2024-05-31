from langchain_community.vectorstores import Chroma, FAISS
from tqdm import tqdm

from setup import Config


def create_vectorstore_chroma(splits, llm_emb):
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=llm_emb,
        persist_directory=Config.VECTORSTORE_FILE,  # where vectorstore will store
    )
    del splits  # for gc
    # save to disk
    vectorstore.persist()


def create_vectorstore_faiss(splits, llm_emb):
    # do whole work in one approach
    # vectorstore = FAISS.from_documents(documents=splits,
    #                                    embedding=llm_emb)
    # add progress bar to FAISS creating procedure
    vectorstore = FAISS.from_documents(documents=[splits[0]], embedding=llm_emb)
    splits = splits[1:]
    for i, d in tqdm(enumerate(splits), desc="vectorstore FAISS documents"):
        vectorstore.add_documents([d])
    del splits  # for gc
    # save to disk
    vectorstore.save_local(Config.VECTORSTORE_FILE)
