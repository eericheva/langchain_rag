from embedders.create_llm_emb_default import create_llm_emb_default
from run_examples.local_rag_chain_multi_query import local_rag_multi_query
from setup import Config, logger
from vectorstores.get_vectorstore import get_vectorstore


def rag_qa():
    rag_chain = local_rag_multi_query()
    result = rag_chain.invoke({"question": Config.MYQ, "question_numbers": 5})
    print(result)


def create_vs():
    Config.RELOAD_VECTORSTORE = True
    logger.info(f"VECTORSTORE {Config.VECTORSTORE_FILE}")
    llm_emb = create_llm_emb_default()
    vectorstore = get_vectorstore(llm_emb)


if __name__ == "__main__":
    create_vs()
    # rag_qa()
