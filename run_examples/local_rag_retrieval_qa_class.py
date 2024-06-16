from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

from embedders.create_llm_emb_default import create_llm_emb_default
from generators.create_llm_gen_default import create_llm_gen_default
from setup import Config, logger, print_config
from tools import prompt_templates_generate
from tools.invoke_result import (
    invoke_input_context_answer,
    invoke_query_source_documents_result,
)
from vectorstores.get_vectorstore import get_vectorstore


def local_rag_retrieval_qa():
    ############## INITIAL SETUP ##############
    print_config()

    ############## EMBEDDING MODEL ##############
    # Load model for embedding documents
    logger.info(f"LLM_EMB : {Config.HF_EMB_MODEL}")
    llm_emb = create_llm_emb_default()

    ############## GENERATOR MODEL ##############
    # Load model for generating answer
    logger.info(f"LLM : {Config.HF_LLM_NAME}")
    llm_gen = create_llm_gen_default()

    ############## VECTORSTORE FOR EMBEDDINGS ##############
    # Create or load vectorstore (FAISS or Chroma)
    logger.info("VECTORSTORE")
    vectorstore = get_vectorstore(llm_emb)

    ############## RETRIEVER MODEL FROM EMBEDDING MODEL ##############
    logger.info("RETRIEVER")
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    del vectorstore

    ############## FULL RAG = RETRIEVER + GENERATOR ##############
    ############## V1 FULL RAG = RETRIEVER + GENERATOR ##############
    logger.info("Classical RETRIEVER and GENERATOR")
    # Prompt
    prompt = PromptTemplate(
        template=prompt_templates_generate.prompt_template_input_context,
        input_variables=["context", "input"],
    )
    question_answer_chain = create_stuff_documents_chain(llm_gen, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    logger.info("rag_chain.invoke")
    result = chain.invoke({"input": Config.MYQ})
    print(invoke_input_context_answer(result))

    ############## V2 FULL RAG = RETRIEVER + GENERATOR ##############
    logger.info("Classical RETRIEVER and GENERATOR with chain type")
    chain = RetrievalQA.from_chain_type(
        llm=llm_gen,
        chain_type="refine",
        retriever=retriever,
        return_source_documents=True,
    )
    logger.info("RetrievalQA.rag_chain.invoke")
    result = chain.invoke({"query": Config.MYQ})
    print(invoke_query_source_documents_result(result))


if __name__ == "__main__":
    local_rag_retrieval_qa()
