from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from embedders.create_llm_emb_default import create_llm_emb_default
from generators.create_llm_gen_default import create_llm_gen_default
from setup import Config, logger, print_config
from tools import prompt_templates
from tools.invoke_result import (
    invoke_generate_queries_with_origin,
    invoke_unique_docs_union_from_retriever,
)
from vectorstores.get_vectorstore import get_vectorstore


def local_rag_multi_query():
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
    logger.info("Multi Query RETRIEVER and RAG Chain")
    ############## MULTIPLE QUERY CHAIN ##############
    # Generate multiple alternatives to the question formulation
    # Prompt for multiple alternatives to the question formulation
    prompt_multi_query = PromptTemplate(
        template=prompt_templates.prompt_multi_query,
        input_variables=["question", "number_questions"],
    )
    # Chain for generating multiple alternatives to the question formulation
    generate_queries_chain = (
        {
            "question": itemgetter("question"),
            "question_numbers": itemgetter("question_numbers"),
        }
        | prompt_multi_query
        | llm_gen
        | StrOutputParser()
    )
    invoke_generate_queries_chain = (
        {"question": itemgetter("question"), "alternatives": generate_queries_chain}
        | RunnableLambda(invoke_generate_queries_with_origin)
        | (lambda x: x.split("\n"))
    )
    # V2 - you can create chain in another function and call it here, leave all other next code as it is:
    # from chains.multiple_queries_chain import multiple_queries_chain
    # invoke_generate_queries_chain = multiple_queries_chain(llm_gen)

    # to check multiple generated questions:
    # result = invoke_generate_queries_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    # print(result)

    ############## RETRIEVAL CHAIN ##############
    # Retrieval Chain for multiple alternatives to the question formulation
    retrieval_chain = (
        invoke_generate_queries_chain
        | retriever.map()
        | invoke_unique_docs_union_from_retriever
    )
    # to check list of retrieved documents
    # result = retrieval_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    # print(result)

    ############## GENERATOR CHAIN ##############
    # Prompt for generation answer with retriever and generation prompt
    prompt_generation = PromptTemplate(
        template=prompt_templates.prompt_template_question_context,
        input_variables=["question", "context"],
    )
    # RAG Chain
    rag_chain = (
        {
            "context": retrieval_chain,
            "question": itemgetter("question"),
        }
        | prompt_generation
        | llm_gen
        | StrOutputParser()
    )

    ############## RUN ALL CHAINS ##############
    result = rag_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    print(result)


if __name__ == "__main__":
    local_rag_multi_query()
