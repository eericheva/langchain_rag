from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from embedders.create_llm_emb_default import create_llm_emb_default
from generators.create_llm_gen_default import create_llm_gen_default
from setup import Config, logger, print_config
from tools import prompt_templates
from vectorstores.get_vectorstore import get_vectorstore


def local_rag_simple():
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
    logger.info("Simple RETRIEVER and RAG Chain")
    ############## RETRIEVAL CHAIN ##############
    # Retrieval Chain for multiple alternatives to the question formulation
    # V1
    retrieval_chain = (
        itemgetter("question")
        # retriever.invoke() takes str as input, so we need to extract "question" key from input to
        # retrieval_chain.invoke({}) dict as str
        | retriever
    )
    # to check list of retrieved documents
    # result = retrieval_chain.invoke({"question": Config.MYQ})
    # print(result)
    # OR for variant2:
    # result = retriever.invoke(Config.MYQ)

    ############## GENERATOR CHAIN ##############
    # Prompt for generation answer with retriever and generation prompt
    prompt_generation = PromptTemplate(
        template=prompt_templates.prompt_template_question_context,
        input_variables=["question", "context"],
    )
    # RAG Chain
    rag_chain = (
        {
            # V1
            "context": retrieval_chain,
            # OR for variant2:
            # "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt_generation
        | llm_gen
        | StrOutputParser()
    )
    # V3 - you can create chain in another function and call it here, leave all other next code as it is:
    # from chains.combine_simple_RAG_chains import simple_retriever_generator_chain1, simple_retriever_generator_chain2
    # rag_chain = simple_retriever_generator_chain1(retriever, llm_gen)
    # rag_chain = simple_retriever_generator_chain2(retriever, llm_gen)

    ############## RUN ALL CHAINS ##############
    result = rag_chain.invoke({"question": Config.MYQ})
    print(result)


if __name__ == "__main__":
    local_rag_simple()
