from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from setup import Config, logger
from tools import prompt_templates


def simple_retriever_generator_chain1(retriever, llm_gen):
    ############## FULL RAG = RETRIEVER + GENERATOR ##############
    logger.info("Simple RETRIEVER and RAG Chain")
    ############## RETRIEVAL CHAIN ##############
    # Retrieval Chain for multiple alternatives to the question formulation
    retrieval_chain = (
        itemgetter("question")
        # retriever.invoke() takes str as input, so we need to extract "question" key from input to
        # retrieval_chain.invoke({}) dict as str
        | retriever
    )
    # to check list of retrieved documents
    # result = retrieval_chain.invoke({"question": Config.MYQ})
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
    result = rag_chain.invoke({"question": Config.MYQ})
    print(result)
    return rag_chain


def simple_retriever_generator_chain2(retriever, llm_gen):
    ############## FULL RAG = RETRIEVER + GENERATOR ##############
    logger.info("Simple RETRIEVER and RAG Chain")
    ############## NO RETRIEVAL CHAIN ##############
    # No additional retrieval chain, just the save retriever (FAISS().as_retriever() or Chroma().as_retriever())
    # to check list of retrieved documents:
    # result = retriever.invoke(Config.MYQ)
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
            "context": itemgetter("question") | retriever,  # retrieval chain is here
            "question": itemgetter("question"),
        }
        | prompt_generation
        | llm_gen
        | StrOutputParser()
    )
    ############## RUN ALL CHAINS ##############
    result = rag_chain.invoke({"question": Config.MYQ})
    print(result)
    return rag_chain
