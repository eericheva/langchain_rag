import os
from operator import itemgetter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from create_vectorstore import create_vectorstore
from setup import Config, logger, print_config
from tools import prompt_templates
from tools.invoke_result import (
    invoke_generate_queries_with_origin,
    invoke_input_context_answer,
    invoke_query_source_documents_result,
    invoke_unique_docs_union_from_retriever,
)


def overview():
    # Go to the setup.py and set you tokens, key and setup params: vectorstore type, models, local paths
    print_config()

    # Load model for embedding documents
    logger.info(f"LLM_EMB : {Config.HF_EMB_MODEL}")
    llm_emb = HuggingFaceEmbeddings(
        model_name=os.path.join(Config.MODEL_SOURCE, Config.HF_EMB_MODEL),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    # Load model for generating answer
    logger.info(f"LLM : {Config.HF_LLM_NAME}")
    llm = HuggingFacePipeline.from_model_id(
        model_id=os.path.join(Config.MODEL_SOURCE, Config.HF_LLM_NAME),
        task="text-generation",
        device=-1,  # -1 stands for CPU
        pipeline_kwargs={"max_new_tokens": 512},
        model_kwargs={
            "do_sample": True,
            "top_k": 30,
            "temperature": 0.0,
            "repetition_penalty": 1.03,
            "max_length": 512,
        },
    )
    # create or load vectorstore
    vectorstore = create_vectorstore(llm_emb)

    logger.info("RETRIEVER")
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    del vectorstore

    #### V1 ####
    logger.info("Classical RETRIEVER and GENERATOR")
    # Prompt
    prompt = PromptTemplate(
        template=prompt_templates.prompt_template_input_context,
        input_variables=["context", "input"],
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    logger.info("rag_chain.invoke")
    result = chain.invoke({"input": Config.MYQ})
    print(invoke_input_context_answer(result))

    #### V2 ####
    logger.info("Classical RETRIEVER and GENERATOR with chain type")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=retriever,
        return_source_documents=True,
    )
    logger.info("RetrievalQA.rag_chain.invoke")
    result = chain.invoke({"query": Config.MYQ})
    print(invoke_query_source_documents_result(result))

    #### V3 with Runnable Sequences ####
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
        | llm
        | StrOutputParser()
    )
    invoke_generate_queries_chain = (
        {"question": itemgetter("question"), "alternatives": generate_queries_chain}
        | RunnableLambda(invoke_generate_queries_with_origin)
        | (lambda x: x.split("\n"))
    )
    # to check multiple generated questions:
    # result = invoke_generate_queries_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    # print(result)

    # Retrieval Chain for multiple alternatives to the question formulation
    retrieval_chain = (
        invoke_generate_queries_chain
        | retriever.map()
        | invoke_unique_docs_union_from_retriever
    )
    # to check list of retrieved documents
    # result = retrieval_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    # print(result)

    # Prompt for generation answer with retriever and generatin prompt
    prompt_generation = PromptTemplate(
        template=prompt_templates.prompt_template_question_context,
        input_variables=["question", "context"],
    )
    # RAG Chain
    rag_chain = (
        {
            "context": retrieval_chain,
            "question": itemgetter("question"),
            # "question_numbers": itemgetter("question_numbers"),
        }
        | prompt_generation
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    print(result)


if __name__ == "__main__":
    overview()
