import os
from operator import itemgetter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import prompt_templates
from create_vectorstore import create_vectorstore
from setup import Config, logger, print_config
from tools.documents_manipulations import get_unique_union
from tools.invoke_result import (
    invoke_input_context_answer,
    invoke_query_source_documents_result,
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

    #### V2 ####
    logger.info("Multi Query RETRIEVER")
    # # Prompt
    prompt = PromptTemplate(
        template=prompt_templates.prompt_multi_query,
        input_variables=["question", "question_numbers"],
    )
    generate_queries = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    # Retrieve
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    # docs = retrieval_chain.invoke({"question": Config.MYQ})

    logger.info("RAG CHAIN")
    rag_chain = (
        {
            "context": retrieval_chain,
            "question": itemgetter("question"),
            "question_numbers": itemgetter("question_numbers"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    print(result)


if __name__ == "__main__":
    overview()
