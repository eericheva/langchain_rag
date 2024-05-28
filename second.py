import os
from operator import itemgetter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import prompt_templates
from create_vectorstore import create_vectorstore
from setup import Config, logger, print_config
from tools.documents_manipulations import get_unique_union


def second():
    print_config()

    logger.info(f"LLM_EMB : {Config.HF_EMB_MODEL}")
    llm_emb = HuggingFaceEmbeddings(
        model_name=os.path.join(Config.MODEL_SOURCE, Config.HF_EMB_MODEL),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

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

    logger.info("VECTORSTORE")
    vectorstore = create_vectorstore(llm_emb)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    del vectorstore

    logger.info("Multi Query RETRIEVER")

    # # Prompt
    prompt = PromptTemplate(
        template=prompt_templates.prompt_multi_query,
        input_variables=["question", "number_questions"],
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
            "number_questions": itemgetter("number_questions"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke({"question": Config.MYQ, "number_questions": 2})
    print(result)
    # print(invoke_input_context_answer(result))


if __name__ == "__main__":
    second()
