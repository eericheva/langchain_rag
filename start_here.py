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
        # https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface
        # .HuggingFaceEmbeddings.html
        model_name=os.path.join(Config.MODEL_SOURCE, Config.HF_EMB_MODEL),
        model_kwargs={
            # full list of parameters for this section with explanation:
            # https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
            # #sentence_transformers.SentenceTransformer
            "device": "cpu"
        },
        encode_kwargs={
            # full list of parameters for this section with explanation:
            # https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
            # #sentence_transformers.SentenceTransformer.encode
            "normalize_embeddings": False
        },
    )
    # Load model for generating answer
    logger.info(f"LLM : {Config.HF_LLM_NAME}")
    llm = HuggingFacePipeline.from_model_id(
        # https://api.python.langchain.com/en/latest/llms/langchain_community.llms.huggingface_pipeline
        # .HuggingFacePipeline.html
        model_id=os.path.join(Config.MODEL_SOURCE, Config.HF_LLM_NAME),
        task="text-generation",
        device=-1,  # -1 stands for CPU
        pipeline_kwargs={
            "max_new_tokens": 512,  # How long could be generated answer
            "return_full_text": False,
        },
        # "return_full_text": True if you want to return within generation answer all prompts, contexts and other
        # serving instrumentals
        model_kwargs={
            # full list of parameters for this section with explanation:
            # https://huggingface.co/docs/transformers/en/main_classes/text_generation
            # Note: some of them (depends on the specific model) should go to the pipeline_kwargs dict
            "do_sample": True,
            "top_k": 10,
            "temperature": 0.0,
            "repetition_penalty": 1.03,
            "max_length": 20,
        },
    )
    # create or load vectorstore (FAISS or Chroma)
    vectorstore = create_vectorstore(llm_emb)

    logger.info("RETRIEVER")
    retriever = vectorstore.as_retriever(
        # full list of parameters for this section with explanation:
        # https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html
        # #langchain_chroma.vectorstores.Chroma.as_retriever
        search_type="similarity",
        search_kwargs={"k": 4},
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
        # you can create any imagined prompt as template.
        # Note: if your prompt refers to some variables in formatting type, you should provide these variables
        # names to input_variables parameter
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
    # The generate_queries_chain is a pipeline built using LangChain's RunnableSequence. How this work?
    # Long story short: output from the previous RunnableSequence element is passed as input to the next
    # RunnableSequence element. The output type of the previous element must be compatible with the input type of the
    # next element.
    # Input Data: The input to the chain is a dictionary that contains at least two keys: "question" and
    # "question_numbers". These values are extracted from the input dictionary using the itemgetter function.
    # Prompt for Multiple Queries: The extracted "question" and "question_numbers" are passed to the
    # prompt_multi_query function. This function likely formats these inputs into a specific prompt template or
    # prepares them for the language model (LLM).
    # Language Model (LLM): The formatted prompt is then passed to the language model (llm). The LLM generates a
    # response based on the input prompt.
    # Output Parsing: The response from the LLM is parsed using StrOutputParser(). This parser converts the raw
    # output string from the LLM into a more structured format.
    # Output: The final output of the chain is the structured response from the LLM, after being parsed by
    # StrOutputParser().
    # This is alternative to:
    # input_dict = {
    #     "question": "What is the capital of France?",
    #     "question_numbers": 1
    # }
    # Extract Question and Question Numbers:
    # question = itemgetter("question")(input_dict)
    # question_numbers = itemgetter("question_numbers")(input_dict)
    # formatted_prompt = prompt_multi_query(question, question_numbers) # Create Prompt for Multiple Queries:
    # llm_response = llm(formatted_prompt) # Generate Response Using LLM:
    # parsed_response = StrOutputParser()(llm_response) # Parse the LLM Response: Output -> parsed_response
    invoke_generate_queries_chain = (
        # Here we need to pass as input to the invoke_generate_queries_with_origin 2 variables:
        # output from the last step of previous chain (generate_queries_chain),
        # as well as additional var (question). Create a dict with them as input to the RunnableLambda
        # We also pass the name for the output of previous chain (generate_queries_chain) as key of the dict
        {"question": itemgetter("question"), "alternatives": generate_queries_chain}
        # To enable function invoke_generate_queries_with_origin to use this dict as input, it should be
        # RunnableLambda
        | RunnableLambda(invoke_generate_queries_with_origin)
        | (lambda x: x.split("\n"))
    )
    # to check multiple generated questions:
    # result = invoke_generate_queries_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    # print(result)

    # Retrieval Chain for multiple alternatives to the question formulation
    retrieval_chain = (
        # We can attach previous chains as input to the next chains:
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
        }
        # Here again: since prompt_generation takes as input 2 variables with names: context and question,
        # we assign these name to the variables as dict keys.
        # "context" will take meaning from the output of retrieval_chain
        # "question" will take meaning from calling this.invoke() with provided "question" key
        | prompt_generation
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    print(result)


if __name__ == "__main__":
    overview()
