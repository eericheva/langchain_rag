import os
from operator import itemgetter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from setup import Config, logger, print_config
from tools import prompt_templates_generate, prompt_templates_retrieve
from tools.invoke_result import (
    invoke_generate_queries_with_origin,
    invoke_input_context_answer,
    invoke_query_source_documents_result,
    invoke_unique_docs_union_from_retriever,
)
from vectorstores.get_vectorstore import collect_documents


def overview():
    ############## INITIAL SETUP ##############
    # Go to the setup.py and set you tokens, key and setup params: vectorstore type, models, local paths
    print_config()

    ############## EMBEDDING MODEL ##############
    # Load model for embedding documents
    logger.info(f"LLM_EMB : {Config.HF_EMB_MODEL}")
    # llm_emb = create_llm_emb_default()
    # OR This return:
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

    ############## GENERATOR MODEL ##############
    # Load model for generating answer
    logger.info(f"LLM : {Config.HF_LLM_NAME}")
    # llm_gen = create_llm_gen_default()
    # OR This returns:
    llm_gen = HuggingFacePipeline.from_model_id(
        # https://api.python.langchain.com/en/latest/llms/langchain_community.llms.huggingface_pipeline
        # .HuggingFacePipeline.html
        model_id=os.path.join(Config.MODEL_SOURCE, Config.HF_LLM_NAME),
        task="text-generation",
        device=-1,  # -1 stands for CPU
        pipeline_kwargs={
            # full list of parameters for this section with explanation:
            # https://huggingface.co/docs/transformers/en/main_classes/text_generation
            # Note: some of them (depends on the specific model) should go to the model_kwargs attribute
            "max_new_tokens": 512,  # How long could be generated answer
            "return_full_text": False,
            # "return_full_text": True if you want to return within generation answer also all prompts,
            # contexts and other serving instrumentals
        },
        model_kwargs={
            # full list of parameters for this section with explanation:
            # https://huggingface.co/docs/transformers/en/main_classes/text_generation
            # Note: some of them (depends on the specific model) should go to the pipeline_kwargs attribute
            "do_sample": True,
            "top_k": 10,
            "temperature": 0.0,
            "repetition_penalty": 1.03,  # 1.0 means no penalty
            "max_length": 20,
        },
    )

    ############## LOAD DOCUMENTS BASE ##############
    # Create new vectorstore (FAISS)
    logger.info("#### RELOAD_VECTORSTORE ####")
    # Load Documents
    docs = collect_documents()  # this contains list of texts from my documents base

    ############## TEXT SPLITTER FOR DOCUMENTS ##############
    # split documents to chunks, retriever will search through embedded chunks, not whole documents
    logger.info("Split")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # num of characters in single chunk
        chunk_overlap=200,  # num of characters to appear in neighborous chunks
    )
    splits = text_splitter.split_documents(docs)
    del docs  # for gc
    logger.info(f"Num of splits : {len(splits)}")

    ############## VECTORSTORE FOR EMBEDDINGS ##############
    # create vector store FAISS
    # https://python.langchain.com/v0.1/docs/integrations/vectorstores/faiss/
    # for Num of splits : 700 will take Time : ~60min
    logger.info("vectorstore FAISS")
    # do whole work in one approach (Note: FAISS has no verbose parameter)
    # vectorstore = FAISS.from_documents(documents=splits,
    #                                    embedding=llm_emb)
    # add progress bar to FAISS creating procedure, to see some verbose:
    vectorstore = FAISS.from_documents(
        documents=[splits[0]], embedding=llm_emb  # here we provide our embedding model
    )
    splits = splits[1:]
    for d in tqdm(splits, desc="vectorstore FAISS documents"):
        vectorstore.add_documents([d])
    del splits  # for gc

    # save vectorstore FAISS to the disk (Note: Chroma has another signature)
    vectorstore.save_local(Config.VECTORSTORE_FILE)

    # load vectorstore FAISS from the disk (Note: Chroma has another signature)
    logger.info("vectorstore FAISS from dump")
    vectorstore = FAISS.load_local(
        folder_path=Config.VECTORSTORE_FILE,
        embeddings=llm_emb,  # here we provide our embedding model
        allow_dangerous_deserialization=True,  # True for data (docs) with loading from a pickle file.
    )

    ############## RETRIEVER MODEL FROM EMBEDDING MODEL ##############
    logger.info("RETRIEVER")
    retriever = vectorstore.as_retriever(
        # full list of parameters for this section with explanation:
        # https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html
        # #langchain_chroma.vectorstores.Chroma.as_retriever
        search_type="similarity",
        search_kwargs={
            "k": 4
        },  # return top-4 relevant (according to search_type) documents for single query
    )
    del vectorstore  # for gc

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

    ############## V3 with Runnable Sequences FULL RAG = RETRIEVER + GENERATOR ##############
    # Generate multiple alternatives to the question formulation
    # Prompt for multiple alternatives to the question formulation
    prompt_multi_query = PromptTemplate(
        template=prompt_templates_retrieve.prompt_multi_query,
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
            # my prompt has a variable for number of alternative questions to generate.
            # Actual value will be taken from this.invoke({}) calling from key "question_numbers"
        }
        | prompt_multi_query
        | llm_gen
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
        # Here we need to pass as input to the invoke_generate_queries_with_origin 2 variables, as keys inside dict:
        # "alternatives" - output from the last step of previous chain (generate_queries_chain),
        # as well as additional var ("question"). Create a dict with them as input to the RunnableLambda
        # We also pass the name for the output of previous chain (generate_queries_chain) as key of the dict
        {"question": itemgetter("question"), "alternatives": generate_queries_chain}
        # To enable function invoke_generate_queries_with_origin to use this dict as input, it should be
        # RunnableLambda
        | RunnableLambda(invoke_generate_queries_with_origin)
    )
    # to check multiple generated questions:
    # result = invoke_generate_queries_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    # print(result)

    # Retrieval Chain for multiple alternatives to the question formulation
    # Retriever will embed input question (as well as my previously generated alternatives) with the same llm_emb
    # model as was using for vectorstore and will provide top_k documents similar to the question by search_type (
    # values of top_k and search_type were provided in calling vectorstore.as_retriever() above.
    retrieval_chain = (
        # We can attach previous chains as input to the next chain:
        invoke_generate_queries_chain
        # Next step is retriever. Here we need to split str with alternative multiple queries into list to
        # allow retriever to deal with them separatedlly and calling .map() function.
        | (lambda x: x.split("\n"))
        | retriever.map()
        | invoke_unique_docs_union_from_retriever
    )
    # to check list of retrieved documents
    # result = retrieval_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    # print(result)

    # Prompt for generation answer with retriever and generation prompt
    prompt_generation = PromptTemplate(
        template=prompt_templates_generate.prompt_template_question_context,
        input_variables=["question", "context"],
    )
    # RAG Chain
    # Generator (could be another model as for retriever) takes list of retrieved (relevant) documents and generate
    # answer for the qustion according to them.
    rag_chain = (
        {
            "context": retrieval_chain,
            "question": itemgetter("question"),
        }
        # Here again: since prompt_generation takes as input 2 variables with names: context and question,
        # we assign these name to the variables as dict keys.
        # "context" will take value from the output of retrieval_chain
        # "question" will take value from calling this.invoke() with provided "question" key
        | prompt_generation
        | llm_gen
        | StrOutputParser()
    )

    result = rag_chain.invoke({"question": Config.MYQ, "question_numbers": 2})
    print(result)


if __name__ == "__main__":
    overview()
