from langchain.load import dumps, loads


def invoke_input_context_answer(chain_invoke_result):
    answer = ""
    answer += "QUESTION: \n"
    answer += chain_invoke_result.get("input")
    answer += "\n\n"
    answer += "BASED DOCUMENTS: \n"
    for d in chain_invoke_result.get("context"):
        answer += (
            d.metadata.get("source").split("/")[-1]
            + ", page : "
            + str(d.metadata.get("page"))
            + "\n"
        )
    answer += "\n\n"
    answer += "ANSWER: \n"
    answer += (
        chain_invoke_result.get("answer").split("*** Helpful Answer***:")[-1].strip()
    )
    return answer


def invoke_query_source_documents_result(chain_invoke_result):
    answer = ""
    answer += "QUESTION: \n"
    answer += chain_invoke_result.get("query")
    answer += "\n\n"
    answer += "BASED DOCUMENTS: \n"
    for d in chain_invoke_result.get("source_documents"):
        answer += (
            d.metadata.get("source").split("/")[-1]
            + ", page : "
            + str(d.metadata.get("page"))
            + "\n"
        )
    answer += "\n\n"
    answer += "ANSWER: \n"
    answer += (
        chain_invoke_result.get("result").split("Generate according to:")[-1].strip()
    )
    return answer


def invoke_generate_queries_with_origin(queries_result: dict) -> str:
    question = queries_result.get("question")
    alternatives = queries_result.get("alternatives").replace("\n\n", "\n")
    new_queries = f"Original question: {question}?" + alternatives
    return new_queries


def invoke_unique_docs_union_from_retriever(documents: list[list]) -> list:
    """Unique union of retrieved docs"""
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]
