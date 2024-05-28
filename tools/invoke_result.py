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
