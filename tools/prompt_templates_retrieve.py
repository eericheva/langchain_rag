# Multi Query: Different Perspectives
prompt_multi_query = """You are an AI language model assistant. Your task is to generate {question_numbers}
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines.

Original question: {question}
"""
