prompt_template_input_context = """
Use the following pieces of context to answer the question at the end.
Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer.
Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {input}

*** Helpful Answer***:
"""

prompt_template_question_context = """
Use the following pieces of context to answer the question at the end.
Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer.
Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {question}

*** Helpful Answer***:
"""
