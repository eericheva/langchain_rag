# RAG system locally with LangChain RunnableSequence. HOWTOs

Repo contains scripts with overly detailed explanations as well as advanced scripts with not an excessive number of
details and comments. These resources aim to provide someone with concise guidance and practical examples for creating a
RAG system from scratch.

How to read and create RAG:

- with RunnableSequences (langchain) _(if you want clean and structured approach and easy-to-follow code sequences)_
- with HuggingFace models _(if you want to try some the very resent releases and cutting-edge technology)_
- localy _(if you love the smell of code in the morning)_

Pls, investigate [`start_here.py`](start_here.py). The file is exceptionally detailed from the start.

How to choose retrieval model (llm embedder)? --> [mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard), tab:
Retrieval or Retrieval w/Instruction

How to choose reranking model (reorder list of releveant documents)? --> [mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard), tab:
Reranking

How to choose generator model (llm for generate finel answer)? --> [open-llm-leaderboard/open_llm_leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

Pls, refer to the other options and files listed below, to get less commented, but more advanced scripts, examples and
techniques.

### Advanced option to rule RAG:

<table>
    <thead>
        <tr>
            <th>HOWTO</th>
            <th>Option</th>
            <th>Go-to file</th>
            <th>Outer documentation</th>
        </tr>
    </thead>
    <tbody>
        <!-- <tr>
            <td rowspan=4 align="left">R1 Text</td>
            <td rowspan=2 align="left">R2 Text A</td>
            <td align="left">R3 Text A</td>
        </tr>
        <tr>
            <td align="left">R3 Text B</td>
        </tr>
        <tr>
            <td rowspan=2 align="left">R2 Text B</td>
            <td align="left">R3 Text C</td>
        </tr>
        <tr>
            <td align="left">R3 Text D</td>
        </tr> -->
        <tr>
            <td rowspan=2 align="left">How to run HuggingFace models</td>
            <td align="left">localy</td>
            <td align="left">

[local_run_example.py](local_run_example.py) </td>
<td align="left"></td>
</tr>
<tr>
<td align="left">with HuggingFaceHub</td>
<td align="left"></td>
<td align="left">

[Hugging Face Hub documentation](https://huggingface.co/docs/hub/en/index) </td>
</tr>
        <tr>
        <td align="left">How to evaluate and monitoring application</td>
        <td align="left">with LangSmith</td>
        <td align="left"></td>
        <td align="left">

[Get started with LangSmith](https://docs.smith.langchain.com/) </td>
</tr>
<tr>
<td align="left">How to store embeddings in vectorstore (FAISS or Chroma)</td>
<td align="left">

default with:

- text splitter
- progress bar on creating vectorstore
- dump and load from disk </td>
  <td align="left">

[get_vectorstore.py](vectorstores/get_vectorstore.py)

[create_vectorstore.py](vectorstores/create_vectorstore.py)</td>
<td align="left">

[FAISS](https://python.langchain.com/v0.1/docs/integrations/vectorstores/faiss/)

[Chroma](https://python.langchain.com/v0.1/docs/integrations/vectorstores/chroma/) </td>
</tr>
<tr>
<td rowspan=3 align="left">How to embed documents</td>
<td align="left">default</td>
<td align="left">

[create_llm_emb_default.py](embedders/create_llm_emb_default.py) </td>
<td align="left">

[Text embedding models](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/) </td>
</tr>
<tr>
<td align="left">with Caching</td>
<td align="left">

[create_llm_emb_cached.py](embedders/create_llm_emb_cached.py)
</td>
<td align="left">

[Caching Embeddings](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/caching_embeddings/) </td>
</tr>
<tr>
<td align="left">with Compressing</td>
<td align="left"></td>
<td align="left"></td>
</tr>
<tr>
<td rowspan=4 align="left">How to retrieve documents</td>
<td align="left">default</td>
<td align="left">

[create_vectorstore.py](vectorstores/create_vectorstore.py) </td>
<td align="left">
</tr>
<tr>
<td align="left">with Multiple Queries Generation</td>
<td align="left"></td>
<td align="left"></td>
</tr>
<tr>
<td align="left">

with `chain_type` :

- `stuff`,
- `map_reduce`,
- `refine`,
- `map_rerank`</td>
  <td align="left"></td>
  <td align="left"></td>
  </tr>
  <tr>
  <td align="left">with Prompting</td>
  <td align="left"></td>
  <td align="left"></td>
  </tr>
  <tr>
        <td align="left">How to generate answer</td>
        <td align="left">default</td>
        <td align="left">

  [create_llm_gen_default.py](generators/create_llm_gen_default.py) </td>
        <td align="left"></td>
</tr>

    </tbody>

</table>
