# RAG Bot: Retrieve and Generate Bot for Contextual Information Retrieval

## Overview

This project showcases a Retrieve and Generate (RAG) bot that combines advanced language models with document retrieval techniques to answer questions based on the context extracted from documents. The bot is capable of providing accurate and relevant responses by leveraging embeddings, vector stores, and large language models.

## Features

- **Document Ingestion**: Load and process text and PDF documents.
- **Vector Embeddings**: Generate embeddings using `OpenAIEmbeddings` for document chunks.
- **Vector Stores**: Store embeddings in efficient vector stores using `Chroma` and `FAISS`.
- **Similarity Search**: Perform similarity searches to retrieve relevant document chunks.
- **Contextual Answer Generation**: Generate responses based on the retrieved context using advanced language models like `Ollama`.

https://github.com/abdoolamunir/Retrieval-Augmented-Generation/issues/1#issue-2438045543

## Installation

To run this project, you'll need to have Python installed along with the necessary dependencies. You can install the required packages using `pip`:

```bash
pip install langchain langchain_community langchain_openai dotenv
```

## Usage

### Data Ingestion

Load and process text documents using the `TextLoader`:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("speech.txt")
text_documents = loader.load()
```

### Vector Embeddings and Vector Store

Generate embeddings for the document chunks and store them in a vector store:

```python
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

db = Chroma.from_documents(documents[:20], OpenAIEmbeddings())
```

### Similarity Search

Perform similarity search queries to find relevant document chunks:

```python
query = "design by"
result = db.similarity_search(query)
print(result[0].page_content)
```

### FAISS Vector Database

Use `FAISS` for creating an alternative vector store and perform similarity searches:

```python
from langchain_community.vectorstores import FAISS

db1 = FAISS.from_documents(documents[:20], OpenAIEmbeddings())
query = "It sounds a dreadful thing to say"
result = db1.similarity_search(query)
print(result[0].page_content)
```

### Full Example

Here is the complete example code integrating all the components:

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("Like_War_The_Weaponization_of_Social_Media.pdf")
docs = loader.load()

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(documents[:50], OpenAIEmbeddings())
query = "It sounds a dreadful thing to say"
result = db.similarity_search(query)
print(result[0].page_content)

from langchain_community.llms import Ollama

llm = Ollama(model="gemma")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the context given, do not assume it is true. If you don't know the answer, just say that you don't know.
<context>
{context}
</context>
Question: {input}
""")

from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = db.as_retriever()

from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "It sounds a dreadful thing to say"})
print(response['answer'])
```

## Conclusion

This project demonstrates the power of combining retrieval and generation techniques to create intelligent systems capable of providing precise and context-aware answers. By leveraging embeddings, vector stores, and advanced language models, we can automate information retrieval and question-answering tasks effectively.

## License

This project is licensed under the MIT License.

---
