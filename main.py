import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import ollama
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings



loader = PyPDFLoader("kathmandupost.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 800,
      chunk_overlap = 80,
      length_function = len,
      is_separator_regex =False,)

chunks = text_splitter.split_documents(pages)

vectorstore = Chroma.from_documents(
    documents = chunks,
    collection_name ="rag_chroma",
    embedding = embeddings.ollama.OllamaEmbeddings(model = "nomic-embed-text")
)

retriever = vectorstore.as_retriever()

model_local = ChatOllama(model = "mistral")

after_rag_template = """Answer the question based on the following context:
{context}
Question : {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("compare between japanese yen and us dollars"))

