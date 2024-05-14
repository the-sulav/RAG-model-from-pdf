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


#loading data from PDF

loader = PyPDFLoader("kathmandu_post.pdf")
pages = loader.load_and_split()

#splitting the pdf files into chunks because the embedding model is limited my token size

text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 800,
      chunk_overlap = 80,
      length_function = len,
      is_separator_regex =False,)

chunks = text_splitter.split_documents(pages)

#storing the embedings into a vector database for easy retrival
#using ChromaDB
vectorstore = Chroma.from_documents(
    documents = chunks,
    collection_name ="rag_chroma",
    embedding = embeddings.ollama.OllamaEmbeddings(model = "nomic-embed-text")
)

#instance of vector database
retriever = vectorstore.as_retriever()

#defining model also can use llama3 

model_local = ChatOllama(model = "mistral")

#template for LLM to answer questions

after_rag_template = """Answer the question based on the following context:
{context}
Question : {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

# Defining pipeline
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)

#The question you want to ask
print(after_rag_chain.invoke("Explain about Women Empowerment in Nepalese Banks"))

