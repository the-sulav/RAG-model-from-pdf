# LangChain Community - Document Analysis

## Overview
This repository showcases a Python script that illustrates document analysis using LangChain, an extensive toolkit designed for various natural language processing tasks. The script exemplifies the process of loading a PDF document, segmenting it into smaller portions, embedding these segments into a vector database, and utilizing a language model to generate responses to questions based on the provided context.

By leveraging the capabilities of LangChain, users can gain insights into efficient and effective methods for processing textual data.

It's important to note that this script represents an initial attempt at document analysis, and there may be limitations. The language model used in this example may not provide satisfactory answers to all questions, and the overall performance of the script may be slow, especially for large documents. Further optimizations and improvements can be explored to enhance the functionality and efficiency of the document analysis process.




## Requirements
- Download Ollama and also download "mistral" and "nomic-embed-text" model in ollama for the code to work
- Python 3.10 ( ChromaDB didnot work in version >3.10 when i tried)
- LangChain library
  - `langchain_community`
  - `langchain_core`
  - `langchain_text_splitters`
  - `langchain.schema`
  
## Installation
  1. Langchain
  2. Langchain_community
  3. Langchain_core
  4. langchain_text_splitters

## Usage
1. Ensure you have a PDF document you want to analyze.
2. Modify the script to load your PDF document:
```python
loader = PyPDFLoader("your_document.pdf")

## Run The Script
python document_analysis.py
