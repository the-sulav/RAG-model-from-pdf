# LangChain Community - Document Analysis

## Overview
This script demonstrates document analysis using LangChain, a toolkit for natural language processing tasks. Specifically, it loads a PDF document, splits it into smaller chunks, embeds those chunks into a vector database, and then uses a language model to answer questions based on the provided context.

## Requirements
- Python 3.x
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
