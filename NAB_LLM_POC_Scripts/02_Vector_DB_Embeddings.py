# Databricks notebook source
!pip install chromadb==0.3.21
# !pip install xformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Langchain Text Loader & Chunking

# COMMAND ----------

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.llms import HuggingFaceHub

# Manual Model building
from transformers import pipeline

#Importing necessary libraries
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# COMMAND ----------

# File_Path = '/dbfs/nab_demo/output_text_files/parsed_nab-low-rate-card.txt'
# File_Path = '/dbfs/nab_demo/output_text_files/parsed_first-debit-card.txt'
# File_Path = '/dbfs/nab_demo/output_text_files/parsed_capital-gains-tax.txt'
File_Path = '/dbfs/nab_demo/output_text_files/parsed_nab-low-fee-card.txt'

# COMMAND ----------

#Create chunking of the text data for a single file
# load the document and split it into chunks
loader = TextLoader(File_Path)
file_content = loader.load()

#Define Character Splitter
text_splitter = CharacterTextSplitter(        
    separator = ".",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
#Load the text splitter and split documents based on chunk size 
docs = text_splitter.split_documents(file_content)

# COMMAND ----------

[str(str(x.metadata).split("/")[-1]).replace("'}","") for x in docs]

# COMMAND ----------

[str(str(x.metadata).split("/")[-1]).replace("'}","") + "_" + str(docs.index(x)) for x in docs],

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Vector DB

# COMMAND ----------

#section to rebuild the vector store 
dbutils.fs.rm('dbfs:/nab_demo/vectorstore_persistence/db', True)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Create Vector Index 

# COMMAND ----------

#Create text embeddings from Hugging Face model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                   model_kwargs={'device': 'cpu'})
                                   
#Defining the persistant storage of the chroma DB
vector_store_directory = '/dbfs/nab_demo/vectorstore_persistence/db'

#Load the embedding data into Chroma 
docsearch = Chroma.from_documents(docs, 
                                  embeddings ,
                                  collection_name="nab_testing",
                                  persist_directory=vector_store_directory
                                  )
                                  
print('The collection : {} index includes: {} documents'.format(docsearch._collection,docsearch._collection.count()))

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Vector Index Query for Similarity Search

# COMMAND ----------

#Quick validation of vector indexing
query = "What is a NAB low rate credit card and what is it's interest rate?"

query_result = docsearch.similarity_search(query)
[i.page_content for i in query_result[0:2]]

# COMMAND ----------

#Quick validation of vector indexing
query = "What are the offers for NAB Low Rate credit card ?"

query_result = docsearch.similarity_search(query)
[i.page_content for i in query_result[0:2]]

# COMMAND ----------

#Quick validation of vector indexing
query = "How can i protect my password or PIN?"
  
query_result = docsearch.similarity_search(query)
[i.page_content for i in query_result[0:2]]

# COMMAND ----------

#Quick validation of vector indexing
query = "what is a NAB low fee card?"
  
query_result = docsearch.similarity_search(query)
[i.page_content for i in query_result[0:2]]

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Archive

# COMMAND ----------

# #Take a sample file 
# import os
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import TextLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma

# import pandas as pd

# NAB_Files = pd.read_parquet('/dbfs/nab_demo/output_text_files/Full_Parsed_Dataset_Personal.parquet')
# local_file_path = '/Workspace/Repos/ananya.roy@databricks.com/LLM_Playground/Local Dev/state_of_the_union.txt'


# loader = TextLoader(local_file_path)
# documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# # os.environ["OPENAI_API_KEY"] = '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'

# embeddings = OpenAIEmbeddings()
# docsearch = Chroma.from_documents(texts, embeddings)

# qa = RetrievalQA.from_chain_type(llm=OpenAI(), 
#                                  chain_type="stuff", 
#                                  retriever=docsearch.as_retriever()
#                                  )
