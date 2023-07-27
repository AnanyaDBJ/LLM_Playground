# Databricks notebook source
!pip install chromadb==0.3.21

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Vector Store Creation

# COMMAND ----------

#Creation and update of Vector store
def create_vector_store(delta_table_name,
                        embedding_model,
                        persistant_store_path,
                        rebuild_flag):
  
  #Define the vector store full path to be deleted
  vector_store_path = 'dbfs:/' + persistant_store_path

  if rebuild_flag == True:
    print(f'Deleting the entries in Vector DB : {vector_store_path}')
    dbutils.fs.rm(vector_store_path,True)
 
  # Download model from Hugging face
  hf_embed = HuggingFaceEmbeddings(model_name=embedding_model)

  #Read the spark table into spark dataframe (This table was created on 01_HTML_Parsers_V2 code ,Refer from Section-2 onwards)
  parsed_dataset_table = spark.table(delta_table_name)

  #Vector DB Path
  vector_db_path = '/dbfs/' + persistant_store_path 

  #Splitting the text data
  all_parsed_results = parsed_dataset_table.collect()

  #Create a document object comprising all parsed_results entries from delta table
  NAB_File_Docs = [Document(page_content=i['parsed_results'],
                            metadata={"source": i["filename"]}) for i in all_parsed_results]
  
  # Texts are too long so splitting it based on character limit
  text_splitter = CharacterTextSplitter(separator=".", 
                                        chunk_size=1000, 
                                        chunk_overlap=100
                                        )
  #split the document based on pre-defined chunk
  documents = text_splitter.split_documents(NAB_File_Docs)

  # Init the chroma db with the sentence-transformers/all-mpnet-base-v2 model loaded from hugging face  (hf_embed)
  db = Chroma.from_documents(collection_name="nab_chroma_store", 
                            documents=documents, 
                            embedding=hf_embed, 
                            persist_directory= vector_db_path,
                            ids=[str(str(x.metadata).split("/")[-1]).replace("'}","") + "_" + str(documents.index(x)) for x in documents],
                            metadatas = [x.metadata for x in documents]
                            )

  db.persist()
  print(f'successfully stored all records into vector db with total : {db._collection.count()} records')

  return db

# COMMAND ----------

if __name__ == '__main__':
  delta_table_name = "nab_llm_demo.nab_website_data.parsed_dataset_v2"
  embedding_model = "sentence-transformers/all-mpnet-base-v2"
  persistant_store_path = "nab_demo/Full_Chroma_Store/db"

  #Create vector store (ChromaDB) with full dataset
  docsearch = create_vector_store(delta_table_name,
                      embedding_model,
                      persistant_store_path,
                      rebuild_flag=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Vector load and similarity search

# COMMAND ----------

def vector_similarity_search(chroma_db_obj, 
                             query,
                             number_of_entries):

  #Quick validation of vector indexing
  passed_query = query
  #Perform the similarity search
  query_result = chroma_db_obj.similarity_search(passed_query)
  #Return the similarity search result
  similarity_list = [i.page_content for i in query_result[0:number_of_entries]]
  return similarity_list

# COMMAND ----------

if  __name__ == '__main__':
  number_of_entries = 3
  embedding_model = "sentence-transformers/all-mpnet-base-v2"
  chroma_persistant_directory = '/dbfs/nab_demo/Full_Chroma_Store/'

  # Download the embedding model from Hugging face
  hf_embed = HuggingFaceEmbeddings(model_name=embedding_model)
  
  # load Chroma DB from Persistant Store
  db3 = Chroma(collection_name="nab_chroma_store",
               persist_directory=chroma_persistant_directory, 
               embedding_function=hf_embed)
  
  # user_query = "How to protect my password and pin?"
  # user_query = "How to download a transaction statement?"
  # user_query = "How to manage NAB online statements?"
  # user_query = "what are the NAB Low Rate Card application checklist ?"
  user_query = "What are the NAB some checklist for applying NAB Classic Banking?"

  #Perform similarity search with user query and chroma DB results
  search_results = vector_similarity_search(db3, user_query,number_of_entries)
  

# COMMAND ----------

search_results

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## LLM RAG Implementation

# COMMAND ----------

from huggingface_hub import notebook_login

# Login to Huggingface to get access to the model
notebook_login()

# COMMAND ----------

# Load model to text generation pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of llamav2-7b-chat in https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/commits/main
model = "meta-llama/Llama-2-7b-chat-hf"
revision = "0ede8dd71e923db6258295621d817ca8714516d4"

tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    revision=revision,
    return_full_text=False
)

# Required tokenizer setting for batch inference
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Prompt Generation

# COMMAND ----------

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

CONTEXT_PROMPT = """\With the below context provided below , answer the question asked at the end."""

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>

<s>[INST]<<CONTXT>>
{context_prompt}
<</CONTXT>>
{context}

{instruction}
[/INST]
""".format(
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    context_prompt=CONTEXT_PROMPT,
    context = "{context_val}",
    instruction="{instruction}"
)

# COMMAND ----------

PROMPT_FOR_GENERATION_FORMAT.format(instruction="What are the NAB some checklist for applying NAB Classic Banking?",
                                    context_val = search_results)

# COMMAND ----------

# Define parameters to generate text
def gen_text(prompts, context_results,use_template=False, **kwargs):
    if use_template:
        full_prompts = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt,context_val = context_results)
            for prompt in prompts
        ]
    else:
        full_prompts = prompts

    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1
    
    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## QnA + Prompt

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text(["What are some checklist for applying NAB Classic Banking?"], 
                   search_results,
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

user_query = "Tell me something about a NAB low fee card?"

#Perform similarity search with user query and chroma DB results
search_results = vector_similarity_search(db3, user_query,number_of_entries)

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text([user_query], 
                   search_results,
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

user_query = "Give me some suggestions on how i protect my password or PIN"

#Perform similarity search with user query and chroma DB results
search_results = vector_similarity_search(db3, user_query,number_of_entries)

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text([user_query], 
                   search_results,
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

user_query = "what is the interest rate of NAB low fee card?"

#Perform similarity search with user query and chroma DB results
search_results = vector_similarity_search(db3, user_query,number_of_entries)

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text([user_query], 
                   search_results,
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

user_query = "Tell me something about  NAB low fee card?"

#Perform similarity search with user query and chroma DB results
search_results = vector_similarity_search(db3, user_query,number_of_entries)

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text([user_query], 
                   search_results,
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

user_query = "Tell me something about  NAB low fee card?"

#Perform similarity search with user query and chroma DB results
search_results = vector_similarity_search(db3, user_query,number_of_entries)

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text([user_query], 
                   search_results,
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

user_query = "I have a dispute. What should i do ? "

#Perform similarity search with user query and chroma DB results
search_results = vector_similarity_search(db3, user_query,number_of_entries)

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text([user_query], 
                   search_results,
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])
