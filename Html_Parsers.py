# Databricks notebook source
# MAGIC %md
# MAGIC ## HTML Parsing for NAB Files

# COMMAND ----------

# MAGIC %ls

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Evaluate the input filepath and the hierarchical structure. This is just a temporary script to evaluate how each files are presented

# COMMAND ----------

import os 

file_list = []

for root, _, files in os.walk('nab_personal_banking_file_set'):
  for filename in files:
    #Remove any files that are hidden
    if filename.startswith("."):
      continue
    else:
      html_file_path = os.path.join(root,filename)
      output_filename = f'parsed-v2-{filename}'
      output_path = os.path.join(root,output_filename)
      print(output_path)
      file_list.append(filename)

print(f'Total files present : {len(file_list)}')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### HTML Parsing via Beautiful Soup
# MAGIC
# MAGIC There are few duplicates across all files in terms of repeat hyperlinks. The logic of the code is something below:
# MAGIC 1. Traverse all directories
# MAGIC 2. Load each html file inside the directory. 
# MAGIC 3. Remove any hyperlinks (Note: Hyperlink section is repititive in each file related to other page buttons and references)
# MAGIC 4. Remove any special characters
# MAGIC 5. Preserve anynewline characters
# MAGIC 6. Remove images and embedded objects (if any) (Majority looked as logos)
# MAGIC 7. Store the clean file as an html object with 'utf-8' format.
# MAGIC 8. Store each parse file in "html" format in the respective repository next to the input file
# MAGIC
# MAGIC The python library "BeautifulSoup" and "re" is used for the parsing.
# MAGIC

# COMMAND ----------

from bs4 import BeautifulSoup
import re

def remove_hyperlinks(soup):
    # Find all anchor tags (hyperlinks) in the HTML content
    for a_tag in soup.find_all('a'):
        # Remove the entire anchor tag (hyperlink element) from the HTML content
        a_tag.extract()

def remove_special_characters_and_spaces(text):
    # Remove special characters using regular expressions
    cleaned_text = re.sub(r'^[\s•·●▪◦▸\-–—‣⁃▹*]+(?=\s*\d*[.,]?\s*)', '', text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text

def preserve_newlines(soup):
    # Preserve newlines within specific tags that may use newlines for formatting
    for tag in soup.find_all(['pre', 'textarea']):
        tag.insert_before('\n')
        tag.insert_after('\n')
        tag.unwrap()
  
def remove_images_and_objects(soup):
    # Find all mage tags (img) and document tags (embed, object, iframe) and remove them from the HTML content
    for doc_tag in soup.find_all(['img','embed', 'object', 'iframe']):
        doc_tag.extract()

def parse_html_file_sample(file_path,output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(content, 'html.parser')

    # Step 1: Remove hyperlinks from the HTML content
    remove_hyperlinks(soup)

    # Step 2: Get the updated HTML content without hyperlinks
    html_without_hyperlinks = soup.prettify()

   # Step 3:  Remove special characters from the HTML content
    clean_html_content = remove_special_characters_and_spaces(html_without_hyperlinks)

  # Step 4: Remove images and embedded documents from the HTML content
    soup_without_images_or_documents = BeautifulSoup(clean_html_content, 'html.parser')
    remove_images_and_objects(soup_without_images_or_documents)

 # Preserve newlines within specific tags (pre, textarea) that may use newlines for formatting
    preserve_newlines(soup_without_images_or_documents)

  # Step 5: Remove newline characters from the HTML content
    # final_cleaned_html = remove_newline_characters(soup_without_images_or_documents.get_text())
    final_cleaned_html = soup_without_images_or_documents.get_text()

  # Save the updated HTML content to a new file in the local repository
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(final_cleaned_html)

# COMMAND ----------

## Execution 
for root, _, files in os.walk('nab_personal_banking_file_set'):
  for filename in files:
    if filename.endswith("html"):
      html_file_path = os.path.join(root,filename)
      output_filename = f'parsed-v3-{filename}'
      output_path = os.path.join(root,output_filename)
      print(f'The output file name is : {output_path}')
      parse_html_file_sample(html_file_path,output_path)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Move to DBFS

# COMMAND ----------

# MAGIC %cp  nab_personal_banking_file_set/  '/dbfs/nab_demo/' --recursive

# COMMAND ----------

dbutils.fs.ls('dbfs:/nab_demo/nab_personal_banking_file_set/buy-now-pay-later/terms-conditions/')

# COMMAND ----------

# MAGIC %ls /dbfs/nab_demo

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Modified Script to return a pandas dataframe 

# COMMAND ----------

from bs4 import BeautifulSoup
import re

def remove_hyperlinks(soup):
    # Find all anchor tags (hyperlinks) in the HTML content
    for a_tag in soup.find_all('a'):
        # Remove the entire anchor tag (hyperlink element) from the HTML content
        a_tag.extract()

def remove_special_characters_and_spaces(text):
    # Remove special characters using regular expressions
    cleaned_text = re.sub(r'^[\s•·●▪◦▸\-–—‣⁃▹*]+(?=\s*\d*[.,]?\s*)', '', text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text

def preserve_newlines(soup):
    # Preserve newlines within specific tags that may use newlines for formatting
    for tag in soup.find_all(['pre', 'textarea']):
        tag.insert_before('\n')
        tag.insert_after('\n')
        tag.unwrap()
  
def remove_images_and_objects(soup):
    # Find all mage tags (img) and document tags (embed, object, iframe) and remove them from the HTML content
    for doc_tag in soup.find_all(['img','embed', 'object', 'iframe']):
        doc_tag.extract()

def parse_html_file_sample(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(content, 'html.parser')

    # Step 1: Remove hyperlinks from the HTML content
    remove_hyperlinks(soup)

    # Step 2: Get the updated HTML content without hyperlinks
    html_without_hyperlinks = soup.prettify()

   # Step 3:  Remove special characters from the HTML content
    clean_html_content = remove_special_characters_and_spaces(html_without_hyperlinks)

  # Step 4: Remove images and embedded documents from the HTML content
    soup_without_images_or_documents = BeautifulSoup(clean_html_content, 'html.parser')
    remove_images_and_objects(soup_without_images_or_documents)

 # Preserve newlines within specific tags (pre, textarea) that may use newlines for formatting
    preserve_newlines(soup_without_images_or_documents)

  # Step 5: Remove newline characters from the HTML content
    # final_cleaned_html = remove_newline_characters(soup_without_images_or_documents.get_text())
    final_cleaned_html = soup_without_images_or_documents.get_text()

  # # Save the updated HTML content to a new file in the local repository
  #   with open(output_path, 'w', encoding='utf-8') as output_file:
  #       output_file.write(final_cleaned_html)

    return final_cleaned_html

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Execution of the code

# COMMAND ----------

import pandas as pd

final_result = []
final_filename = []

# Execution 
for root, _, files in os.walk('personal'):
  for filename in files:
    if filename.endswith("html"):
      #Input html file 
      html_file_path = os.path.join(root,filename)
      print(f"input file location is : {html_file_path}")

      # output_path = os.path.join(root,output_filename)
      # print(f'The output file name is : {output_path}')

      parsed_result = parse_html_file_sample(html_file_path)
      final_result.append(parsed_result)
      final_filename.append(filename)

#Create a single pandas dataframe with filename and 
Parsed_output = pd.DataFrame({'filename':final_filename,'parsed_results':final_result}) 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Convert dataframe to spark and store it as a Delta table

# COMMAND ----------

#Convert the pandas dataframe into a spark datadframe
parsed_spark_df = spark.createDataFrame(Parsed_output)

#Save the pyspark dataframe as a delta table in unity catalog 
parsed_spark_df.write.mode("overwrite").saveAsTable("nab_llm_demo.nab_website_data.parsed_dataset")

#Manually Evaluate the dataset
parsed_spark_df.show()
