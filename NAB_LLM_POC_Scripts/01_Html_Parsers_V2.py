# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC ## Evaluate latest files (for Personal only)

# COMMAND ----------

dbutils.fs.ls('dbfs:/nab_demo/input_files_v2/extracted/nabrwd/en/personal')

# COMMAND ----------

#Copy the latest set of files received from NAB to local folder for processing 
#We are only focusing on (Personal) and later we can expand it with other files

%cp '/dbfs/nab_demo/input_files_v2/extracted/nabrwd/en/personal' personal_v2 --recursive

# COMMAND ----------

# MAGIC %md
# MAGIC ## HTML Parsing for NAB Files

# COMMAND ----------

# MAGIC %ls

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Evaluate input filepath and hierarchy
# MAGIC
# MAGIC This is just a temporary script to evaluate how each files are presented

# COMMAND ----------

import os 

file_list = []

for root, _, files in os.walk('personal_v2'):
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
# MAGIC # Section 2 
# MAGIC ## LATEST Modified Script to return a pandas dataframe 

# COMMAND ----------

#Create exclusion list to manually remove the common elements from each files
#This includes both header and footer columns

header_excl_part_1 = "NAB Search Search nab.com.au Search nab.com.au Latest offers close notification" 
header_excl_part_2 = "Update your browser"
header_excl_part_3 = "NAB Mobile Banking app Update your browser." 
header_excl_part_4 = "This website doesn't support your browser and may impact your experience."

header_excl_part_5 = "How was your visit to the NAB website? We’d love to hear from you. New NAB Now Pay Later Link your NAB Classic Banking account to NAB Now Pay Later to split your purchases into four simple repayments and pay no interest or fees. Offer Up to $300 cash back Looking for a credit card that not only offers a low interest rate but provides up to $300 cash back? Offer applies to a new NAB Low Rate Card . $100 cash back per month for the first three months from account opening when you spend $500 per month on purchases. Awarded monthly based on statement period and credited on closing date of statement. Purchases must be processed and charged in the relevant month. Excludes gambling/gaming related transactions. Offer may vary or end at any time. Not available when closing or transferring from another NAB credit card or with other NAB card offer. View calculators $0 international transfer fee Transfer your money securely overseas using NAB Internet Banking or the NAB app. $0 transfer fee when sending in a foreign currency. Discounts and benefits With a NAB business transaction account, you can enjoy discounts, benefits and offers from our partners. Get a fast, simple unsecured loan with NAB QuickBiz. No physical assets required for security and fast access to funds. It's easy to apply online and you'll receive an instant decision. Offer 100,000 NAB Rewards bonus points Earn 100,000 NAB Rewards Bonus Points when you spend $4,000 on everyday business purchases within 60 days of your account opening. Terms and conditions apply. Take payments with NAB Easy Tap Download the NAB Easy Tap app to your Android device for a low-cost, simple and easy way to take contactless card payments. Related tools and help NAB Connect NAB Connect is a powerful online banking solution that offers your business the flexibility of multiple users, advanced reporting and much more. Business product selection, made easy. Explore multiple products all in one place with Small Biz Explorer. The Morning Call Podcast Start your day with the NAB Morning Call Podcast, for the latest overnight key economic and market information straight from our team of experts. The Morning Call Podcast Start your day with the NAB Morning Call Podcast, for the latest overnight key economic and market information straight from our team of experts. More about sustainability Fraud and scams support Troubleshooting guides Fraud and scams support Search nab.com.au Login Internet Banking"

footer_exclusion_blurb = "Any advice on our website has been prepared without considering your objectives, financial situation or needs. Before acting on any advice, consider whether it is appropriate for your circumstances and view the Product Disclosure Statement or Terms and Conditions available online or by contacting us. Credit applications are subject to credit assessment criteria. Interest rates, fees and charges are subject to change. Target Market Determinations for our products are available at . Products issued by NAB unless stated otherwise. © National Australia Bank Limited ABN 12 004 044 937 AFSL and Australian Credit Licence 230686"


# COMMAND ----------

#Code to parse the html files  and create a clean set of files to be loaded as a delta table

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
    cleaned_text = re.sub("\n\n\n+",' ', cleaned_text )
    
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

def remove_specific_text(passed_string , blurb_to_remove):
  #Find and remove the blurb of text from all files
  cleaned_text = passed_string.replace(blurb_to_remove ,'')
  
  return cleaned_text

def parse_html_file_sample(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read() 

     # Remove non-UTF-8 characters
    content = content.encode('utf-8', 'ignore').decode('utf-8')

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

  #Step 6: Remove any additional special characters from the text string 
    without_special = re.sub("\s+",' ', final_cleaned_html )

    # Remove the common text blurb across all files
    for i in [header_excl_part_1,header_excl_part_2,header_excl_part_3,header_excl_part_4,header_excl_part_5,footer_exclusion_blurb]:
      without_special = remove_specific_text(without_special,i)
      
  # # Save the updated HTML content to a new file in the local repository
  #   with open(output_path, 'w', encoding='utf-8') as output_file:
  #       output_file.write(final_cleaned_html)

    # return cleaned_html
    return without_special

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Execution of the code

# COMMAND ----------

import pandas as pd
import os

final_result = []
final_filename = []

# Execution 
for root, _, files in os.walk('personal_v2'):
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
Parsed_output_v2 = pd.DataFrame({'filename':final_filename,'parsed_results':final_result}) 

# COMMAND ----------

#Store the pandas dataframe into DBFS
Parsed_output_v2.to_parquet('/dbfs/nab_demo/output_text_files/Full_Parsed_Dataset_Personal.parquet')

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Sample Validation of Parsed Files

# COMMAND ----------

Parsed_output_v2

# COMMAND ----------

Parsed_output_v2[Parsed_output_v2["filename"] == 'nab-straightup-card.html']

# COMMAND ----------

# String Validation
Parsed_output_v2[Parsed_output_v2["filename"] == "nab-straightup-card.html"][
    "parsed_results"
].tolist()[0]

# COMMAND ----------

Parsed_output_v2[Parsed_output_v2["filename"] == 'first-home-buyers.html']['parsed_results'].tolist()[0]

# COMMAND ----------

Parsed_output_v2[Parsed_output_v2["filename"] == 'nab-low-rate-card.html']['parsed_results'].tolist()[0]

# COMMAND ----------

Parsed_output_v2[Parsed_output_v2['filename'].str.contains('card-based')]['parsed_results'].tolist()[0]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Convert dataframe to spark and store it as a Delta table

# COMMAND ----------

from pyspark.sql.functions import *
# import pyspark.sql.functions as f

#Convert the pandas dataframe into a spark datadframe
parsed_spark_df_v2 = spark.createDataFrame(Parsed_output_v2)


#Create a new column to determine the length of the strings
parsed_spark_df_v2 = parsed_spark_df_v2 \
                     .withColumn("word_count",size(split(col('parsed_results'), ' '))) \
                      .withColumn("length_of_string", length(col('parsed_results'))) 

#Save the pyspark dataframe as a delta table in unity catalog 
parsed_spark_df_v2.write.mode("overwrite").saveAsTable("nab_llm_demo.nab_website_data.parsed_dataset_v2")

print(f'Length of dataframe : {parsed_spark_df_v2.count()}')

parsed_spark_df_v2.orderBy('word_count',ascending=False).show(20)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Store Text Files into DBFS

# COMMAND ----------

#Function to store all text excerpts as text file into DBFS
def store_string_as_text_file(file_path, text):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
        print(f'stored the file in : {file_path}')


for i in range(len(Parsed_output_v2)):
  #Name of the text file 
  filename = 'parsed_' + Parsed_output_v2.iloc[i]['filename'][0:-5] + '.txt'
  #File path name of the text file 
  filepath = os.path.join('/dbfs/nab_demo/output_text_files',filename)
  print(f'file path is : {filepath}')
  #Content of the file 
  file_content = Parsed_output_v2.iloc[i]['parsed_results']
  print (f'character length of one entry : {len(file_content)} and word length is : {len(file_content.split())}')
  store_string_as_text_file(filepath, file_content)
