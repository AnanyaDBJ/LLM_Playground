# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC ## Initial version of Web Crawler for Nab Portal
# MAGIC
# MAGIC The below function crawls through Nab portal ( Starting from a random page) and traverses through all hyperlinks and stores each page as an html file into local repo. 
# MAGIC This code stores PDF/Doc/Docx files also in local repo. 
# MAGIC The last part of the code is copying the data from local to /dbfs/nab_demo

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess
import os
import re

class NabSpider(scrapy.Spider):
    name = "nab_spider"
    start_urls = ["https://www.nab.com.au/business/international-and-foreign-exchange/international-money-transfers"]
    allowed_domains = ["nab.com.au"]
    
    def __init__():
      self.path = "/dbfs/nab_demo/"

    def parse(self,response):
        #Save the HTML content to a file 
        content_name = response.url.strip("/")
        file_name = f"{content_name.replace('/', '_')}.html"
        file_path = os.path.join(self.path, file_name)
        print(f"file path is : {file_path}")
        print(f'unique file name is : {file_name}')
        
        with open(file_path,"wb") as file:
            file.write(response.body)
        self.log(f"saved page :{file_path}")
        
        # Extract all links from the current page
        links = response.css("a::attr(href)").getall()

#         for link in links:
#             if re.search(r"\bPDF\b|\bpdf\b", link):
#                 yield response.follow(link, callback=self.save_attachment)
#             elif re.search(r"\bDOC\b|\bdoc\b|\bDOCX\b|\bdocx\b", link):
#                 yield response.follow(link, callback=self.save_attachment)
# #           else:
# #               yield response.follow(link,callback=self.parse)
            
        for link in links:
            if link.endswith((".pdf", ".doc", ".docx")):
              yield response.follow(link, callback=self.save_attachment)
            # elif link.startswith(("http:", "https:")):
            #   yield response.follow(link, callback=self.parse)

    def save_attachment(self,response):
        "Extract the filename from the url"
        print("I am saving")
        file_name = os.path.basename(response.url)
        # file_path = os.path.join("nab_attachments",file_name)
        file_path = os.path.join(self.path,file_name)
        with open(file_path,"wb") as file:
            file.write(response.body)
        self.log(f"saved attachment: {file_path}")
            
        
#Create a sub-folder
os.makedirs("nab_pages",exist_ok=True)
os.makedirs("nab_attachments",exist_ok=True)

#Run the spider
process = CrawlerProcess(settings={"LOG_LEVEL":"ERROR"})

process.crawl(NabSpider)
process.start()

# COMMAND ----------

# MAGIC %cp 'nab_attachments' '/dbfs/nab_demo/' --recursive
# MAGIC %cp 'nab_pages' '/dbfs/nab_demo/' --recursive

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/nab_demo/'))

# COMMAND ----------

ls -p '/dbfs/nab_demo/nab_attachments/' | grep -v / | wc -l

# COMMAND ----------

ls -p '/dbfs/nab_demo/nab_pages/' | grep -v / | wc -l

# COMMAND ----------


