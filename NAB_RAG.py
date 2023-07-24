# Databricks notebook source
# MAGIC %ls /dbfs/nab_demo/nab_personal_banking_file_set/accounts/parsed-v2-joint-accounts.html*  

# COMMAND ----------

!pip install lxml
!pip install html5lib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

!pip install lxml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
pd.read_html('/dbfs/nab_demo/nab_personal_banking_file_set/accounts/joint-accounts.html')

# COMMAND ----------

#Save as a delta table. 
#1 row each file 

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


