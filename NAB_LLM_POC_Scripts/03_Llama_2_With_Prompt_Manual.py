# Databricks notebook source
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

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>

{instruction}
[/INST]
""".format(
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    instruction="{instruction}"
)

# COMMAND ----------

# Define parameters to generate text
def gen_text(prompts, use_template=False, **kwargs):
    if use_template:
        full_prompts = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
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
# MAGIC ### Inference

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text(["What is a large language model?"], 
                   temperature=0.5, 
                   max_new_tokens=100, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Vector DB Similarity Search Results

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Question 1 : 
# MAGIC NAB Low Rate Credit Card

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Without Context

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text(["What is NAB low rate credit card?"], 
                   temperature=0.5, 
                   max_new_tokens=100, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### With context

# COMMAND ----------

context = ['Low rate credit card | Balance transfer or cash back offer -  Notification: NAB Mobile Banking app .   NAB Low Rate Credit Card NAB Low Rate Card A simple low interest rate credit card with your choice of offers: get a promotional balance transfer or select our cash back offer. Apply online in 15 minutes and get a response in 60 seconds. Rates and fees Learn more about our standard interest rates and minimum credit limits. Variable purchase rate 12.49% p.a. Interest free days on purchases Up to 55 Annual card fee $59 p.a. Minimum credit limit $1,000 Variable cash advance rate This is the interest rate charged on amounts you withdraw as cash, gambling transactions (including lottery ticket purchases), or transfer from your credit card to another account. 21.74% p.a. Choose one of our available credit card offers Your choice of offers to suit your credit needs, select either our promotional balance transfer offer or our cash back offer. Offer Promotional 0% p.a',
 'See important information. Offer Up to $300 cash back Looking for a credit card that not only offers a low interest rate but provides up to $300 cash back? Available on a new NAB Low Rate credit card. $100 cash back per month for the first three months from account opening when you spend $500 per month on purchases. Enjoy 0% p.a. for 12 months standard balance transfer (BT) rate. 3% BT fee applies. Unpaid BT reverts to variable cash advance rate. Purchases must be processed and charged to your account in the relevant month to count toward your monthly spend. Minimum monthly repayments required. NAB may vary or end this offer at any time. See important information below. A balance transfer is when you transfer debt from an existing credit card to a new credit card, usually at a new bank. The new bank may offer a lower interest rate for a limited time to help you pay back your debt. Cash back is when you receive credit back to your account']

query = "What is NAB low rate credit card?"
Prompt = f"""Provided the below context :### {context} ### , answer the below query : {query} """

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text([Prompt], 
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Question 2: 
# MAGIC Promotion on NAB Low Rate 

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text(["Is there any promotion running on NAB low rate balance transfer?"], 
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

#Providing with the relevant context from Vector DB Similarity Search

context = ['a. Choose one of our available credit card offers Your choice of offers to suit your credit needs, select either our promotional balance transfer offer or our cash back offer. Offer Promotional 0% p.a. balance transfer (BT) for 32 months with no BT fee Take advantage of our balance transfer offer to help pay off your existing credit card balance/s sooner. Available on new NAB Low Rate credit card. This could help you to consolidate your debt and reduce the amount of interest you pay on it. Try using our to see how much you could save. Enjoy no annual fee for the first year (usually $59). No balance transfer fee applies. Unpaid BT reverts to the variable cash advance rate after 32 months. Minimum monthly repayments required. NAB may vary or end this offer at any time. See important information. Offer Up to $300 cash back Looking for a credit card that not only offers a low interest rate but provides up to $300 cash back? Available on a new NAB Low Rate credit card',
 'Low rate credit card | Balance transfer or cash back offer -  Notification: NAB Mobile Banking app .   NAB Low Rate Credit Card NAB Low Rate Card A simple low interest rate credit card with your choice of offers: get a promotional balance transfer or select our cash back offer. Apply online in 15 minutes and get a response in 60 seconds. Rates and fees Learn more about our standard interest rates and minimum credit limits. Variable purchase rate 12.49% p.a. Interest free days on purchases Up to 55 Annual card fee $59 p.a. Minimum credit limit $1,000 Variable cash advance rate This is the interest rate charged on amounts you withdraw as cash, gambling transactions (including lottery ticket purchases), or transfer from your credit card to another account. 21.74% p.a. Choose one of our available credit card offers Your choice of offers to suit your credit needs, select either our promotional balance transfer offer or our cash back offer. Offer Promotional 0% p.a']

query = "Is there any promotion running on NAB low rate balance transfer?"
Prompt = f"""Provided the below context :### {context} ### , answer the below query : {query} """

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text([Prompt], 
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )

# COMMAND ----------

print(results[0])

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Question 3
# MAGIC Lab low rate fee credit card.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Without any context

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text(["what is a NAB low fee card?"], 
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### With Vector similarity search results

# COMMAND ----------

context = ['NAB Low Fee Credit Card | Lowest annual fee -  Notification: NAB Mobile Banking app .   NAB Low Fee Card NAB Low Fee Card Enjoy our lowest annual card fee of only $30 and our special cashback offer. Apply online in 15 minutes and get a response in 60 seconds. Offer Up to $240 cash back $80 cashback per month for the first three months from account opening when you spend $500 per month on purchases. Purchases must be processed and charged to your account in the relevant month to count toward your monthly spend. Minimum monthly repayments required. NAB may vary or end this offer at any time. See important information below. If you’d like more information, you can read more about how . Rates and fees Learn more about our standard interest rates and minimum credit limits. Variable purchase rate 19.74% p.a. Standard balance transfer (BT) rate for 12 months 3% BT fee applies. Unpaid BT reverts to variable cash advance rate 0% p.a',
           
 'Benefits of this card Get more for your money with complimentary services and special offers through Visa Entertainment. Save on annual fees Reduce ongoing card costs with our lowest annual card fee. Complimentary purchase protection insurance For loss or theft of, or accidental damage to new personal items purchased with your card anywhere in the world, for three months from the date of purchase. Add another cardholder Add an at no extra cost. Visa Entertainment Special offers on shows, events, experiences and movies from . Fraud protection NAB Defence proactively detects fraudulent transactions, so you can enjoy 100% peace of mind with all purchases. Digital wallet payments Tap and pay with the smartphones and wearables you already own and use everyday. Things to consider We want to help you choose the right card, so here are a few things to consider about this card before you apply. Higher interest rates compared to our NAB Low Rate Card. You won’t earn rewards points']

query = "what is a NAB low fee card?"
Prompt = f"""Provided the below context :### {context} ### , answer the below query : {query} """

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text([Prompt], 
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Question 4:

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Without any context

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text(["Give me some suggestions on how i protect my password or PIN"], 
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### With context

# COMMAND ----------

context = ['Never write down or share your PIN with anyone. This also applies to your passwords for internet banking and your devices used to make payments, like smartphones and wearables. Don’t let other people use your card and treat your device like your wallet – always keep it close. Always cover the keypad on an ATM or EFTPOS machine when entering your PIN Keep your receipts and regularly check your transactions in the NAB app to make sure you’re being charged correctly. Using your NAB Visa Debit card safely online You can shop online with your NAB Visa Debit card by entering the card number, expiry date and CVV. Always be careful when entering your card details online. Read the fine print and make sure you’re not accidentally signing up for subscriptions or future payments. If you’re shopping on overseas websites, be aware you may be charged an . Learn more about how to',
 'The difference between debit cards and credit cards With a debit card, you’re spending your own money. You can only spend what’s in your transaction account. A credit card is a way to borrow money or get ‘credit’ from a bank, which gives you funds up to an agreed limit. You can spend up to that agreed limit, but you have to pay the money back by the due date. If you don’t, you’ll be charged interest and other fees may apply. You must be at least 18 years old and meet other eligibility criteria to apply for a . Security tips When you’re using your debit card in-person or online, follow these tips. Sign the back of your new card as soon as you get it. When setting up your PIN, don’t use numbers that are easy to guess, like your birthday . Never write down or share your PIN with anyone. This also applies to your passwords for internet banking and your devices used to make payments, like smartphones and wearables']

query = "Give me some suggestions on how i protect my password or PIN."
Prompt = f"""Provided the below context :### {context} ### , answer the below query : {query} """

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text([Prompt], 
                   temperature=0.5, 
                   max_new_tokens=512, 
                   use_template=True
                   )
print(results[0])
