#! /usr/bin/env python3
# import our client
import sys
from llm_vm.client import Client
import os
from llm_vm.config import settings
from transformers import GPT3LMHeadMode, GPT3Tokenizer
# First, we are defining an main function
def main():
    # We are Instantiate the client specifying which LLM we wish to use
    client = Client(big_model='chat_gpt', small_model='pythia')
    
    # We are define the fine-tuning prompt and context
    prompt = "Answer question Q."
    context = "Q: What is the currency in Myanmar"
    
    # We are Load the pre-trained GPT-3 model and tokenizer
    model_name = "gpt3"
    model = GPT3LMHeadMode.from_pretrained(model_name)
    tokenizer = GPT3Tokenizer.from_pretrained(model_name)
    
    # We are Tokenize the prompt and context
    input_ids = tokenizer.encode(prompt + context, return_tensors="pt")
    
    # We are Fine-tuning parameters
    fine_tuning_params = {
        'openai_key': settings.openai_api_key,
        'temperature': 0.0,
        'data_synthesis': True,
        'finetune': True
    }
    
    # We are Cnvert input_ids to torch tensor
    input_ids = input_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # We are Fine-tune the model with the provided prompt and context
    response = client.complete(model=model, input_ids=input_ids, **fine_tuning_params)
    
    # We are Print the response to stderr
    print(response, file=sys.stderr)
    

# we are defining the entry point of the function
if __name__ == "__main__":
    main()
# Instantiate the client specifying which LLM you want to use
# client = Client(big_model='chat_gpt', small_model='pythia')

# # Put in your prompt and go!
# response = client.complete(prompt = "Answer question Q. ",context="Q: What is the currency in myanmmar",
#                            openai_key=settings.openai_api_key,
#                            temperature=0.0,
#                            data_synthesis=True,
#                            finetune=True,)
# print(response, file=sys.stderr)

# # response = client.complete(prompt = "Answer question Q. ",context="Q: What is the economic situation in France",
# #                            openai_key=settings.openai_api_key,
# #                            temperature=0.0,
# #                            data_synthesis=True,
# #                            finetune=True,)
# # print(response)
# # response = client.complete(prompt = "Answer question Q. ",context="Q: What is the currency in myanmmar",
# #                            openai_key=settings.openai_api_key,
# #                            temperature=0.0,
# #                            data_synthesis=True,
# #                            finetune=True,)
# # print(response)
# # Anarchy is a political system in which the state is abolished and the people are free...

# 