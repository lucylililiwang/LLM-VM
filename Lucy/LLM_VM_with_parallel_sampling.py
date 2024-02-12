# Importing the library
import os
import requests
import json
import concurrent.futures

# We are Define te openai_key
openai_key = os.getenv('LLM_VM_OPENAI_API_KEY')
url = "http://localhost:3002/v1/complete"

# We are Define a function that represents your sampling process
def sample(index):
    # We are Define the prompt for text generation
    prompt_text = "what is the economic situation in canada?"
    
    # We are define te payload for the request
    json_payload = {
        "prompt": prompt_text,
        "context": "",
        "temperature": 0.0,
        "openai_key": openai_key
        # We can also add in "finetune": True if needed
    }
    
    # We are Sending a POST request to the LLM vrtual macine
    response = requests.post(url, data=json.dumps(json_payload))
    # We are Extract the generated text
    generated_text = response.text
    

# We are definethe main function
def main():
    # We are Define the number of samples you want to generated
    num_samples = 10
    
    # We are Create a ThreadPoolExecutor with a maximum of 5 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # We are submit the sampling tasks to the executor
        futures = [executor.submit(sample, i) for i in range(num_samples)]
        
        # We are Waiting for all tasks to complete and retrieve the results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
    # We are Print the results
    for result in results:
        print(result)
        

if __name__ == "__main__":
    main()