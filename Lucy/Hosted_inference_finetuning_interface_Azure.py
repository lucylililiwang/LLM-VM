# We are importing the require library
from flask import Flask, request, jsonify
from llm_vm.client import Client
from transformers import GPT3LMHeadModel, GPT3Tokenizer
import sys
import os

# We are define the app
app = Flask(__name__)

@app.route('/fine-tune', methods=['POST'])
def fine_tune():
    # Extract input data from the request
    input_data = request.json.get('input_data')

    # Define fine-tuning prompt and context
    prompt = "Answer question Q."
    context = "Q: What is the currency in Myanmar"

    # Define the fine-tuning parameters
    fine_tuning_params = {
        'openai_key': os.getenv('LLM_VM_OPENAI_API_KEY'),
        'temperature': 0.0,
        'data_synthesis': True,
        'finetune': True
    }

    # Define the LLM models to fine-tune
    model_uris = [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
        "facebook/opt-66b",
        "openlm-research/open_llama_3b_v2",
        "openlm-research/open_llama_7b_v2",
        "openlm-research/open_llama_13b",
        "bigscience/bloom-560m",
        "bigscience/bloom-1b7",
        "bigscience/bloom-3b",
        "bigscience/bloom-7b1",
        "EleutherAI/gpt-125m",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
        "EleutherAI/gpt-neox-20b"
    ]

    # Fine-tune the models
    fine_tune_results = fine_tune_models(model_uris, prompt, context, fine_tuning_params)

    return jsonify(fine_tune_results)

def fine_tune_models(model_uris, prompt, context, fine_tuning_params):
    fine_tune_results = {}
    for model_uri in model_uris:
        # Instantiate the client specifying which LLM you want to use
        client = Client(big_model=model_uri, small_model='pythia')

        # Load the pre-trained GPT-3 model and tokenizer
        model = GPT3LMHeadModel.from_pretrained(model_uri)
        tokenizer = GPT3Tokenizer.from_pretrained(model_uri)

        # Tokenize the prompt and context
        input_ids = tokenizer.encode(prompt + context, return_tensors="pt")

        # Fine-tune the model with the provided prompt and context
        response = client.complete(model=model, input_ids=input_ids, **fine_tuning_params)

        # Store the fine-tuning result
        fine_tune_results[model_uri] = response

        # Print response to stderr
        print(f"Model: {model_uri}, Response: {response}", file=sys.stderr)

    return fine_tune_results

if __name__ == '__main__':
    app.run(debug=True)