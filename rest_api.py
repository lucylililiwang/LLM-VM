from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load LLM-VM model and tokenizer
model_name = "chat_gpt"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# We are define the endpoint for text generation
@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 50)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    
    generate_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return jsonify({'generated_text': generate_text})


# We are define the entry point of the function
if __name__ == "__main__":
    app.run(debug=True)
