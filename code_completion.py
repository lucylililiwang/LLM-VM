# We are import the require library
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# We are define the custom dataset class if needed
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    
# We are define the fine-tuning function
def fine_tune_models(model_names, train_dataset,tokenizer, num_epochs=3, batch_size=8, device="cuda" if torch.cuda.is_availabel() else "cpu"):
    for model_name in model_names:
        print(f"Fine-tuning model: {model_name}")
        
        # We are load pre-trained model
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # We are define the optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_sie=1, gamma=0.9)
        
        # We are create data loader for training
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # We are fine-tuning loop
        for epcoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                input_ids = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
                labels = input_ids.clone()
                
                optimizer.zero_grad()
                outputs = model(input_ids, labels=labels)
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            # Print the average loss for the epoch
            print(f"Average Loss: {total_loss / len(train_loader)}")

        # Save the fine-tuned model
        model.save_pretrained(f"fine_tuned_{model_name}")
        

# We are define the code completion function
def code_completion(input_code, model,tokenizer, device="cuda" if torch.cuda.is_availabel() else "cpu"):
    model.eval()
    input_ids = tokenizer.encode(input_code, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=100, num_return_sequences=5)
    completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    return completions


# We are define the entrypoint function
if __name__ == "__main__":
    # We are define the models to fine-tune
    model_names = [
        "facebook/opt-125m",
        "facebook/opt-350m",
    ]
    
    # We are load preprocessed data and tokenizer
    train_dataset = CustomDataset(...)
    tokenizer = AutoTokenizer.from_pretrained("tokenizer-name")
    
    # We are fine tune the models
    fine_tune_models(model_names, train_dataset, tokenizer)
    
    # We are load the code completion model and tokenizer
    code_completion_model = AutoModelForCausalLM.from_pretrained("code-completion-model").to(device)
    
    code_completion_tokenizer = AutoTokenizer.from_pretrained("code-completion-tokenizer")
    
    # We are define the usage of code completion
    input_code = "import numpy as np\nnp."
    completions = code_completion(input_code, code_completion_model, code_completion_tokenizer)
    print("Code completions:")
    for completion in completions
