# We are import the require library
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import numpy as np

# We are define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
# We are generate sample multi-label data
X, y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=5, n_labels=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We are train a multi-label classifier
classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
classifier.fit(X_train, y_train)


# We are using function to predict labels based on user input
def predict_labels(input_data, classifer, mlb):
    input_data = np.array([input_data])
    predicted_labels = classifer.predict(input_data)
    predicted_labels = mlb.inverse_transform(predicted_labels)[0]
    return predicted_labels

# We are define function to provided guided choice completion
def guided_choice_completion(input_data, classifier, mlb, all_labels):
    predicted_labels = predict_labels(input_data, classifier, mlb)
    remaining_labels = list(set(all_labels) - set(predicted_labels))
    print("Predicted labels:", predicted_labels)
    print("Choose from the remaining labels:", remaining_labels)
    
    
# We are fine-tune funcion
def fine_tune_models(model_name, train_dataset, tokenizer,num_epochs=3, batch_size=8, device="cuda" if torch.cuda.is_availabel() else "cpu"):
    print(f"Fine-tuning model: {model_name}")
    
    # We are load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # We are define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    # We are create data loader for training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # We are fine-tuning the loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
            labels = input_ids.clone()
            
            predicted_labels = [predict_labels(data, classifier, mlb) for data in batch_data]
            batch_labels = [mlb.transform([labels])[0] for labels in predicted_labels]
            labels = torch.tensor(batch_labels).to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        # We are print the average loss for the epoch
        print(f"Average Loss: {total_loss / len(train_loader)}")
        
    # We are save the fine-tuned model
    model.save_pretrained(f"fine_tuned_{model_name}")
    
    
# We are define te entrypoint of the function
if __name__ == "__main__":
    # We are define the model use for data synthesis
    model_name = "facebook/opt-125m"
    
    # We are convert multi-label data to binary labels
    mlb = MultiLabelBinarizer()
    mlb.fit(y_train)
    
    # We are convert the data into CustomDataset format
    train_dataset = CustomDataset(X_train, y_train)
    tokenizer = AutoTokenizer.from_pretrained("tokenizer-name")
    
    # We are fine-tune the model
    fine_tune_models(model_name, train_dataset, tokenizer)
