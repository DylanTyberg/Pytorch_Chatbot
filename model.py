import torch
import pandas as pd
import numpy as np
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

with open("train_data.json", "r", encoding="utf-8") as file:
    train_data = json.load(file)

with open("val_data.json", "r", encoding="utf-8") as file:
    val_data = json.load(file)
    
with open("test_data.json", "r", encoding="utf-8") as file:
    test_data = json.load(file)
    
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_pairs(pairs, max_length=50):
    tokenized_pairs = []
    for input, target in pairs:
        # Join tokens into a single string for tokenization
        input_text = ' '.join(input)
        target_text = ' '.join(target)

        # Tokenize input and target
        input_ids = tokenizer.encode(input_text, add_special_tokens=True, truncation=True, max_length=max_length)
        target_ids = tokenizer.encode(target_text, add_special_tokens=True, truncation=True, max_length=max_length)

        # Pad sequences manually to max_length
        input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
        target_ids = target_ids + [tokenizer.pad_token_id] * (max_length - len(target_ids))

        # Append tokenized and padded pair
        tokenized_pairs.append((input_ids[:max_length], target_ids[:max_length]))

    return tokenized_pairs

tokenized_test_data = tokenize_pairs(test_data)
tokenized_train_data = tokenize_pairs(train_data)
tokenized_val_data = tokenize_pairs(val_data)

class CornellDataset(Dataset):
    def __init__(self, tokenized_data, device):
        self.data = tokenized_data
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids, target_ids = self.data[idx]
        return {
            'input_ids' : torch.tensor(input_ids, dtype=torch.long).to(self.device),
            'target_ids': torch.tensor(target_ids, dtype=torch.long).to(self.device)
        }
        
train_dataset = CornellDataset(tokenized_train_data, device)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

test_dataset = CornellDataset(tokenized_test_data, device)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

val_dataset = CornellDataset(tokenized_val_data, device)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

split = int(len(tokenized_train_data) * 0.02)
partial_train_data = tokenized_train_data[:split]
partial_train_dataset = CornellDataset(partial_train_data, device)
partial_train_loader = DataLoader(partial_train_dataset, batch_size=2, shuffle=True)
    
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

epochs = 3
best_val_loss = float('inf')
for epoch in range(epochs):
    print(epoch)
    model.train()
    train_loss = 0
    for batch in partial_train_loader:
        input_ids = batch['input_ids']
        optimizer.zero_grad()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    train_loss = train_loss / len(train_loader)
        
    model.eval()
    val_loss = 0
    with torch.no_grad():
    #    for batch in val_loader:
    #        outputs = model(batch)
    #        loss = outputs.loss
    #        val_loss += loss.item()
    #
    #val_loss = val_loss / len(val_loader)
    #if val_loss < best_val_loss:
    #    best_val_loss = val_loss
        model.save_pretrained("best_model")
        tokenizer.save_pretrained("best_tokenizer")


print('success')

