import torch
from transformers import AutoModelForCausalLM, AdamW
from transformers import GPT2Tokenizer

model = AutoModelForCausalLM.from_pretrained("best_model")
tokenizer = GPT2Tokenizer.from_pretrained("best_tokenizer")

input = "Tell me something interesting?"

inputs = tokenizer(input, truncation=True, return_attention_mask=True,  return_tensors="pt")
print(inputs)
attention_mask = inputs['attention_mask']
outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, attention_mask=attention_mask, no_repeat_ngram_size=2)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)        