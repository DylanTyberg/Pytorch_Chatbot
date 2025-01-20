import torch
from transformers import AutoModelForCausalLM, AdamW
from transformers import GPT2Tokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit

app = QApplication([])

window = QWidget()
window.setWindowTitle("Test")
text_edit = QTextEdit(window)
window.show()
app.exec_()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id

#input = tokenizer.encode(input, return_tensors='pt')
#output = model.generate(input, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)



