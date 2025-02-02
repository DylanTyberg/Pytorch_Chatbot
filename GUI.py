import torch
from transformers import AutoModelForCausalLM, AdamW
from transformers import GPT2Tokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QMainWindow
from PyQt5.QtCore import Qt

class text_box(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            text = self.toPlainText()
            print(text)
            event.accept()
        else:
            super().keyPressEvent(event)

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Test")
        self.text_edit = text_box(self)
        self.text_edit.resize(300, 50)
        self.text_edit.move(600, 500)
                            
        self.resize(900, 600)



if __name__ == '__main__':
    app = QApplication([])
    window = GUI()
    window.show()
    app.exec_()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id 


#input = tokenizer.encode(input, return_tensors='pt')
#output = model.generate(input, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)



