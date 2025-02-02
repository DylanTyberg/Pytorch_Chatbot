import torch
from transformers import AutoModelForCausalLM, AdamW
from transformers import GPT2Tokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QMainWindow, QLabel
from PyQt5.QtCore import Qt

class text_box(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            text = self.toPlainText()
            self.parent().generate_response(text)
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
        
        self.output_label = QLabel(self)
        self.output_label.setText("Response will apear here")
        self.output_label.move(50, 120)
        self.output_label.resize(300, 50)
        self.output_label.setWordWrap(True)
                            
        self.resize(900, 600)
        
        self.model, self.tokenizer = self.load_model()
        
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("best_model")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
        
    def generate_response(self, input):
        inputs = self.tokenizer.encode(input, padding=True, truncation=True,  return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)            
        self.output_label.setText(f"Generated {response}")

if __name__ == '__main__':
    app = QApplication([])
    window = GUI()
    window.show()
    app.exec_()


#input = tokenizer.encode(input, return_tensors='pt')
#output = model.generate(input, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)



