import torch
from transformers import AutoModelForCausalLM, AdamW
from transformers import GPT2Tokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QMainWindow, QLabel
from PyQt5.QtCore import Qt

#creating text_box class that inherits from PyQT QTextEdit
class text_box(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    #if enter key is pressed it will input text into model and generate response
    #else calls QTextEdit keyPressEvent function
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            text = self.toPlainText()
            self.parent().generate_response(text)
            event.accept()
        else:
            super().keyPressEvent(event)

#creating GUI class to add functions to load model and generate response
class GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        #setting up the window and text box
        self.setWindowTitle("Test")
        self.text_edit = text_box(self)
        self.text_edit.resize(300, 50)
        self.text_edit.move(600, 500)
        
        #setting up the label representing the model's response
        self.output_label = QLabel(self)
        self.output_label.setText("Response will apear here")
        self.output_label.move(50, 120)
        self.output_label.resize(300, 50)
        self.output_label.setWordWrap(True)
                            
        self.resize(900, 600)
        
        self.model, self.tokenizer = self.load_model()
    
        
    def load_model(self):
        """
        loads the saved model and tokenizer 
        """
        model = AutoModelForCausalLM.from_pretrained("best_model")
        tokenizer = GPT2Tokenizer.from_pretrained("best_tokenizer")
        return model, tokenizer
        
    def generate_response(self, input):
        """
        tokenizes input, then gives it to the model to generate a response. Sets the text
        on the window to be the response.
        """
        inputs = self.tokenizer(input, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1,
                                       attention_mask=inputs['attention_mask'], no_repeat_ngram_size=2)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)         
        self.output_label.setText(f"{response}")

if __name__ == '__main__':
    app = QApplication([])
    window = GUI()
    window.show()
    app.exec_()




