from torch.utils.data import DataLoader, Dataset
import os
import re
import sklearn
import json
from nltk.tokenize import word_tokenize

lines_file = "CornellData/movie_lines.txt"
conversations_file = "CornellData/movie_conversations.txt"

def load_lines(file_path):
    """
    opens the file specified by file_path and creates a dictonary with Ids mapped to 
    the text. Returns lines dictionary
    """
    lines = {}
    with open(file_path, encoding="utf-8", errors='ignore') as file:
        for line in file:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 5:
                lines[parts[0]] = parts[4]
    return lines

def load_conversattions(file_path, lines):
    """
    opens file and retrieves line ids from each line. Calls parse_line_ids on the list of ids
    then checks if the line Ids are all in the lines dictionary. Adds the line Ids with the corresponding text to 
    a conversation and appends it to the list of conversations.
    """
    conversations = []
    def parse_line_ids(raw_ids):
        """
        takes in raw_ids, gets rid of any [] and replaces ' with whitespace. Then splits 
        by , delimeter and returns result.
        """
        raw_ids = raw_ids.strip("[]")
        line_ids = raw_ids.replace("'", "").split(", ")
        return line_ids
    
    with open(file_path, encoding='utf-8', errors='ignore') as file:
        for line in file:
            parts = line.strip().split("+++$+++")
            if len(parts) == 4:
                raw_ids = parts[3]
                line_ids = parse_line_ids(raw_ids)
                existing_ids = [line_id for line_id in line_ids if line_id in lines]
                conversation = [lines[line_id] for line_id in existing_ids]
                if conversation:
                    conversations.append(conversation)
    return conversations

lines = load_lines(lines_file)
conversations = load_conversattions(conversations_file, lines)

def extract_dialogue_pairs(conversations):
    """
    loops through conversation list and creates pairs for each piece of text containing
    the text and then the response.
    """
    pairs = []
    for conversation in conversations:
        for i in range(len(conversation) - 1):
            input_line = conversation[i].strip()
            target_line = conversation[i + 1].strip()
            if input_line and target_line:
                pairs.append((input_line, target_line))
    return pairs

dialogue_pairs = extract_dialogue_pairs(conversations)

def preprocess_text(text):
    text = text.lower()
    #removes anything that isn't a letter or digit
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    #replaces consecutive whitespace with one whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

preprocessed_pairs = [(preprocess_text(input_line), preprocess_text(target_line)) 
                      for input_line, target_line in dialogue_pairs]

#splliting the data into train, test, validation.
from sklearn.model_selection import train_test_split
train_pairs, test_pairs = train_test_split(preprocessed_pairs, test_size=0.1, random_state=42)
train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.1, random_state=42)
print(f"Train: {len(train_pairs)}, Validation: {len(val_pairs)}, Test: {len(test_pairs)}")

#import nltk
#nltk.download("punkt_tab")

#def tokenize_pairs(pairs):
#    return [[word_tokenize(input), word_tokenize(target)] for input, target in pairs]
    
#tokenized_train = tokenize_pairs(train_pairs)
#tokenized_val = tokenize_pairs(val_pairs)
#tokenized_test = tokenize_pairs(test_pairs)

with open("train_data.json", "w") as f:
    json.dump(train_pairs, f)
with open("val_data.json", "w") as f:
    json.dump(val_pairs, f)
with open("test_data.json", "w") as f:
    json.dump(test_pairs, f)
