import json

class DataExtractor:
    def __init__(self, file_path) -> None:
        self.data = {}
        self.training_sentences = []
        self.training_labels = []
        self.labels = []
        self.responses = []
        self.num_classes = None
        self.file_path = file_path
    
    def extract(self):
        print("Data extraction started: \n")
        with open(self.file_path) as file:
            self.data = json.load(file)
            file.close()

        for intent in self.data['intents']:
            for pattern in intent['patterns']:
                self.training_sentences.append(pattern)
                self.training_labels.append(intent['tag'])
            self.responses.append(intent['responses'])
            
            if intent['tag'] not in self.labels:
                self.labels.append(intent['tag'])
                
        self.num_classes = len(self.labels)
        print("Sucessfully! extracted the data.")
