from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np


class ChatbotModel:
    def __init__(self, data_processor, dense_units=16):
        self.processor = data_processor
        self.model = Sequential()
        self.dense_1 = Dense(dense_units, activation="relu")
        self.dense_2 = Dense(dense_units, activation="relu")
        self.embedding = Embedding(self.processor.vocab_size, self.processor.embedding_dim, 
                                input_length=self.processor.max_seq_length)
        self.pooling = GlobalAveragePooling1D()
        self.output = Dense(units=self.processor.num_classes, activation="softmax")

    def build(self):
        self.model.add(self.embedding)
        self.model.add(self.pooling)
        self.model.add(self.dense_1)
        self.model.add(self.dense_2)
        self.model.add(self.output)

        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])
        print("Model is created sucessfully!\n")
        return self.model.summary()
    
    def fit(self, batch_size=32, epochs=10, verbose=1, validation_split=0.1):
        self.model.fit(self.processor.padded_sequences, self.processor.training_labels, batch_size=batch_size, epochs=epochs, 
                       verbose=verbose, validation_split=validation_split)
        self.model.save("chatbot_model")
    
    @staticmethod
    def predict(model, text, data, verbose=0):
        tokenizer = None
        encoder = None
        max_seq_length = 20

        with open('tokenizer.pickle', 'rb') as file:
            tokenizer = pickle.load(file)
            file.close()

        with open('label_encoder.pickle', 'rb') as file:
            encoder = pickle.load(file)
            file.close

        padded_sequences = pad_sequences(tokenizer.texts_to_sequences([text]), truncating='post', maxlen=max_seq_length)
        res = model.predict(padded_sequences, verbose=verbose)
        tag = encoder.inverse_transform([np.argmax(res)])

        for i in data['intents']:
            if i['tag'] == tag:
                print("\nChatBot: ", np.random.choice(i['responses']), "\n")