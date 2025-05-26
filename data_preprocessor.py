from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle


class DataPreprocessor:
    def __init__(self, data_extractor) -> None:
        self.extractor = data_extractor
        self.vocab_size = 1000
        self.embedding_dim = 16
        self.max_seq_length = 20
        self.oov_token = "<OOV>"
        self.training_labels = None
        self.padded_sequences = None
        self.vocab = None
        self.num_classes = data_extractor.num_classes

    def process(self):
        encoder = LabelEncoder()
        encoder.fit(self.extractor.training_labels)
        self.training_labels = encoder.transform(self.extractor.training_labels)

        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_token)
        tokenizer.fit_on_texts(self.extractor.training_sentences)
        self.vocab = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(self.extractor.training_sentences)
        self.padded_sequences = pad_sequences(sequences, truncating='post', maxlen=self.max_seq_length)
        print("Data preprocessing is successful!")

        with open('tokenizer.pickle', 'wb') as file:
            pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

        with open('label_encoder.pickle', 'wb') as file:
            pickle.dump(encoder, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()