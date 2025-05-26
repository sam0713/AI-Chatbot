from data_extraction import DataExtractor
from data_preprocessor import DataPreprocessor
from model import ChatbotModel
from tensorflow.keras.models import load_model

data_extractor = DataExtractor('datasets.json')
data_extractor.extract()

data_processor = DataPreprocessor(data_extractor)
data_processor.process()

model = ChatbotModel(data_processor)
model.build()

model.fit(epochs=1000)

if __name__ == "__main__":
    model = load_model('chatbot_model')

    while True:
        text = input("Query: ")
        ChatbotModel.predict(model, text, data_extractor.data)
