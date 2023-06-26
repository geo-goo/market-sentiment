import nltk
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as pt

# Download the required resources for NLTK
nltk.data.path.append("/home/geo/Downloads/Trade/text-mining-bot/ntlk_data/")
# nltk.download('vader_lexicon',download_dir="./ntlk_data/")

# Text data and corresponding sentiment labels
texts = ["This is a great product! I really like it.",
         "I'm feeling very frustrated as this investment resulted in a loss.",
         "The market outlook is positive and the stock prices are rising.",
         "The company released a major positive news.",
         "Investor sentiment is low, and they have a pessimistic view of the market.",
         "It's not really prefect."]

# Sentiment analysis and sentiment prediction
sia = SentimentIntensityAnalyzer()
sentiments = []

class emotion_index :
    def __init__(self,emotion) -> None:
        self.emotion = emotion

    def emotion_score(self):
        # emotion_weights = {'negative': 0.4, 'neutral': 0.2,'positive': 0.6}
        weights = np.array([0.4,0.2,0.6])
        result = self.emotion * weights
        return result.sum()

    def emotion_variation(self):
        result = max(self.emotion) - min(self.emotion)    
        return result

    def emotion_ratios(self):
        emotion = np.array(self.emotion)
        result = emotion / emotion.sum()
        return result

if __name__ == '__main__':

    for text in texts:
        emotion = []
        sentiment = sia.polarity_scores(text)
        for key , value in sentiment.items():
            if key in 'compound' :
                compound = value
                continue
            emotion.append(value)
        # print(sentiment['neg'] , sentiment['neu'] , sentiment['pos'])
        e = emotion_index(emotion)
        _ , __ , ___ = e.emotion_score() , e.emotion_ratios() , e.emotion_variation()
        print(f'{_:.2f} , {__} , {___:.2f} , {compound:.2f}')
    

# Visualize sentiment scores and predictions
# pt.plot(sentiments, "o-" ,label='Sentiment Score')
# pt.xlabel('Text Index')
# pt.ylabel('Sentiment')
# pt.title('Market Sentiment and Prediction')
# pt.legend()
# pt.show()

# print(sentiments)