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

class emotion_index :
    def __init__(self,texts) -> None:
        self.texts = texts 

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
        return np.round(result,2)
    
    def main(self):
        # Sentiment analysis and sentiment prediction
        sia = SentimentIntensityAnalyzer()
        for text in self.texts:
            self.emotion = []
            sentiment = sia.polarity_scores(text)
            for key , value in sentiment.items():
                if key in 'compound' :
                    compound = value
                    continue
                self.emotion.append(value)
            # print(sentiment['neg'] , sentiment['neu'] , sentiment['pos'])
            _ , __ , ___ = self.emotion_score() , self.emotion_ratios() , self.emotion_variation()
            print(f'emotion score : {_:.2f} , emotion ratio : {__} , emotion variation : {___:.2f} , emotion compound :{compound:.2f}')

if __name__ == '__main__':

    # Text data and corresponding sentiment labels
    texts = ["This is a great product! I really like it.",
            "I'm feeling very frustrated as this investment resulted in a loss.",
            "The market outlook is positive and the stock prices are rising.",
            "The company released a major positive news.",
            "Investor sentiment is low, and they have a pessimistic view of the market.",
            "It's not really prefect."]

    emotion_index(texts).main()
    

# Visualize sentiment scores and predictions
# pt.plot(sentiments, "o-" ,label='Sentiment Score')
# pt.xlabel('Text Index')
# pt.ylabel('Sentiment')
# pt.title('Market Sentiment and Prediction')
# pt.legend()
# pt.show()

# print(sentiments)