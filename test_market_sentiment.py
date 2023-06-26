import nltk
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
labels = [1, -1, 1, 1, -1, -1]  # Positive sentiment as 1, negative sentiment as -1

# Data preprocessing and feature extraction
# vectorizer = TfidfVectorizer()
# features = vectorizer.fit_transform(texts)
# # print(features)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# # Model training
# model = SVC(kernel='linear')
# model.fit(X_train, y_train)

# Sentiment analysis and sentiment prediction
sia = SentimentIntensityAnalyzer()
sentiments = []

for text in texts:
    sentiment = sia.polarity_scores(text)
    sentiments.append(sentiment['compound'])

# predictions = model.predict(features)

# Output sentiment analysis and sentiment prediction results
# for i, text in enumerate(texts):
#     print(f"Text: {text}")
#     print(f"Sentiment Analysis Score: {sentiments[i]}")
#     print(f"Sentiment Prediction: {'Positive.' if predictions[i] == 1 else 'Negative or Cant Predict.'}")
#     print("-----")

# Visualize sentiment scores and predictions
pt.plot(sentiments, "o-" ,label='Sentiment Score')
# pt.plot(predictions, linestyle='--', label='Sentiment Prediction')
pt.xlabel('Text Index')
pt.ylabel('Sentiment')
pt.title('Market Sentiment and Prediction')
pt.legend()
pt.show()

"""
# 情绪得分(Emotion Score)计算方法
# 假设情感分析结果为每种情绪的得分字典，如 {'positive': 0.8, 'negative': 0.2, 'neutral': 0.0}

emotion_scores = {'positive': 0.8, 'negative': 0.2, 'neutral': 0.0}

# 计算总的情绪得分
total_score = sum(emotion_scores.values())

# 打印每种情绪的得分和占比
for emotion, score in emotion_scores.items():
    emotion_ratio = score / total_score if total_score != 0 else 0
    print(f"{emotion}: Score = {score}, Ratio = {emotion_ratio:.2f}")

"""

"""
# 情绪比例(Emotion Ratio)计算方法
# 假设情感分析结果为每种情绪的计数，如 {'positive': 80, 'negative': 20, 'neutral': 0}

emotion_counts = {'positive': 80, 'negative': 20, 'neutral': 0}

# 计算总的情绪计数
total_count = sum(emotion_counts.values())

# 打印每种情绪的比例
for emotion, count in emotion_counts.items():
    emotion_ratio = count / total_count if total_count != 0 else 0
    print(f"{emotion}: Count = {count}, Ratio = {emotion_ratio:.2f}")
"""

"""
# 情绪变化(Emotion Variation)计算方法
# 假设情感分析结果为每天的情绪得分列表，如 [0.8, 0.6, 0.5, 0.7, 0.9]

emotion_scores = [0.8, 0.6, 0.5, 0.7, 0.9]

# 计算情绪得分的变化幅度
variation = max(emotion_scores) - min(emotion_scores)

print(f"Emotion Variation: {variation}")

"""
