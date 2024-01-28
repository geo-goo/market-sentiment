from snownlp import SnowNLP

def classify_sentiment(text):
    s = SnowNLP(text)

    # 获取情感分数
    sentiment_score = s.sentiments

    # 将情感分数转换为类别
    if sentiment_score > 0.6:
        return 1  # 正面
    elif sentiment_score < 0.4:
        return -1  # 负面
    else:
        return 0  # 中性

# 测试一些示例
examples = ["我非常喜欢这个", "这真是太糟糕了", "我觉得还可以"]

for example in examples:
    sentiment = classify_sentiment(example)
    print(f"文本: '{example}'，情感分类: {sentiment}")
