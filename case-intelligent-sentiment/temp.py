import jieba
from snownlp import SnowNLP

# 示例文本
text = "这个产品非常好，我非常喜欢。"

# 分词
words = jieba.lcut(text)

# 为了演示，我们创建一个简单的词注字典
word_annotations = {
    "产品": "物品或服务",
    "喜欢": "正面情感的表达"
}

# 添加词注
annotated_words = {word: word_annotations.get(word, "") for word in words}

# 打印分词和词注结果
for word, annotation in annotated_words.items():
    print(f"词语：{word}, 词注：{annotation}")

# 情感分析
sentiment = SnowNLP(text)
print(f"情感倾向得分：{sentiment.sentiments}")  # 得分越接近1表示越正面，越接近0表示越负面
