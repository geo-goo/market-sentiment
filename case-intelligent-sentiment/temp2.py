import thulac

# 初始化Thulac分词器

thu = thulac.thulac(user_dict='/home/geo/Downloads/geo/text_mining_bot/case-intelligent-auditing/custom.txt')

# 要分词和词性标注的文本
text = "这是一个简单的例子m,不是要做爱用于演示Thulac的使用方法。"

# 执行分词和词性标注
result = thu.cut(text)

# 打印结果
for word, pos in result:
    print(f"词语: {word}, 词性: {pos}")

