import hanlp

# 初始化分词器
tokenizer = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)  # 或选择其他预训练模型

# 待处理的文本
text = "我爱北京天安门"

# 进行分词和词性标注
tokens = tokenizer(text)

# 打印结果
for word, tag in zip(tokens[0], tokens[1]):
    print(f"{word}/{tag}", end=' ')