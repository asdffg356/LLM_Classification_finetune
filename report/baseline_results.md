

## TF-IDF + Logistic Regression Baseline

模型采用 TF-IDF 文本特征和多分类 Logistic Regression。输入由 prompt、response_a 和 response_b 拼接而成，不使用 model_a/model_b 模型名称信息。

最佳参数：
- max_features = 100000
- ngram_range = (1, 2)
- C = 0.1

本地验证集结果：
- log loss = 1.080877
- accuracy = 0.385264
- macro F1 = 0.383576

Kaggle 提交结果：
- Public Score = 1.07737
- Public Rank = 165

结果分析：
调参后的 TF-IDF + Logistic Regression 在本地验证集和 Kaggle Public 测试集上的 log loss 均低于随机三分类的 ln(3)=1.0986，说明模型能够从 prompt 与两个回答的词面特征中学习到一定的人类偏好规律。但整体准确率仍不高，说明仅依赖词频特征难以充分理解回答质量、逻辑正确性和语义差异。