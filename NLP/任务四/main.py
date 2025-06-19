import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words


def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    all_words = []
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]


def build_feature_matrix(feature_method='frequency', top_num=100):
    """
    构建特征矩阵，可以选择使用高频词特征或TF-IDF加权特征

    Args:
        feature_method: 特征提取方法，'frequency'表示高频词特征，'tfidf'表示TF-IDF加权特征
        top_num: 高频词特征的词数量

    Returns:
        vector: 特征矩阵
        top_words: 特征词列表
        vectorizer: 向量化器（仅在使用TF-IDF时有效）
    """
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]

    if feature_method == 'frequency':
        # 高频词特征
        top_words = get_top_words(top_num)

        all_words = []
        for filename in filename_list:
            all_words.append(get_words(filename))

        vector = []
        for words in all_words:
            word_map = list(map(lambda word: words.count(word), top_words))
            vector.append(word_map)

        vector = np.array(vector)
        return vector, top_words, None

    elif feature_method == 'tfidf':
        # TF-IDF加权特征
        corpus = []
        for filename in filename_list:
            words = get_words(filename)
            corpus.append(' '.join(words))

        vectorizer = TfidfVectorizer()
        vector = vectorizer.fit_transform(corpus).toarray()
        top_words = vectorizer.get_feature_names_out()

        return vector, top_words, vectorizer

    else:
        raise ValueError("feature_method参数必须是'frequency'或'tfidf'")


# 训练模型
feature_method = 'tfidf'  # 可以切换为'frequency'使用高频词特征
vector, top_words, vectorizer = build_feature_matrix(feature_method, top_num=100)

# 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
labels = np.array([1] * 127 + [0] * 24)

model = MultinomialNB()
model.fit(vector, labels)


def predict(filename, feature_method='frequency', top_words=None, vectorizer=None):
    """对未知邮件分类"""
    words = get_words(filename)

    if feature_method == 'frequency':
        # 高频词特征
        if top_words is None:
            raise ValueError("使用高频词特征时必须提供top_words")

        current_vector = np.array(
            list(map(lambda word: words.count(word), top_words)))

    elif feature_method == 'tfidf':
        # TF-IDF加权特征
        if vectorizer is None:
            raise ValueError("使用TF-IDF特征时必须提供vectorizer")

        current_vector = vectorizer.transform([' '.join(words)]).toarray()[0]

    else:
        raise ValueError("feature_method参数必须是'frequency'或'tfidf'")

    # 预测结果
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'


# 测试预测
print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt', feature_method, top_words, vectorizer)))
print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt', feature_method, top_words, vectorizer)))
print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt', feature_method, top_words, vectorizer)))
print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt', feature_method, top_words, vectorizer)))
print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt', feature_method, top_words, vectorizer)))