# 垃圾邮件分类器 - 支持多种特征提取模式
本程序基于朴素贝叶斯算法实现邮件垃圾/正常分类，\
支持高频词特征和TF-IDF加权特征两种提取模式.


import re\
import os\
from jieba import cut\
from itertools import chain\
from collections import Counter\
import numpy as np\
from sklearn.naive_bayes import MultinomialNB\
from sklearn.feature_extraction.text import TfidfVectorizer

class SpamClassifier:\
    """垃圾邮件分类器主类"""
    
    def __init__(self, feature_method='frequency', top_num=100):
        """
        初始化分类器
        
        Args:
            feature_method: 特征提取方法，支持'frequency'（高频词）和'tfidf'（TF-IDF）
            top_num: 高频词特征模式下选择的高频词数量（默认100）
        """
        self.feature_method = feature_method
        self.top_num = top_num
        self.top_words = None      # 特征词列表
        self.vectorizer = None     # TF-IDF向量化器
        self.model = MultinomialNB()
    
    def get_words(self, filename):
        """
        读取文本并过滤无效字符和长度为1的词
        
        Args:
            filename: 文件名
            
        Returns:
            words: 处理后的词语列表
        """
        words = []
        try:
            with open(filename, 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = line.strip()
                    # 过滤无效字符
                    line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
                    # 使用jieba分词
                    line = cut(line)
                    # 过滤长度为1的词
                    line = filter(lambda word: len(word) > 1, line)
                    words.extend(line)
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")
        return words
    
    def build_feature_matrix(self, file_list):
        """
        构建特征矩阵，支持两种特征提取模式
        
        Args:
            file_list: 训练文件列表
            
        Returns:
            feature_matrix: 特征矩阵
        """
        if self.feature_method == 'frequency':
            return self._build_frequency_features(file_list)
        elif self.feature_method == 'tfidf':
            return self._build_tfidf_features(file_list)
        else:
            raise ValueError(f"不支持的特征方法: {self.feature_method}")
    
    def _build_frequency_features(self, file_list):
        """构建高频词特征矩阵"""
        # 收集所有文档的词汇
        all_words = []
        for filename in file_list:
            words = self.get_words(filename)
            all_words.append(words)
        
        # 统计词频并获取高频词
        freq = Counter(chain(*all_words))
        self.top_words = [i[0] for i in freq.most_common(self.top_num)]
        
        # 构建特征矩阵
        feature_matrix = []
        for words in all_words:
            # 统计每个高频词在当前文档中出现的次数
            word_map = [words.count(word) for word in self.top_words]
            feature_matrix.append(word_map)
        
        return np.array(feature_matrix)
    
    def _build_tfidf_features(self, file_list):
        """构建TF-IDF加权特征矩阵"""
        # 构建语料库
        corpus = []
        for filename in file_list:
            words = self.get_words(filename)
            corpus.append(' '.join(words))
        
        # 使用TfidfVectorizer计算特征
        self.vectorizer = TfidfVectorizer()
        feature_matrix = self.vectorizer.fit_transform(corpus).toarray()
        self.top_words = self.vectorizer.get_feature_names_out()
        
        return feature_matrix
    
    def train(self, file_list, labels):
        """
        训练分类器
        
        Args:
            file_list: 训练文件列表
            labels: 对应的标签列表 (1表示垃圾邮件，0表示正常邮件)
        """
        # 构建特征矩阵
        feature_matrix = self.build_feature_matrix(file_list)
        
        # 训练模型
        self.model.fit(feature_matrix, labels)
        print(f"模型训练完成，使用特征方法: {self.feature_method}")
    
    def predict(self, filename):
        """
        对未知邮件进行分类
        
        Args:
            filename: 待分类的邮件文件名
            
        Returns:
            '垃圾邮件' 或 '普通邮件'
        """
        words = self.get_words(filename)
        
        if self.feature_method == 'frequency':
            # 构建高频词特征向量
            feature_vector = np.array([words.count(word) for word in self.top_words])
        else:  # TF-IDF
            # 构建TF-IDF特征向量
            feature_vector = self.vectorizer.transform([' '.join(words)]).toarray()[0]
        
        # 预测结果
        result = self.model.predict(feature_vector.reshape(1, -1))
        return '垃圾邮件' if result[0] == 1 else '普通邮件'


def demo_tfidf_mode():\
    """演示TF-IDF特征模式的使用"""\
    print("\n=== TF-IDF特征模式演示 ===")
    
    # 创建分类器实例
    classifier = SpamClassifier(feature_method='tfidf')
    
    # 准备训练数据
    train_files = [f'邮件_files/{i}.txt' for i in range(151)]
    labels = np.array([1]*127 + [0]*24)  # 0-126为垃圾邮件，127-150为正常邮件
    
    # 训练模型
    classifier.train(train_files, labels)
    
    # 预测测试文件
    test_files = [f'邮件_files/{i}.txt' for i in range(151, 156)]
    for file in test_files:
        result = classifier.predict(file)
        print(f'{os.path.basename(file)} 分类结果: {result}')


def demo_frequency_mode():\
    """演示高频词特征模式的使用"""\
    print("\n=== 高频词特征模式演示 ===")\
    
    # 创建分类器实例
    classifier = SpamClassifier(feature_method='frequency', top_num=150)
    
    # 准备训练数据
    train_files = [f'邮件_files/{i}.txt' for i in range(151)]
    labels = np.array([1]*127 + [0]*24)  # 0-126为垃圾邮件，127-150为正常邮件
    
    # 训练模型
    classifier.train(train_files, labels)
    
    # 预测测试文件
    test_files = [f'邮件_files/{i}.txt' for i in range(151, 156)]
    for file in test_files:
        result = classifier.predict(file)
        print(f'{os.path.basename(file)} 分类结果: {result}')


if __name__ == "__main__":\
    # 显示文档说明\
    print("="*50)\
    print("垃圾邮件分类器 - 特征选择模式说明")\
    print("="*50)\
    print("一、核心功能概述")\
    print("本分类器基于朴素贝叶斯算法实现邮件垃圾/正常分类，支持两种特征提取模式：")\
    print("1. 高频词特征模式：基于词频统计选择最常出现的词汇构建特征向量")\
    print("2. TF-IDF加权特征模式：基于词频-逆文档频率计算特征权重，突出有类别区分力的词汇")\
    print("\n更多详细说明请参考代码注释")\
    print("="*50, "\n")
    
    # 运行演示
    demo_tfidf_mode()
    demo_frequency_mode()



