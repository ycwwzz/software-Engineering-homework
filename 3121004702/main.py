# 导入所需的库
import sys

import jieba  # 中文分词库
import nltk  # 英文分词库
import numpy as np  # 数学计算库
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF加权方法
from line_profiler_pycharm import profile


# 定义一个函数来读取文件内容并返回一个字符串
@profile
def read_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


# 定义一个函数来判断文件内容是中文还是英文
@profile
def detect_language(text):
    # 使用nltk库中的stopwords列表来判断
    # nltk.download("stopwords")
    stopwords = nltk.corpus.stopwords.words('english')
    # 统计text中出现的英文停用词的数量
    count = 0
    for word in text.split():
        if word.lower() in stopwords:
            count += 1
    # 如果出现的英文停用词数量超过text单词总数的一半，则认为是英文，否则认为是中文
    if count > len(text.split()) / 2:
        return 'english'
    else:
        return 'chinese'


# 定义一个函数来计算两个字符串的余弦相似度
@profile
def cosine_similarity(str1, str2):
    # 判断两个字符串的语言是否一致
    lang1 = detect_language(str1)
    lang2 = detect_language(str2)
    if lang1 != lang2:
        print('两个文件的语言不一致，请检查输入')
        return None
    # 根据语言选择分词库
    if lang1 == 'chinese':
        tokenizer = jieba.cut  # 中文分词函数
    else:
        tokenizer = nltk.word_tokenize  # 英文分词函数
    # 使用分词函数进行分词
    words1 = tokenizer(str1)
    words2 = tokenizer(str2)
    # 使用空格连接分词结果
    text1 = ' '.join(words1)
    text2 = ' '.join(words2)
    # 使用Tfidf Vectorizer类来转换成词袋模型并加权
    vectorizer = TfidfVectorizer()
    # 计算两个文本的TF-IDF矩阵
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    # 转换成numpy数组
    tfidf_array = tfidf_matrix.toarray()
    # 取出两个向量
    vector1 = tfidf_array[0]
    vector2 = tfidf_array[1]
    # 计算两个向量的点积
    dot_product = np.dot(vector1, vector2)
    # 计算两个向量的模长
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    # 计算余弦相似度
    cos_sim = dot_product / (norm1 * norm2)
    return cos_sim


if __name__ == "__main__":
    original_file = sys.argv[1]  # 原文文件路径
    plagiarized_file = sys.argv[2]  # 抄袭版文件路径
    output_file = sys.argv[3]  # 输出文件路径

    # # 定义原文文件和抄袭版文件的路径
    # original_file = 'original.txt'
    # plagiarized_file = 'plagiarized.txt'
    # 读取文件内容
    original_text = read_file(original_file)
    plagiarized_text = read_file(plagiarized_file)
    # 计算余弦相似度
    similarity = cosine_similarity(original_text, plagiarized_text)
    # 计算重复率

    repetition_percentage = round(similarity * 100, 2)

    print(f'重复率为{repetition_percentage}%')
    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'重复率为{repetition_percentage}%')
