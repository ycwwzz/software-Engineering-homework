import unittest
from unittest import mock
from main import read_file, detect_language, cosine_similarity

class TestCode(unittest.TestCase):

    # 测试 read_file 函数是否能正确读取文件内容
    def test_read_file(self):
        # 使用 mock 库来模拟文件对象
        mock_file = mock.mock_open(read_data='这是一个测试文件')
        # 使用 patch 函数来替换 open 函数
        with mock.patch('code.open', mock_file):
            # 读取一个已知内容的文件
            content = read_file('test.txt')
        # 预期的文件内容
        expected = '这是一个测试文件'
        # 断言文件内容是否与预期相等
        self.assertEqual(content, expected)

    # 测试 detect_language 函数是否能正确判断语言
    def test_detect_language(self):
        # 一个中文文本
        chinese_text = '这是一段中文文本'
        # 一个英文文本
        english_text = 'This is an English text'
        # 断言中文文本的语言是中文
        self.assertEqual(detect_language(chinese_text), 'chinese')
        # 断言英文文本的语言是英文
        self.assertEqual(detect_language(english_text), 'english')

    # 测试 cosine_similarity 函数是否能正确计算余弦相似度
    def test_cosine_similarity(self):
        # 两个相同的文本
        text1 = '这是一段相同的文本'
        text2 = '这是一段相同的文本'
        # 两个不同的文本
        text3 = '这是一段不同的文本'
        text4 = 'This is a different text'
        # 断言相同的文本的余弦相似度是1
        self.assertEqual(cosine_similarity(text1, text2), 1)
        # 断言不同语言的文本无法计算余弦相似度，返回None
        self.assertIsNone(cosine_similarity(text3, text4))

if __name__ == "__main__":
    # 运行测试类
    unittest.main()
