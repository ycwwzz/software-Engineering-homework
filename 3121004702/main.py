# 导入所需的库
import difflib


# 定义一个函数来读取文件内容并返回一个字符串
def read_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


if __name__ == "__main__":
    # original_file = sys.argv[1]  # 原文文件路径
    # plagiarized_file = sys.argv[2]  # 抄袭版文件路径
    # output_file = sys.argv[3]  # 输出文件路径
    #
    # # 读取文件内容
    # with open(original_file, 'r', encoding='utf-8') as f:
    #     original_text = f.read()
    # with open(plagiarized_file, 'r', encoding='utf-8') as f:
    #     plagiarized_text = f.read()

    # 定义原文文件和抄袭版文件的路径
    original_file = 'original.txt'
    plagiarized_file = 'plagiarized.txt'
    # 读取文件内容
    original_text = read_file(original_file)
    plagiarized_text = read_file(plagiarized_file)

    # 计算余弦相似度
    similarity = difflib.SequenceMatcher(None, original_text, plagiarized_text).ratio()

    # 计算重复率
    repetition_rate = similarity
    # 输出结果

    # 将相似度转换为百分比形式
    repetition_percentage = round(repetition_rate * 100, 2)

    print(f'重复率为{repetition_percentage}%')

    # 将结果写入输出文件
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     f.write(f'重复率为{repetition_percentage}%')
