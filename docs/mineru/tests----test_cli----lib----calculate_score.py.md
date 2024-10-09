# `.\MinerU\tests\test_cli\lib\calculate_score.py`

```
"""
calculate_score
"""  # 模块说明，表示该模块的功能是计算得分
import os  # 导入操作系统功能模块
import re  # 导入正则表达式模块，用于字符串处理
import json  # 导入 JSON 模块，用于处理 JSON 数据
from Levenshtein import distance  # 从 Levenshtein 库导入计算编辑距离的函数
from lib import scoring  # 从自定义库 lib 中导入 scoring 模块
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # 从 NLTK 导入计算 BLEU 分数的函数和流平滑方法
from nltk.tokenize import word_tokenize  # 从 NLTK 导入分词函数
import nltk  # 导入 NLTK 库
nltk.download('punkt')  # 下载 NLTK 分词器的数据包

class Scoring:  # 定义一个名为 Scoring 的类
    """
    calculate_score 
    """  # 类的说明，表示该类的功能是计算得分
    def __init__(self, result_path):  # 初始化方法，接受结果路径作为参数
        """
        init
        """  # 方法说明，表示初始化操作
        self.edit_distances = []  # 初始化编辑距离列表
        self.bleu_scores = []  # 初始化 BLEU 分数列表
        self.sim_scores = []  # 初始化相似度分数列表
        self.filenames = []  # 初始化文件名列表
        self.score_dict = {}  # 初始化得分字典
        self.anntion_cnt = 0  # 初始化注意力计数器
        self.fw = open(result_path, "w+", encoding='utf-8')  # 打开结果文件以写入模式，设置编码为 UTF-8

    def simple_bleu_score(self, candidate, reference):  # 定义计算简单 BLEU 分数的方法，接受候选句子和参考句子作为参数
        """
        get bleu score
        """  # 方法说明，表示获取 BLEU 分数
        candidate_tokens = word_tokenize(candidate)  # 对候选句子进行分词
        reference_tokens = word_tokenize(reference)  # 对参考句子进行分词
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=SmoothingFunction().method1)  # 计算并返回 BLEU 分数，使用流平滑方法

    def preprocess_string(self, s):  # 定义预处理字符串的方法，接受字符串 s 作为参数
        """
        preprocess_string
        """  # 方法说明，表示对字符串进行预处理
        sub_enter = re.sub(r'\n+', '\n', s)  # 替换多个换行符为单个换行符
        return re.sub(r'  ', ' ', sub_enter)  # 替换多个空格为单个空格，并返回结果
    # 计算相似度的方法，接收注释文件夹、实际文件夹和工具类型作为参数
    def calculate_similarity(self, annotion, actual, tool_type):
        """
        calculate_similarity
        """
        # 用于存储每个文件的相似度结果
        class_dict = {}
        # 存储编辑距离的列表
        edit_distances = []
        # 存储 BLEU 分数的列表
        bleu_scores = []
        # 存储相似度分数的列表
        sim_scores = list()
        # 记录总文件数
        total_file = 0
        
        # 遍历注释文件夹中的每个文件
        for filename in os.listdir(annotion):
            # 只处理以 .md 结尾且不以 . 开头的文件
            if filename.endswith('.md') and not filename.startswith('.'):
                # 总文件数加 1
                total_file = total_file + 1
                # 打开并读取注释文件内容
                with open(os.path.join(annotion, filename), 'r', encoding='utf-8') as file_a:
                    content_a = file_a.read()
                # 注释计数加 1
                self.anntion_cnt = self.anntion_cnt + 1
                # 生成实际文件的完整路径
                filepath_b = os.path.join(actual, filename)
                
                # 检查实际文件是否存在
                if os.path.exists(filepath_b):
                    # 打开并读取实际文件内容
                    with open(filepath_b, 'r', encoding='utf-8') as file_b:
                        content_b = file_b.read()
                        # 将文件名加入到文件名列表中
                        self.filenames.append(filename)
                        # 计算编辑距离并归一化
                        edit_dist = distance(self.preprocess_string(content_b),self.preprocess_string(content_a)) / max(len(content_a), len(content_b))
                        # 将编辑距离添加到相应列表
                        self.edit_distances.append(edit_dist)
                        edit_distances.append(edit_dist)
                        # 计算 BLEU 分数
                        bleu_score = self.simple_bleu_score(content_b, content_a)
                        # 将 BLEU 分数添加到相应列表
                        bleu_scores.append(bleu_score)
                        self.bleu_scores.append(bleu_score)
                        # 计算相似度分数
                        score = scoring.score_text(content_b, content_a)
                        # 将相似度分数添加到相应列表
                        sim_scores.append(score)
                        self.sim_scores.append(score)
                        # 将结果存入类字典
                        class_dict[filename] = {"edit_dist": edit_dist, "bleu_score": bleu_score, "sim_score": score}
                        # 将结果存入评分字典
                        self.score_dict[filename] = {"edit_dist": edit_dist, "bleu_score": bleu_score, "sim_score": score}
                else:  
                    # 打印文件未找到的信息
                    print(f"File {filename} not found in actual directory.")
        
        # 计算类的平均编辑距离
        class_average_edit_distance = sum(edit_distances) / len(edit_distances) if edit_distances else 0
        # 计算类的平均 BLEU 分数
        class_average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        # 计算类的平均相似度分数
        class_average_sim_score = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        
        # 将类字典转换为 JSON 并写入文件
        self.fw.write(json.dumps(class_dict, ensure_ascii=False) + "\n")
        # 计算提取比率
        ratio = len(class_dict)/total_file
        # 将提取比率写入文件
        self.fw.write(f"{tool_type} extract ratio:  {ratio}" + "\n")
        # 将平均编辑距离写入文件
        self.fw.write(f"{tool_type} Average Levenshtein Distance: {class_average_edit_distance}" + "\n")
        # 将平均 BLEU 分数写入文件
        self.fw.write(f"{tool_type} Average BLEU Score: {class_average_bleu_score}" + "\n")
        # 将平均相似度分数写入文件
        self.fw.write(f"{tool_type} Average Sim Score: {class_average_sim_score}" + "\n")
        # 打印提取比率
        print (f"{tool_type} extract ratio: {ratio}")
        # 打印平均编辑距离
        print (f"{tool_type} Average Levenshtein Distance: {class_average_edit_distance}")
        # 打印平均 BLEU 分数
        print (f"{tool_type} Average BLEU Score: {class_average_bleu_score}")
        # 打印平均相似度分数
        print (f"{tool_type} Average Sim Score: {class_average_sim_score}")
        # 返回评分字典
        return self.score_dict
    # 定义一个计算总结得分的方法
    def summary_scores(self):
        """
        计算编辑距离、BLEU 分数和相似度分数的平均值
        """
        # 创建一个空字典用于存储平均值
        over_all_dict = dict()
        # 计算编辑距离的平均值，如果没有数据则返回 0
        average_edit_distance = sum(self.edit_distances) / len(self.edit_distances) if self.edit_distances else 0  
        # 计算 BLEU 分数的平均值，如果没有数据则返回 0
        average_bleu_score = sum(self.bleu_scores) / len(self.bleu_scores) if self.bleu_scores else 0  
        # 计算相似度分数的平均值，如果没有数据则返回 0
        average_sim_score = sum(self.sim_scores) / len(self.sim_scores) if self.sim_scores else 0
        # 将编辑距离的平均值存入字典
        over_all_dict["average_edit_distance"] = average_edit_distance
        # 将 BLEU 分数的平均值存入字典
        over_all_dict["average_bleu_score"] = average_bleu_score
        # 将相似度分数的平均值存入字典
        over_all_dict["average_sim_score"] = average_sim_score
        # 将结果字典转换为 JSON 格式并写入文件，同时确保字符不被 ASCII 编码
        self.fw.write(json.dumps(over_all_dict, ensure_ascii=False) + "\n")
        # 返回包含平均值的字典
        return over_all_dict

    # 定义一个计算相似度总分的方法
    def calculate_similarity_total(self, tool_type, download_dir):
        """
        计算编辑距离、BLEU 分数和相似度分数的平均值
        """
        # 构建注释文件的路径
        annotion = os.path.join(download_dir, "annotations", "cleaned")
        # 构建实际文件的路径
        actual = os.path.join(download_dir, tool_type, "cleaned")
        # 计算相似度得分
        score = self.calculate_similarity(annotion, actual, tool_type)
        # 返回计算得到的得分
        return score
```