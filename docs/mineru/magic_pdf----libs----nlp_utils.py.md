# `.\MinerU\magic_pdf\libs\nlp_utils.py`

```
# 导入正则表达式模块
import re
# 从操作系统路径模块导入 path
from os import path

# 从集合模块导入 Counter，用于计数
from collections import Counter

# 导入日志记录器
from loguru import logger

# 从语言检测模块导入 detect（已注释掉）
# from langdetect import detect
# 导入 spaCy 库
import spacy
# 导入英文模型
import en_core_web_sm
# 导入中文模型
import zh_core_web_sm

# 从自定义库导入语言检测功能
from magic_pdf.libs.language import detect_lang

# 定义 NLPModels 类
class NLPModels:
    """
    如何将本地模型上传到 S3:
        - 配置 AWS CLI:
            doc\SETUP-CLI.md
            doc\setup_cli.sh
            app\config\__init__.py
        - $ cd {local_dir_storing_models}
        - $ ls models
            en_core_web_sm-3.7.1/
            zh_core_web_sm-3.7.0/
        - $ aws s3 sync models/ s3://llm-infra/models --profile=p_project_norm
        - $ aws s3 --profile=p_project_norm ls  s3://llm-infra/models/
            PRE en_core_web_sm-3.7.1/
            PRE zh_core_web_sm-3.7.0/
    """

    # 初始化方法
    def __init__(self):
        # 如果操作系统是 Windows，将 "TMP_DIR" 设置为 "D:/tmp"

        # 获取当前用户的主目录
        home_dir = path.expanduser("~")
        # 设置默认本地路径为用户主目录下的 .nlp_models 文件夹
        self.default_local_path = path.join(home_dir, ".nlp_models")
        # 设置默认共享路径
        self.default_shared_path = "/share/pdf_processor/nlp_models"
        # 设置默认 HDFS 路径
        self.default_hdfs_path = "hdfs://pdf_processor/nlp_models"
        # 设置默认 S3 路径
        self.default_s3_path = "s3://llm-infra/models"
        # 定义 NLP 模型字典，包括模型名称、类型和版本
        self.nlp_models = {
            "en_core_web_sm": {
                "type": "spacy",
                "version": "3.7.1",
            },
            "en_core_web_md": {
                "type": "spacy",
                "version": "3.7.1",
            },
            "en_core_web_lg": {
                "type": "spacy",
                "version": "3.7.1",
            },
            "zh_core_web_sm": {
                "type": "spacy",
                "version": "3.7.0",
            },
            "zh_core_web_md": {
                "type": "spacy",
                "version": "3.7.0",
            },
            "zh_core_web_lg": {
                "type": "spacy",
                "version": "3.7.0",
            },
        }
        # 加载英文小型模型
        self.en_core_web_sm_model = en_core_web_sm.load()
        # 加载中文小型模型
        self.zh_core_web_sm_model = zh_core_web_sm.load()

    # 定义加载模型的方法
    def load_model(self, model_name, model_type, model_version):
        # 检查模型名称、类型和版本是否支持
        if (
            model_name in self.nlp_models
            and self.nlp_models[model_name]["type"] == model_type
            and self.nlp_models[model_name]["version"] == model_version
        ):
            # 如果是有效模型，加载并返回模型
            return spacy.load(model_name) if spacy.util.is_package(model_name) else None

        else:
            # 如果模型名称或版本不支持，记录错误日志
            logger.error(f"Unsupported model name or version: {model_name} {model_version}")
            # 返回 None
            return None
    # 定义一个检测语言的方法，接受文本和一个可选的参数来使用语言检测库
        def detect_language(self, text, use_langdetect=False):
            # 检查文本长度是否为零
            if len(text) == 0:
                # 如果文本为空，返回 None
                return None
            # 判断是否使用语言检测
            if use_langdetect:
                # print("use_langdetect")  # 打印调试信息
                # print(detect_lang(text))  # 打印检测结果
                # return detect_lang(text)  # 返回检测的语言
                # 如果检测结果为中文，返回 "zh"
                if detect_lang(text) == "zh":
                    return "zh"
                # 否则返回英文
                else:
                    return "en"
    
            # 如果不使用语言检测
            if not use_langdetect:
                # 计算文本中英文字符的数量
                en_count = len(re.findall(r"[a-zA-Z]", text))
                # 计算文本中中文字符的数量
                cn_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    
                # 如果英文字符数量大于中文字符数量，返回 "en"
                if en_count > cn_count:
                    return "en"
    
                # 如果中文字符数量大于英文字符数量，返回 "zh"
                if cn_count > en_count:
                    return "zh"
    
    # 定义一个使用 NLP 检测实体类别的方法，接受文本和阈值参数
        def detect_entity_catgr_using_nlp(self, text, threshold=0.5):
            """
            检测实体类别，使用 NLP 模型并返回最频繁的实体类型。
    
            Parameters
            ----------
            text : str
                要处理的文本。
    
            Returns
            -------
            str
                最频繁的实体类型。
            """
            # 检测文本语言
            lang = self.detect_language(text, use_langdetect=True)
    
            # 如果语言为英文，选择英文 NLP 模型
            if lang == "en":
                nlp_model = self.en_core_web_sm_model
            # 如果语言为中文，选择中文 NLP 模型
            elif lang == "zh":
                nlp_model = self.zh_core_web_sm_model
            # 如果语言不支持，返回空字典
            else:
                # logger.error(f"Unsupported language: {lang}")  # 记录错误日志
                return {}
    
            # 将文本拆分为较小的部分
            text_parts = re.split(r"[,;，；、\s & |]+", text)
    
            # 移除非单词部分
            text_parts = [part for part in text_parts if not re.match(r"[\d\W]+", part)]  # 去除非单词
            # 将文本部分组合成一个字符串
            text_combined = " ".join(text_parts)
    
            try:
                # 处理组合后的文本
                doc = nlp_model(text_combined)
                # 计算每种实体的出现次数
                entity_counts = Counter([ent.label_ for ent in doc.ents])
                word_counts_in_entities = Counter()
    
                # 遍历每个实体，统计其单词数
                for ent in doc.ents:
                    word_counts_in_entities[ent.label_] += len(ent.text.split())
    
                # 计算实体中总单词数
                total_words_in_entities = sum(word_counts_in_entities.values())
                # 计算总单词数（排除标点符号）
                total_words = len([token for token in doc if not token.is_punct])
    
                # 如果没有实体或总单词数为零，返回 None
                if total_words_in_entities == 0 or total_words == 0:
                    return None
    
                # 计算实体所占百分比
                entity_percentage = total_words_in_entities / total_words
                # 如果实体百分比小于 0.5，返回 None
                if entity_percentage < 0.5:
                    return None
    
                # 找到出现最频繁的实体及其单词数
                most_common_entity, word_count = word_counts_in_entities.most_common(1)[0]
                # 计算实体占总实体的百分比
                entity_percentage = word_count / total_words_in_entities
    
                # 如果实体占比大于等于阈值，返回最常见的实体
                if entity_percentage >= threshold:
                    return most_common_entity
                else:
                    return None
            except Exception as e:
                # 记录实体检测中的错误
                logger.error(f"Error in entity detection: {e}")
                return None
# 定义主函数
def __main__():
    # 创建 NLPModels 类的实例
    nlpModel = NLPModels()

    # 测试字符串列表，包括多种姓名和格式
    test_strings = [
        "张三",  # 单个中文姓名
        "张三, 李四，王五; 赵六",  # 多个中文姓名，包含不同分隔符
        "John Doe",  # 单个英文姓名
        "Jane Smith",  # 单个英文姓名
        "Lee, John",  # 反向姓名格式
        "John Doe, Jane Smith; Alice Johnson，Bob Lee",  # 多个英文姓名，包含不同分隔符
        "孙七, Michael Jordan；赵八",  # 中英文混合姓名
        "David Smith  Michael O'Connor; Kevin ßáçøñ",  # 多个英文姓名，包含特殊字符
        "李雷·韩梅梅, 张三·李四",  # 中文姓名带有特殊字符
        "Charles Robert Darwin, Isaac Newton",  # 著名英文姓名
        "莱昂纳多·迪卡普里奥, 杰克·吉伦哈尔",  # 中英文混合著名姓名
        "John Doe, Jane Smith; Alice Johnson",  # 多个英文姓名，包含不同分隔符
        "张三, 李四，王五; 赵六",  # 重复的中文姓名组合
        "Lei Wang, Jia Li, and Xiaojun Chen, LINKE YANG OU, and YUAN ZHANG",  # 多个英文姓名，包含 "and"
        "Rachel Mills  &  William Barry  &  Susanne B. Haga",  # 多个英文姓名，包含 "&"
        "Claire Chabut* and Jean-François Bussières",  # 带有特殊字符的姓名
        "1 Department of Chemistry, Northeastern University, Shenyang 110004, China 2 State Key Laboratory of Polymer Physics and Chemistry, Changchun Institute of Applied Chemistry, Chinese Academy of Sciences, Changchun 130022, China",  # 复杂的机构地址信息
        "Changchun",  # 单个地名
        "china",  # 单个国家名
        "Rongjun Song, 1,2 Baoyan Zhang, 1 Baotong Huang, 2 Tao Tang 2",  # 多个姓名及其对应的编号
        "Synergistic Effect of Supported Nickel Catalyst with Intumescent Flame-Retardants on Flame Retardancy and Thermal Stability of Polypropylene",  # 复杂的科学研究标题
        "Synergistic Effect of Supported Nickel Catalyst with",  # 部分科学研究标题
        "Intumescent Flame-Retardants on Flame Retardancy",  # 部分科学研究标题
        "and Thermal Stability of Polypropylene",  # 部分科学研究标题
    ]

    # 遍历测试字符串列表
    for test in test_strings:
        print()  # 打印空行以增加可读性
        # 打印原始字符串
        print(f"Original String: {test}")

        # 使用 NLP 模型检测字符串中的实体分类
        result = nlpModel.detect_entity_catgr_using_nlp(test)
        # 打印检测到的实体
        print(f"Detected entities: {result}")

# 如果当前模块是主程序，则执行 __main__ 函数
if __name__ == "__main__":
    __main__()
```