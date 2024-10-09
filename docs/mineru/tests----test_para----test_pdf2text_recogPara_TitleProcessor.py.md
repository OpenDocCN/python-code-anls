# `.\MinerU\tests\test_para\test_pdf2text_recogPara_TitleProcessor.py`

```
# 导入 JSON 模块，用于处理 JSON 数据
import json
# 导入 unittest 模块，用于单元测试框架
import unittest

# 从 utils_for_test_para 模块导入 UtilsForTestPara 类
from utils_for_test_para import UtilsForTestPara
# 从 magic_pdf.post_proc.detect_para 模块导入 TitleProcessor 类
from magic_pdf.post_proc.detect_para import TitleProcessor

# from ... pdf2text_recogPara import * # 另一种导入方式，已注释

"""
执行以下命令在 code-clean 目录下运行测试：

    python -m tests.test_para.test_pdf2text_recogPara_ClassName
    
    或者 
    
    pytest -v -s app/pdf_toolbox/tests/test_para/test_pdf2text_recogPara_TitleProcessor.py
    
"""

# 定义 TestTitleProcessor 类，继承自 unittest.TestCase
class TestTitleProcessor(unittest.TestCase):
    # 在每个测试用例执行前设置测试环境
    def setUp(self):
        # 初始化 TitleProcessor 实例
        self.title_processor = TitleProcessor()
        # 初始化 UtilsForTestPara 实例
        self.utils = UtilsForTestPara()
        # 读取预处理输出 JSON 文件列表
        self.preproc_out_jsons = self.utils.read_preproc_out_jfiles()

    # 测试 detect_titles 函数，使用预处理的输出 JSON
    def test_batch_process_blocks_detect_titles(self):
        """
        测试 detect_titles 函数，使用预处理的输出 JSON
        """
        # 遍历每个预处理输出 JSON 文件
        for preproc_out_json in self.preproc_out_jsons:
            # 以只读模式打开 JSON 文件，指定编码为 UTF-8
            with open(preproc_out_json, "r", encoding="utf-8") as f:
                # 加载 JSON 文件内容到字典
                preproc_dict = json.load(f)
                # 初始化统计信息为空字典
                preproc_dict["statistics"] = {}
                # 调用 TitleProcessor 的 batch_detect_titles 方法
                result = self.title_processor.batch_detect_titles(preproc_dict)
                # 遍历预处理字典中的每个页面 ID 和块
                for page_id, blocks in preproc_dict.items():
                    # 如果页面 ID 以 "page_" 开头，继续处理
                    if page_id.startswith("page_"):
                        pass
                    else:
                        # 否则跳过
                        continue

    # 测试 batch_process_blocks_recog_title_level 函数，使用预处理的输出 JSON
    def test_batch_process_blocks_recog_title_level(self):
        """
        测试 batch_process_blocks_recog_title_level 函数，使用预处理的输出 JSON
        """
        # 遍历每个预处理输出 JSON 文件
        for preproc_out_json in self.preproc_out_jsons:
            # 以只读模式打开 JSON 文件，指定编码为 UTF-8
            with open(preproc_out_json, "r", encoding="utf-8") as f:
                # 加载 JSON 文件内容到字典
                preproc_dict = json.load(f)
                # 初始化统计信息为空字典
                preproc_dict["statistics"] = {}
                # 调用 TitleProcessor 的 batch_recog_title_level 方法
                result = self.title_processor.batch_recog_title_level(preproc_dict)
                # 遍历预处理字典中的每个页面 ID 和块
                for page_id, blocks in preproc_dict.items():
                    # 如果页面 ID 以 "page_" 开头，继续处理
                    if page_id.startswith("page_"):
                        pass
                    else:
                        # 否则跳过
                        continue

# 如果该脚本作为主程序运行，启动 unittest 测试
if __name__ == "__main__":
    unittest.main()
```