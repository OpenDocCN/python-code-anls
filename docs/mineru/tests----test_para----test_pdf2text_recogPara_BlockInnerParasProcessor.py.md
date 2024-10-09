# `.\MinerU\tests\test_para\test_pdf2text_recogPara_BlockInnerParasProcessor.py`

```
# 导入单元测试模块
import unittest

# 从指定模块导入 BlockTerminationProcessor 类
from magic_pdf.post_proc.detect_para import BlockTerminationProcessor

# 注释：运行测试的命令行示例，指定测试文件位置

class TestIsConsistentLines(unittest.TestCase):
    # 设置测试环境，初始化 BlockTerminationProcessor 实例
    def setUp(self):
        self.obj = BlockTerminationProcessor()

    # 测试当前行与前一行的一致性
    def test_consistent_with_prev_line(self):
        # 定义当前行的属性
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        # 定义前一行的属性
        prev_line = {"spans": [{"size": 12, "font": "Arial"}]}
        next_line = None  # 没有下一行
        consistent_direction = 0  # 一致性方向为0（前一行）
        # 调用一致性检查方法并获取结果
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        # 断言结果为真
        self.assertTrue(result)

    # 测试当前行与下一行的一致性
    def test_consistent_with_next_line(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        prev_line = None  # 没有前一行
        # 定义下一行的属性
        next_line = {"spans": [{"size": 12, "font": "Arial"}]}
        consistent_direction = 1  # 一致性方向为1（下一行）
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertTrue(result)

    # 测试当前行与前一行和下一行的一致性
    def test_consistent_with_both_lines(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        prev_line = {"spans": [{"size": 12, "font": "Arial"}]}
        next_line = {"spans": [{"size": 12, "font": "Arial"}]}
        consistent_direction = 2  # 一致性方向为2（两者）
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertTrue(result)

    # 测试当前行与前一行不一致
    def test_inconsistent_with_prev_line(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        # 定义前一行的不同属性
        prev_line = {"spans": [{"size": 14, "font": "Arial"}]}
        next_line = None  # 没有下一行
        consistent_direction = 0  # 一致性方向为0（前一行）
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertFalse(result)

    # 测试当前行与下一行不一致
    def test_inconsistent_with_next_line(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        prev_line = None  # 没有前一行
        # 定义下一行的不同属性
        next_line = {"spans": [{"size": 14, "font": "Arial"}]}
        consistent_direction = 1  # 一致性方向为1（下一行）
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertFalse(result)

    # 测试当前行与前一行和下一行都不一致
    def test_inconsistent_with_both_lines(self):
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        # 定义前一行和下一行的不同属性
        prev_line = {"spans": [{"size": 14, "font": "Arial"}]}
        next_line = {"spans": [{"size": 14, "font": "Arial"}]}
        consistent_direction = 2  # 一致性方向为2（两者）
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        self.assertFalse(result)
    # 测试无效的一致方向
    def test_invalid_consistent_direction(self):
        # 当前行的样式信息，包括字体大小和字体类型
        curr_line = {"spans": [{"size": 12, "font": "Arial"}]}
        # 前一行设置为 None，表示不存在
        prev_line = None
        # 下一行设置为 None，表示不存在
        next_line = None
        # 一致方向的标志，值为 3
        consistent_direction = 3
        # 调用方法检查当前行与前后行的一致性
        result = self.obj._is_consistent_lines(curr_line, prev_line, next_line, consistent_direction)
        # 断言结果应为 False，表示不一致
        self.assertFalse(result)
    
    # 测试可能的段落起始位置
    def test_possible_start_of_para(self):
        # 当前行的边界框信息
        curr_line = {"bbox": (0, 0, 100, 10)}
        # 前一行的边界框信息
        prev_line = {"bbox": (0, 20, 100, 30)}
        # 下一行的边界框信息
        next_line = {"bbox": (0, 40, 100, 50)}
        # X0 和 X1 定义了段落的水平范围
        X0 = 0
        X1 = 100
        # 平均字符宽度
        avg_char_width = 5
        # 平均字体大小
        avg_font_size = 10
    
        # 调用方法检查当前行是否为段落的可能起始位置
        result, _, _ = self.obj._is_possible_start_of_para(
            curr_line, prev_line, next_line, X0, X1, avg_char_width, avg_font_size
        )
        # 断言结果应为 True，表示可能是段落起始
        self.assertTrue(result)
    
    # 测试不可能的段落起始位置
    def test_not_possible_start_of_para(self):
        # 当前行的边界框信息
        curr_line = {"bbox": (0, 0, 100, 10)}
        # 前一行的边界框信息
        prev_line = {"bbox": (0, 20, 100, 30)}
        # 下一行的边界框信息
        next_line = {"bbox": (0, 40, 100, 50)}
        # X0 和 X1 定义了段落的水平范围
        X0 = 0
        X1 = 100
        # 平均字符宽度
        avg_char_width = 5
        # 平均字体大小
        avg_font_size = 10
    
        # 调用方法检查当前行是否为段落的不可能起始位置
        result, _, _ = self.obj._is_possible_start_of_para(curr_line, prev_line, next_line, X0, X1, avg_char_width, avg_font_size)
        # 断言结果应为 True，表示不可能是段落起始
        self.assertTrue(result)
# 如果该文件是作为主程序运行
if __name__ == "__main__":
    # 执行所有单元测试
    unittest.main()
```