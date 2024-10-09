# `.\MinerU\tests\test_para\test_pdf2text_recogPara_Common.py`

```
# 导入 unittest 测试框架
import unittest

# 从 magic_pdf.post_proc.detect_para 模块中导入多个函数
from magic_pdf.post_proc.detect_para import (
    is_bbox_overlap,  # 导入检测边界框重叠的函数
    is_in_bbox,       # 导入检测一个边界框是否在另一个边界框内的函数
    is_line_right_aligned_from_neighbors,  # 导入检测当前行是否与邻近行右对齐的函数
    is_line_left_aligned_from_neighbors,   # 导入检测当前行是否与邻近行左对齐的函数
)

# from ... pdf2text_recogPara import * # 另一种导入方式

"""
执行以下命令以在代码清理目录下运行测试：

    python -m tests.test_para.test_pdf2text_recogPara_Common
    
    或 
    
    pytest -v -s app/pdf_toolbox/tests/test_para/test_pdf2text_recogPara_Common.py
    
"""

# 创建一个测试类，用于测试边界框重叠的功能
class TestIsBboxOverlap(unittest.TestCase):
    # 测试边界框重叠的情况
    def test_overlap(self):
        bbox1 = [0, 0, 10, 10]  # 定义第一个边界框
        bbox2 = [5, 5, 15, 15]  # 定义第二个边界框
        result = is_bbox_overlap(bbox1, bbox2)  # 检查两个边界框是否重叠
        self.assertTrue(result)  # 断言结果为真

    # 测试边界框不重叠的情况
    def test_no_overlap(self):
        bbox1 = [0, 0, 10, 10]  # 定义第一个边界框
        bbox2 = [11, 11, 15, 15]  # 定义第二个边界框
        result = is_bbox_overlap(bbox1, bbox2)  # 检查两个边界框是否重叠
        self.assertFalse(result)  # 断言结果为假

    # 测试边界框部分重叠的情况
    def test_partial_overlap(self):
        bbox1 = [0, 0, 10, 10]  # 定义第一个边界框
        bbox2 = [5, 5, 15, 15]  # 定义第二个边界框
        result = is_bbox_overlap(bbox1, bbox2)  # 检查两个边界框是否重叠
        self.assertTrue(result)  # 断言结果为真

    # 测试两个边界框完全相同的情况
    def test_same_bbox(self):
        bbox1 = [0, 0, 10, 10]  # 定义第一个边界框
        bbox2 = [0, 0, 10, 10]  # 定义第二个边界框
        result = is_bbox_overlap(bbox1, bbox2)  # 检查两个边界框是否重叠
        self.assertTrue(result)  # 断言结果为真


# 创建一个测试类，用于测试边界框包含关系的功能
class TestIsInBbox(unittest.TestCase):
    # 测试边界框1在边界框2内的情况
    def test_bbox1_in_bbox2(self):
        bbox1 = [0, 0, 10, 10]  # 定义边界框1
        bbox2 = [0, 0, 20, 20]  # 定义边界框2
        result = is_in_bbox(bbox1, bbox2)  # 检查边界框1是否在边界框2内
        self.assertTrue(result)  # 断言结果为真

    # 测试边界框1不在边界框2内的情况
    def test_bbox1_not_in_bbox2(self):
        bbox1 = [0, 0, 30, 30]  # 定义边界框1
        bbox2 = [0, 0, 20, 20]  # 定义边界框2
        result = is_in_bbox(bbox1, bbox2)  # 检查边界框1是否在边界框2内
        self.assertFalse(result)  # 断言结果为假

    # 测试两个边界框完全相同的情况
    def test_bbox1_equal_to_bbox2(self):
        bbox1 = [0, 0, 20, 20]  # 定义边界框1
        bbox2 = [0, 0, 20, 20]  # 定义边界框2
        result = is_in_bbox(bbox1, bbox2)  # 检查边界框1是否在边界框2内
        self.assertTrue(result)  # 断言结果为真

    # 测试边界框1部分在边界框2内的情况
    def test_bbox1_partially_in_bbox2(self):
        bbox1 = [10, 10, 30, 30]  # 定义边界框1
        bbox2 = [0, 0, 20, 20]  # 定义边界框2
        result = is_in_bbox(bbox1, bbox2)  # 检查边界框1是否在边界框2内
        self.assertFalse(result)  # 断言结果为假


# 创建一个测试类，用于测试行右对齐的功能
class TestIsLineRightAlignedFromNeighbors(unittest.TestCase):
    # 测试当前行与上一行右对齐的情况
    def test_right_aligned_with_prev_line(self):
        curr_line_bbox = [0, 0, 100, 100]  # 定义当前行的边界框
        prev_line_bbox = [0, 0, 90, 100]  # 定义上一行的边界框
        next_line_bbox = None  # 定义下一行的边界框（无）
        avg_char_width = 10  # 定义平均字符宽度
        direction = 0  # 定义方向（0表示未定义方向）
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)  # 检查对齐情况
        self.assertFalse(result)  # 断言结果为假

    # 测试当前行与下一行右对齐的情况
    def test_right_aligned_with_next_line(self):
        curr_line_bbox = [0, 0, 100, 100]  # 定义当前行的边界框
        prev_line_bbox = None  # 定义上一行的边界框（无）
        next_line_bbox = [0, 0, 110, 100]  # 定义下一行的边界框
        avg_char_width = 10  # 定义平均字符宽度
        direction = 1  # 定义方向（1表示定义的方向）
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)  # 检查对齐情况
        self.assertFalse(result)  # 断言结果为假
    # 测试当前行与前后行均右对齐的情况
    def test_right_aligned_with_both_lines(self):
        # 当前行的边界框，表示左上角和右下角坐标
        curr_line_bbox = [0, 0, 100, 100]
        # 前一行的边界框
        prev_line_bbox = [0, 0, 90, 100]
        # 下一行的边界框
        next_line_bbox = [0, 0, 110, 100]
        # 平均字符宽度
        avg_char_width = 10
        # 方向参数，表示检测的方向
        direction = 2
        # 调用函数检测当前行是否右对齐
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果应为 False
        self.assertFalse(result)

    # 测试当前行与前一行不右对齐的情况
    def test_not_right_aligned_with_prev_line(self):
        # 当前行的边界框
        curr_line_bbox = [0, 0, 100, 100]
        # 前一行的边界框
        prev_line_bbox = [0, 0, 80, 100]
        # 下一行边界框为空，表示不存在
        next_line_bbox = None
        # 平均字符宽度
        avg_char_width = 10
        # 方向参数
        direction = 0
        # 调用函数检测当前行是否右对齐
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果应为 False
        self.assertFalse(result)

    # 测试当前行与下一行不右对齐的情况
    def test_not_right_aligned_with_next_line(self):
        # 当前行的边界框
        curr_line_bbox = [0, 0, 100, 100]
        # 前一行边界框为空
        prev_line_bbox = None
        # 下一行的边界框
        next_line_bbox = [0, 0, 120, 100]
        # 平均字符宽度
        avg_char_width = 10
        # 方向参数
        direction = 1
        # 调用函数检测当前行是否右对齐
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果应为 False
        self.assertFalse(result)

    # 测试当前行与前后行均不右对齐的情况
    def test_not_right_aligned_with_both_lines(self):
        # 当前行的边界框
        curr_line_bbox = [0, 0, 100, 100]
        # 前一行的边界框
        prev_line_bbox = [0, 0, 80, 100]
        # 下一行的边界框
        next_line_bbox = [0, 0, 120, 100]
        # 平均字符宽度
        avg_char_width = 10
        # 方向参数
        direction = 2
        # 调用函数检测当前行是否右对齐
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果应为 False
        self.assertFalse(result)

    # 测试无效的方向参数
    def test_invalid_direction(self):
        # 当前行的边界框
        curr_line_bbox = [0, 0, 100, 100]
        # 前一行和下一行均为空
        prev_line_bbox = None
        next_line_bbox = None
        # 平均字符宽度
        avg_char_width = 10
        # 无效的方向参数
        direction = 3
        # 调用函数检测当前行是否右对齐
        result = is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果应为 False
        self.assertFalse(result)
# 测试 is_line_left_aligned_from_neighbors 函数
class TestIsLineLeftAlignedFromNeighbors(unittest.TestCase):

    # 测试当前行是否与前一行左对齐
    def test_left_aligned_with_prev_line(self):
        # 定义当前行的边界框
        curr_line_bbox = [10, 20, 30, 40]
        # 定义前一行的边界框
        prev_line_bbox = [5, 20, 30, 40]
        # 定义下一行的边界框为 None
        next_line_bbox = None
        # 定义平均字符宽度
        avg_char_width = 5.0
        # 定义方向
        direction = 0
        # 调用函数检查左对齐情况，并保存结果
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果为 False
        self.assertFalse(result)

    # 测试当前行是否与下一行左对齐
    def test_left_aligned_with_next_line(self):
        # 定义当前行的边界框
        curr_line_bbox = [10, 20, 30, 40]
        # 定义前一行的边界框为 None
        prev_line_bbox = None
        # 定义下一行的边界框
        next_line_bbox = [15, 20, 30, 40]
        # 定义平均字符宽度
        avg_char_width = 5.0
        # 定义方向
        direction = 1
        # 调用函数检查左对齐情况，并保存结果
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果为 False
        self.assertFalse(result)

    # 测试当前行是否与前一行和下一行都左对齐
    def test_left_aligned_with_both_lines(self):
        # 定义当前行的边界框
        curr_line_bbox = [10, 20, 30, 40]
        # 定义前一行的边界框
        prev_line_bbox = [5, 20, 30, 40]
        # 定义下一行的边界框
        next_line_bbox = [15, 20, 30, 40]
        # 定义平均字符宽度
        avg_char_width = 5.0
        # 定义方向
        direction = 2
        # 调用函数检查左对齐情况，并保存结果
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果为 False
        self.assertFalse(result)

    # 测试当前行不与前一行左对齐
    def test_not_left_aligned_with_prev_line(self):
        # 定义当前行的边界框
        curr_line_bbox = [10, 20, 30, 40]
        # 定义前一行的边界框
        prev_line_bbox = [5, 20, 30, 40]
        # 定义下一行的边界框为 None
        next_line_bbox = None
        # 定义平均字符宽度
        avg_char_width = 5.0
        # 定义方向
        direction = 0
        # 调用函数检查左对齐情况，并保存结果
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果为 False
        self.assertFalse(result)

    # 测试当前行不与下一行左对齐
    def test_not_left_aligned_with_next_line(self):
        # 定义当前行的边界框
        curr_line_bbox = [10, 20, 30, 40]
        # 定义前一行的边界框为 None
        prev_line_bbox = None
        # 定义下一行的边界框
        next_line_bbox = [15, 20, 30, 40]
        # 定义平均字符宽度
        avg_char_width = 5.0
        # 定义方向
        direction = 1
        # 调用函数检查左对齐情况，并保存结果
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果为 False
        self.assertFalse(result)

    # 测试当前行不与前一行和下一行都左对齐
    def test_not_left_aligned_with_both_lines(self):
        # 定义当前行的边界框
        curr_line_bbox = [10, 20, 30, 40]
        # 定义前一行的边界框
        prev_line_bbox = [5, 20, 30, 40]
        # 定义下一行的边界框
        next_line_bbox = [15, 20, 30, 40]
        # 定义平均字符宽度
        avg_char_width = 5.0
        # 定义方向
        direction = 2
        # 调用函数检查左对齐情况，并保存结果
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果为 False
        self.assertFalse(result)

    # 测试无效的方向值
    def test_invalid_direction(self):
        # 定义当前行的边界框
        curr_line_bbox = [10, 20, 30, 40]
        # 定义前一行的边界框为 None
        prev_line_bbox = None
        # 定义下一行的边界框为 None
        next_line_bbox = None
        # 定义平均字符宽度
        avg_char_width = 5.0
        # 定义无效的方向
        direction = 3
        # 调用函数检查左对齐情况，并保存结果
        result = is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction)
        # 断言结果为 False
        self.assertFalse(result)


# 如果当前脚本作为主程序运行，执行单元测试
if __name__ == "__main__":
    unittest.main()
```