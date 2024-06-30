# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\report.py`

```
"""Progress report printers."""

from __future__ import annotations  # 导入用于支持类型注解的模块

class ReportBase:
    COLUMN_NAMES: list[str] = NotImplemented  # 列名列表，未实现具体内容
    COLUMN_WIDTHS: list[int] = NotImplemented  # 列宽列表，未实现具体内容
    ITERATION_FORMATS: list[str] = NotImplemented  # 迭代格式列表，未实现具体内容

    @classmethod
    def print_header(cls):
        """打印报告的表头信息"""
        fmt = ("|"  # 格式字符串开始
               + "|".join([f"{{:^{x}}}" for x in cls.COLUMN_WIDTHS])  # 根据列宽生成格式化字符串
               + "|")
        separators = ['-' * x for x in cls.COLUMN_WIDTHS]  # 根据列宽生成分隔线字符串列表
        print(fmt.format(*cls.COLUMN_NAMES))  # 使用列名打印表头第一行
        print(fmt.format(*separators))  # 打印表头第二行的分隔线

    @classmethod
    def print_iteration(cls, *args):
        """打印迭代信息"""
        iteration_format = [f"{{:{x}}}" for x in cls.ITERATION_FORMATS]  # 根据迭代格式生成格式化字符串列表
        fmt = "|" + "|".join(iteration_format) + "|"  # 生成完整的格式化字符串
        print(fmt.format(*args))  # 使用传入的参数打印迭代行

    @classmethod
    def print_footer(cls):
        """打印报告的页脚信息"""
        print()  # 打印空行，作为页脚的一部分

class BasicReport(ReportBase):
    COLUMN_NAMES = ["niter", "f evals", "CG iter", "obj func", "tr radius",
                    "opt", "c viol"]  # 基本报告的列名列表
    COLUMN_WIDTHS = [7, 7, 7, 13, 10, 10, 10]  # 基本报告的列宽列表
    ITERATION_FORMATS = ["^7", "^7", "^7", "^+13.4e",
                         "^10.2e", "^10.2e", "^10.2e"]  # 基本报告的迭代格式列表

class SQPReport(ReportBase):
    COLUMN_NAMES = ["niter", "f evals", "CG iter", "obj func", "tr radius",
                    "opt", "c viol", "penalty", "CG stop"]  # SQP 报告的列名列表
    COLUMN_WIDTHS = [7, 7, 7, 13, 10, 10, 10, 10, 7]  # SQP 报告的列宽列表
    ITERATION_FORMATS = ["^7", "^7", "^7", "^+13.4e", "^10.2e", "^10.2e",
                         "^10.2e", "^10.2e", "^7"]  # SQP 报告的迭代格式列表

class IPReport(ReportBase):
    COLUMN_NAMES = ["niter", "f evals", "CG iter", "obj func", "tr radius",
                    "opt", "c viol", "penalty", "barrier param", "CG stop"]  # IP 报告的列名列表
    COLUMN_WIDTHS = [7, 7, 7, 13, 10, 10, 10, 10, 13, 7]  # IP 报告的列宽列表
    ITERATION_FORMATS = ["^7", "^7", "^7", "^+13.4e", "^10.2e", "^10.2e",
                         "^10.2e", "^10.2e", "^13.2e", "^7"]  # IP 报告的迭代格式列表
```