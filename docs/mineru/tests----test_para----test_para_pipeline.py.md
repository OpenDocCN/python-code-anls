# `.\MinerU\tests\test_para\test_para_pipeline.py`

```
# 导入 unittest 模块以进行单元测试
import unittest

"""
执行以下命令以在代码清理目录下运行测试：

    python -m tests.test_para.test_para_pipeline
    
    或者
    
    pytest -v -s app/pdf_toolbox/tests/test_para/test_para_pipeline.py
    
"""

# 从测试模块导入多个测试类
from tests.test_para.test_pdf2text_recogPara_Common import (
    TestIsBboxOverlap,  # 导入测试类：测试边界框重叠
    TestIsInBbox,       # 导入测试类：测试是否在边界框内
    TestIsBboxOverlap,  # 再次导入测试类：测试边界框重叠
    TestIsLineLeftAlignedFromNeighbors,  # 导入测试类：测试行是否左对齐
    TestIsLineRightAlignedFromNeighbors,  # 导入测试类：测试行是否右对齐
)
from tests.test_para.test_pdf2text_recogPara_EquationsProcessor import TestCalcOverlapPct  # 导入测试类：计算重叠百分比
from tests.test_para.test_pdf2text_recogPara_BlockInnerParasProcessor import TestIsConsistentLines  # 导入测试类：测试行一致性
from tests.test_para.test_pdf2text_recogPara_BlockContinuationProcessor import (
    TestIsAlphabetChar,  # 导入测试类：测试是否为字母字符
    TestIsChineseChar,   # 导入测试类：测试是否为汉字字符
    TestIsOtherLetterChar,  # 导入测试类：测试是否为其他字母字符
)
from tests.test_para.test_pdf2text_recogPara_TitleProcessor import TestTitleProcessor  # 导入测试类：标题处理器的测试类

# 创建一个测试套件
suite = unittest.TestSuite()

# 将来自 test_pdf2text_recogPara_Common 的测试用例添加到测试套件中
suite.addTest(unittest.makeSuite(TestIsBboxOverlap))  # 添加边界框重叠测试用例
suite.addTest(unittest.makeSuite(TestIsInBbox))  # 添加边界框内测试用例
suite.addTest(unittest.makeSuite(TestIsBboxOverlap))  # 再次添加边界框重叠测试用例
suite.addTest(unittest.makeSuite(TestIsLineLeftAlignedFromNeighbors))  # 添加左对齐测试用例
suite.addTest(unittest.makeSuite(TestIsLineRightAlignedFromNeighbors))  # 添加右对齐测试用例

# 将来自 test_pdf2text_recogPara_EquationsProcessor 的测试用例添加到测试套件中
suite.addTest(unittest.makeSuite(TestCalcOverlapPct))  # 添加重叠百分比测试用例

# 将来自 test_pdf2text_recogPara_BlockInnerParasProcessor 的测试用例添加到测试套件中
suite.addTest(unittest.makeSuite(TestIsConsistentLines))  # 添加行一致性测试用例

# 将来自 test_pdf2text_recogPara_BlockContinuationProcessor 的测试用例添加到测试套件中
suite.addTest(unittest.makeSuite(TestIsAlphabetChar))  # 添加字母字符测试用例
suite.addTest(unittest.makeSuite(TestIsChineseChar))  # 添加汉字字符测试用例
suite.addTest(unittest.makeSuite(TestIsOtherLetterChar))  # 添加其他字母字符测试用例

# 将来自 test_pdf2text_recogPara_TitleProcessor 的测试用例添加到测试套件中
suite.addTest(unittest.makeSuite(TestTitleProcessor))  # 添加标题处理器测试用例

# 运行测试套件
unittest.TextTestRunner(verbosity=2).run(suite)  # 以详细模式运行测试
```