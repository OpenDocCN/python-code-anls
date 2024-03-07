# `.\marker\marker\logger.py`

```py
# 导入 logging 模块
import logging
# 导入 fitz 模块并重命名为 pymupdf
import fitz as pymupdf
# 导入 warnings 模块

# 配置日志记录
def configure_logging():
    # 设置日志级别为 WARNING
    logging.basicConfig(level=logging.WARNING)

    # 设置 pdfminer 模块的日志级别为 ERROR
    logging.getLogger('pdfminer').setLevel(logging.ERROR)
    # 设置 PIL 模块的日志级别为 ERROR
    logging.getLogger('PIL').setLevel(logging.ERROR)
    # 设置 fitz 模块的日志级别为 ERROR
    logging.getLogger('fitz').setLevel(logging.ERROR)
    # 设置 ocrmypdf 模块的日志级别为 ERROR
    logging.getLogger('ocrmypdf').setLevel(logging.ERROR)
    # 设置 fitz 模块的错误显示为 False
    pymupdf.TOOLS.mupdf_display_errors(False)
    # 忽略 FutureWarning 类别的警告
    warnings.simplefilter(action='ignore', category=FutureWarning)
```