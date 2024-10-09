# `.\MinerU\magic_pdf\rw\DiskReaderWriter.py`

```
# 导入操作系统相关模块
import os
# 从 magic_pdf.rw 导入抽象类 AbsReaderWriter
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
# 导入日志记录库 loguru 的 logger
from loguru import logger

# 定义 DiskReaderWriter 类，继承自 AbsReaderWriter
class DiskReaderWriter(AbsReaderWriter):
    # 初始化方法，接受父路径和编码格式
    def __init__(self, parent_path, encoding="utf-8"):
        # 保存父路径
        self.path = parent_path
        # 保存编码格式
        self.encoding = encoding

    # 读取文件方法，接受路径和模式
    def read(self, path, mode=AbsReaderWriter.MODE_TXT):
        # 检查路径是否为绝对路径
        if os.path.isabs(path):
            abspath = path  # 如果是绝对路径，直接赋值
        else:
            # 如果是相对路径，连接父路径与文件路径
            abspath = os.path.join(self.path, path)
        # 检查文件是否存在
        if not os.path.exists(abspath):
            # 记录错误日志，文件不存在
            logger.error(f"file {abspath} not exists")
            # 抛出异常
            raise Exception(f"file {abspath} no exists")
        # 根据模式读取文件
        if mode == AbsReaderWriter.MODE_TXT:
            # 以文本模式打开文件，读取内容并返回
            with open(abspath, "r", encoding=self.encoding) as f:
                return f.read()
        elif mode == AbsReaderWriter.MODE_BIN:
            # 以二进制模式打开文件，读取内容并返回
            with open(abspath, "rb") as f:
                return f.read()
        else:
            # 如果模式不正确，抛出值错误异常
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")

    # 写入文件方法，接受内容、路径和模式
    def write(self, content, path, mode=AbsReaderWriter.MODE_TXT):
        # 检查路径是否为绝对路径
        if os.path.isabs(path):
            abspath = path  # 如果是绝对路径，直接赋值
        else:
            # 如果是相对路径，连接父路径与文件路径
            abspath = os.path.join(self.path, path)
        # 获取文件的目录路径
        directory_path = os.path.dirname(abspath)
        # 检查目录是否存在
        if not os.path.exists(directory_path):
            # 如果目录不存在，创建目录
            os.makedirs(directory_path)
        # 根据模式写入文件
        if mode == AbsReaderWriter.MODE_TXT:
            # 以文本模式打开文件，写入内容
            with open(abspath, "w", encoding=self.encoding, errors="replace") as f:
                f.write(content)

        elif mode == AbsReaderWriter.MODE_BIN:
            # 以二进制模式打开文件，写入内容
            with open(abspath, "wb") as f:
                f.write(content)
        else:
            # 如果模式不正确，抛出值错误异常
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")

    # 根据偏移量读取文件的方法，接受路径、偏移量和限制字节数
    def read_offset(self, path: str, offset=0, limit=None):
        abspath = path  # 直接赋值路径
        # 检查路径是否为绝对路径
        if not os.path.isabs(path):
            # 如果是相对路径，连接父路径与文件路径
            abspath = os.path.join(self.path, path)
        # 以二进制模式打开文件
        with open(abspath, "rb") as f:
            f.seek(offset)  # 移动文件指针到指定偏移量
            return f.read(limit)  # 读取指定长度的内容并返回

# 检查是否是主程序运行
if __name__ == "__main__":
    # 0: 测试代码块
    if 0:
        # 定义文件路径
        file_path = "io/test/example.txt"
        # 创建 DiskReaderWriter 实例
        drw = DiskReaderWriter("D:\projects\papayfork\Magic-PDF\magic_pdf")

        # 写入内容到文件
        drw.write(b"Hello, World!", path="io/test/example.txt", mode="binary")

        # 从文件读取内容
        content = drw.read(path=file_path)
        # 如果读取到内容，记录日志
        if content:
            logger.info(f"从 {file_path} 读取的内容: {content}")
    # 1: 另一个测试代码块
    if 1:
        # 创建 DiskReaderWriter 实例
        drw = DiskReaderWriter("/opt/data/pdf/resources/test/io/")
        # 从文件中读取内容并赋值
        content_bin = drw.read_offset("1.txt")
        # 确保读取的内容正确
        assert content_bin == b"ABCD!"

        # 从文件中读取内容，指定偏移量和限制
        content_bin = drw.read_offset("1.txt", offset=1, limit=2)
        # 确保读取的内容正确
        assert content_bin == b"BC"
```