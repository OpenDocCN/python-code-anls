# `.\MinerU\magic_pdf\libs\json_compressor.py`

```
# 导入 JSON 处理模块
import json
# 导入 Brotli 压缩模块
import brotli
# 导入 Base64 编码模块
import base64

# 定义一个处理 JSON 压缩和解压缩的类
class JsonCompressor:

    # 定义一个静态方法用于压缩 JSON 数据
    @staticmethod
    def compress_json(data):
        """
        压缩 JSON 对象并使用 Base64 编码
        """
        # 将传入的数据转换为 JSON 字符串
        json_str = json.dumps(data)
        # 将 JSON 字符串编码为字节
        json_bytes = json_str.encode('utf-8')
        # 使用 Brotli 压缩字节数据，质量等级设为 6
        compressed = brotli.compress(json_bytes, quality=6)
        # 将压缩后的字节数据进行 Base64 编码并解码为字符串
        compressed_str = base64.b64encode(compressed).decode('utf-8')  # convert bytes to string
        # 返回压缩后的 Base64 字符串
        return compressed_str

    # 定义一个静态方法用于解压缩 JSON 数据
    @staticmethod
    def decompress_json(compressed_str):
        """
        解码 Base64 字符串并解压缩 JSON 对象
        """
        # 将 Base64 字符串解码为字节
        compressed = base64.b64decode(compressed_str.encode('utf-8'))  # convert string to bytes
        # 使用 Brotli 解压缩字节数据
        decompressed_bytes = brotli.decompress(compressed)
        # 将解压缩后的字节数据解码为 JSON 字符串
        json_str = decompressed_bytes.decode('utf-8')
        # 将 JSON 字符串解析为 Python 对象
        data = json.loads(json_str)
        # 返回解析后的数据
        return data
```