# `ZeroNet\src\Test\TestCryptHash.py`

```
# 导入 base64 模块
import base64

# 从 Crypt 模块中导入 CryptHash 类
from Crypt import CryptHash

# 定义 SHA-512 散列值的十六进制表示
sha512t_sum_hex = "2e9466d8aa1f340c91203b4ddbe9b6669879616a1b8e9571058a74195937598d"
# 定义 SHA-512 散列值的二进制表示
sha512t_sum_bin = b".\x94f\xd8\xaa\x1f4\x0c\x91 ;M\xdb\xe9\xb6f\x98yaj\x1b\x8e\x95q\x05\x8at\x19Y7Y\x8d"
# 定义 SHA-256 散列值的十六进制表示
sha256_sum_hex = "340cd04be7f530e3a7c1bc7b24f225ba5762ec7063a56e1ae01a30d56722e5c3"

# 定义 TestCryptBitcoin 类
class TestCryptBitcoin:

    # 定义测试 SHA 散列函数的方法
    def testSha(self, site):
        # 获取文件路径
        file_path = site.storage.getPath("dbschema.json")
        # 断言文件的 SHA-512 散列值与预定义的十六进制表示相等
        assert CryptHash.sha512sum(file_path) == sha512t_sum_hex
        # 断言打开文件后的 SHA-512 散列值与预定义的十六进制表示相等
        assert CryptHash.sha512sum(open(file_path, "rb")) == sha512t_sum_hex
        # 断言打开文件后的 SHA-512 散列值（以 digest 格式）与预定义的二进制表示相等
        assert CryptHash.sha512sum(open(file_path, "rb"), format="digest") == sha512t_sum_bin

        # 断言文件的 SHA-256 散列值与预定义的十六进制表示相等
        assert CryptHash.sha256sum(file_path) == sha256_sum_hex
        # 断言打开文件后的 SHA-256 散列值与预定义的十六进制表示相等
        assert CryptHash.sha256sum(open(file_path, "rb")) == sha256_sum_hex

        # 使用 with 语句打开文件
        with open(file_path, "rb") as f:
            # 计算文件前 100 字节的 SHA-512t 散列值
            hash = CryptHash.Sha512t(f.read(100))
            # 断言散列值与预定义的十六进制表示不相等
            hash.hexdigest() != sha512t_sum_hex
            # 更新散列值，计算整个文件的 SHA-512t 散列值，并与预定义的十六进制表示相等
            hash.update(f.read(1024 * 1024))
            assert hash.hexdigest() == sha512t_sum_hex

    # 定义测试随机函数的方法
    def testRandom(self):
        # 断言随机生成的字节串长度为 64
        assert len(CryptHash.random(64)) == 64
        # 断言两次随机生成的字节串不相等
        assert CryptHash.random() != CryptHash.random()
        # 断言将十六进制编码的随机字节串解码为字节串
        assert bytes.fromhex(CryptHash.random(encoding="hex"))
        # 断言将 base64 编码的随机字节串解码为字节串
        assert base64.b64decode(CryptHash.random(encoding="base64"))
```