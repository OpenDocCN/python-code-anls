# `ZeroNet\src\Crypt\CryptHash.py`

```
# 导入 hashlib 模块，用于计算哈希值
# 导入 os 模块，用于生成随机字节流
# 导入 base64 模块，用于编码和解码 base64 数据

# 计算文件的 SHA-512 哈希值
def sha512sum(file, blocksize=65536, format="hexdigest"):
    # 如果传入的是文件名，则打开文件
    if type(file) is str:  
        file = open(file, "rb")
    # 创建 SHA-512 哈希对象
    hash = hashlib.sha512()
    # 以块的形式读取文件内容并更新哈希对象
    for block in iter(lambda: file.read(blocksize), b""):
        hash.update(block)

    # 如果指定格式为十六进制字符串，则返回截取后的 256 位哈希值
    if format == "hexdigest":
        return hash.hexdigest()[0:64]
    # 否则返回截取后的 256 位哈希值的原始字节形式
    else:
        return hash.digest()[0:32]

# 计算文件的 SHA-256 哈希值
def sha256sum(file, blocksize=65536):
    # 如果传入的是文件名，则打开文件
    if type(file) is str:  
        file = open(file, "rb")
    # 创建 SHA-256 哈希对象
    hash = hashlib.sha256()
    # 以块的形式读取文件内容并更新哈希对象
    for block in iter(lambda: file.read(blocksize), b""):
        hash.update(block)
    # 返回文件的 SHA-256 哈希值
    return hash.hexdigest()

# 生成指定长度和编码方式的随机字符串
def random(length=64, encoding="hex"):
    # 如果指定编码方式为 base64，则生成随机字节流并进行 base64 编码
    if encoding == "base64":  
        hash = hashlib.sha512(os.urandom(256)).digest()
        return base64.b64encode(hash).decode("ascii").replace("+", "").replace("/", "").replace("=", "")[0:length]
    # 否则以十六进制编码方式生成随机字符串
    else:  
        return hashlib.sha512(os.urandom(256)).hexdigest()[0:length]

# 创建一个 SHA-512 哈希对象，截取后的 256 位哈希值
class Sha512t:
    def __init__(self, data):
        # 如果传入数据，则创建 SHA-512 哈希对象
        if data:
            self.sha512 = hashlib.sha512(data)
        # 否则创建空的 SHA-512 哈希对象
        else:
            self.sha512 = hashlib.sha512()

    # 返回截取后的 256 位哈希值的十六进制字符串形式
    def hexdigest(self):
        return self.sha512.hexdigest()[0:64]

    # 返回截取后的 256 位哈希值的原始字节形式
    def digest(self):
        return self.sha512.digest()[0:32]

    # 更新哈希对象的数据
    def update(self, data):
        return self.sha512.update(data)

# 创建一个 SHA-512 哈希对象，截取后的 256 位哈希值
def sha512t(data=None):
    return Sha512t(data)
```