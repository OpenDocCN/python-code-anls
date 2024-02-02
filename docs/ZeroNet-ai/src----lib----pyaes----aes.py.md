# `ZeroNet\src\lib\pyaes\aes.py`

```py
# 这是 AES 算法和 AES 常见操作模式的纯 Python 实现

# 参见：https://en.wikipedia.org/wiki/Advanced_Encryption_Standard

# 老实说，关于操作模式的最好描述是维基百科上精美的图表。它们在瞬间解释了我用文字无法达到的效果。因此，这里的内联文档比我希望的要简洁。
# 参见：https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation

# 还有一个有用的工具，PyCrypto，这是一个用 C 实现并带有 Python 绑定的加密库：
# https://www.dlitz.net/software/pycrypto/

# 支持的密钥长度：
#   128位
#   192位
#   256位

# 支持的操作模式：
#   ECB - 电子密码本
#   CBC - 密码块链接
#   CFB - 密码反馈
#   OFB - 输出反馈
#   CTR - 计数器

# 有关 API 详细信息和一般信息，请参阅 README.md
# 导入 copy 模块
import copy
# 导入 struct 模块
import struct

# 定义 __all__ 列表，包含 AES 相关的类名
__all__ = ["AES", "AESModeOfOperationCTR", "AESModeOfOperationCBC", "AESModeOfOperationCFB",
           "AESModeOfOperationECB", "AESModeOfOperationOFB", "AESModesOfOperation", "Counter"]

# 定义函数 _compact_word，将 4 个字节的列表合并成一个 32 位的整数
def _compact_word(word):
    return (word[0] << 24) | (word[1] << 16) | (word[2] << 8) | word[3]

# 定义函数 _string_to_bytes，将字符串转换为字节列表
def _string_to_bytes(text):
    return list(ord(c) for c in text)

# 定义函数 _bytes_to_string，将字节列表转换为字符串
def _bytes_to_string(binary):
    return "".join(chr(b) for b in binary)

# 定义函数 _concat_list，将两个列表连接起来
def _concat_list(a, b):
    return a + b

# Python 3 兼容性处理
try:
    # 如果 xrange 不存在，则定义 xrange 为 range
    xrange
except Exception:
    xrange = range

    # 如果 text 是 bytes 类型，则直接返回
    def _string_to_bytes(text):
        if isinstance(text, bytes):
            return text
        return [ord(c) for c in text]

    # 在 Python 3 中返回 bytes 类型
    def _bytes_to_string(binary):
        return bytes(binary)

    # Python 3 无法将列表连接到 bytes 类型，因此先将其转换为 bytes 类型再连接
    def _concat_list(a, b):
        return a + bytes(b)

# 基于 Rijndael 实现的 AES 类
class AES(object):
    '''Encapsulates the AES block cipher.

    You generally should not need this. Use the AESModeOfOperation classes
    below instead.'''

    # 根据密钥长度确定轮数
    number_of_rounds = {16: 10, 24: 12, 32: 14}

    # 轮常数
    rcon = [ 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91 ]

    # S-box 和逆 S-box
    # 定义一个包含十六进制数的列表
    S = [ 0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 ]
    Si =[ 0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb, 0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb, 0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e, 0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25, 0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92, 0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84, 0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06, 0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b, 0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73, 0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e, 0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b, 0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4, 0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f, 0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef, 0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61, 0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d ] 

    # 加密的转换
    # 解密的转换
    # 解密密钥扩展的转换
    # 使用 AES 块密码加密一块明文

    # 如果明文长度不是 16，则抛出数值错误异常
    if len(plaintext) != 16:
        raise ValueError('wrong block length')

    # 计算加密轮数
    rounds = len(self._Ke) - 1
    # 初始化轮数变量
    (s1, s2, s3) = [1, 2, 3]
    # 初始化临时变量数组
    a = [0, 0, 0, 0]

    # 将明文转换为（整数 ^ 密钥）
    t = [(_compact_word(plaintext[4 * i:4 * i + 4]) ^ self._Ke[0][i]) for i in xrange(0, 4)]

    # 应用轮次变换
    for r in xrange(1, rounds):
        for i in xrange(0, 4):
            a[i] = (self.T1[(t[ i          ] >> 24) & 0xFF] ^
                    self.T2[(t[(i + s1) % 4] >> 16) & 0xFF] ^
                    self.T3[(t[(i + s2) % 4] >>  8) & 0xFF] ^
                    self.T4[ t[(i + s3) % 4]        & 0xFF] ^
                    self._Ke[r][i])
        t = copy.copy(a)

    # 最后一轮是特殊的
    result = [ ]
    for i in xrange(0, 4):
        tt = self._Ke[rounds][i]
        result.append((self.S[(t[ i           ] >> 24) & 0xFF] ^ (tt >> 24)) & 0xFF)
        result.append((self.S[(t[(i + s1) % 4] >> 16) & 0xFF] ^ (tt >> 16)) & 0xFF)
        result.append((self.S[(t[(i + s2) % 4] >>  8) & 0xFF] ^ (tt >>  8)) & 0xFF)
        result.append((self.S[ t[(i + s3) % 4]        & 0xFF] ^  tt       ) & 0xFF)

    # 返回加密结果
    return result
    # 使用 AES 块密码解密一块密文
    def decrypt(self, ciphertext):
        # 如果密文长度不为 16，则抛出数值错误异常
        if len(ciphertext) != 16:
            raise ValueError('wrong block length')

        # 计算解密轮数
        rounds = len(self._Kd) - 1
        # 初始化位移量
        (s1, s2, s3) = [3, 2, 1]
        # 初始化临时变量数组
        a = [0, 0, 0, 0]

        # 将密文转换为（整数 ^ 密钥）
        t = [(_compact_word(ciphertext[4 * i:4 * i + 4]) ^ self._Kd[0][i]) for i in xrange(0, 4)]

        # 应用轮次变换
        for r in xrange(1, rounds):
            for i in xrange(0, 4):
                a[i] = (self.T5[(t[ i          ] >> 24) & 0xFF] ^
                        self.T6[(t[(i + s1) % 4] >> 16) & 0xFF] ^
                        self.T7[(t[(i + s2) % 4] >>  8) & 0xFF] ^
                        self.T8[ t[(i + s3) % 4]        & 0xFF] ^
                        self._Kd[r][i])
            t = copy.copy(a)

        # 最后一轮是特殊的
        result = [ ]
        for i in xrange(0, 4):
            tt = self._Kd[rounds][i]
            result.append((self.Si[(t[ i           ] >> 24) & 0xFF] ^ (tt >> 24)) & 0xFF)
            result.append((self.Si[(t[(i + s1) % 4] >> 16) & 0xFF] ^ (tt >> 16)) & 0xFF)
            result.append((self.Si[(t[(i + s2) % 4] >>  8) & 0xFF] ^ (tt >>  8)) & 0xFF)
            result.append((self.Si[ t[(i + s3) % 4]        & 0xFF] ^  tt       ) & 0xFF)

        # 返回解密结果
        return result
# 定义一个计数器对象，用于计数模式（CTR）操作
class Counter(object):
    '''A counter object for the Counter (CTR) mode of operation.

       To create a custom counter, you can usually just override the
       increment method.'''

    def __init__(self, initial_value = 1):
        # 将初始值转换为长度为16的字节数组
        self._counter = [ ((initial_value >> i) % 256) for i in xrange(128 - 8, -1, -8) ]

    # 定义属性，返回计数器的值
    value = property(lambda s: s._counter)

    # 增加计数器的值（溢出则归零）
    def increment(self):
        '''Increment the counter (overflow rolls back to 0).'''
        for i in xrange(len(self._counter) - 1, -1, -1):
            self._counter[i] += 1
            if self._counter[i] < 256: break
            # 进位
            self._counter[i] = 0
        # 溢出处理
        else:
            self._counter = [ 0 ] * len(self._counter)


# 定义需要使用块的AES操作的超类
class AESBlockModeOfOperation(object):
    '''Super-class for AES modes of operation that require blocks.'''
    def __init__(self, key):
        self._aes = AES(key)

    # 解密方法，抛出未实现异常
    def decrypt(self, ciphertext):
        raise Exception('not implemented')

    # 加密方法，抛出未实现异常
    def encrypt(self, plaintext):
        raise Exception('not implemented')


# 定义需要使用流密码的AES操作的超类
class AESStreamModeOfOperation(AESBlockModeOfOperation):
    '''Super-class for AES modes of operation that are stream-ciphers.'''


# 定义需要对数据进行分段的AES操作的超类
class AESSegmentModeOfOperation(AESStreamModeOfOperation):
    '''Super-class for AES modes of operation that segment data.'''
    # 定义分段字节数
    segment_bytes = 16


# 定义AES ECB模式操作的类
class AESModeOfOperationECB(AESBlockModeOfOperation):
    # AES Electronic Codebook Mode of Operation.
    # AES电子密码本模式操作
    
    # Block-cipher, so data must be padded to 16 byte boundaries
    # 块密码，因此数据必须填充到16字节边界
    
    # Security Notes:
    # 安全注意事项：
    # This mode is not recommended
    # 不推荐使用此模式
    # Any two identical blocks produce identical encrypted values, exposing data patterns. (See the image of Tux on wikipedia)
    # 任何两个相同的块产生相同的加密值，暴露数据模式。（参见维基百科上的Tux图像）
    
    # Also see:
    # 另请参阅：
    # https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Electronic_codebook_.28ECB.29
    # https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Electronic_codebook_.28ECB.29
    # See NIST SP800-38A (http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf); section 6.1
    # 参见NIST SP800-38A（http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf）；第6.1节
    
    # 设置名称为"Electronic Codebook (ECB)"
    name = "Electronic Codebook (ECB)"
    
    # 加密函数，接受明文参数
    def encrypt(self, plaintext):
        # 如果明文长度不等于16，则引发值错误异常
        if len(plaintext) != 16:
            raise ValueError('plaintext block must be 16 bytes')
    
        # 将明文转换为字节流
        plaintext = _string_to_bytes(plaintext)
        # 使用AES加密字节流，并将结果转换为字符串
        return _bytes_to_string(self._aes.encrypt(plaintext))
    
    # 解密函数，接受密文参数
    def decrypt(self, ciphertext):
        # 如果密文长度不等于16，则引发值错误异常
        if len(ciphertext) != 16:
            raise ValueError('ciphertext block must be 16 bytes')
    
        # 将密文转换为字节流
        ciphertext = _string_to_bytes(ciphertext)
        # 使用AES解密字节流，并将结果转换为字符串
        return _bytes_to_string(self._aes.decrypt(ciphertext))
# 定义 AES 密码块链模式的操作类，继承自 AESBlockModeOfOperation 类
class AESModeOfOperationCBC(AESBlockModeOfOperation):
    '''AES Cipher-Block Chaining Mode of Operation.

       o The Initialization Vector (IV)
       o Block-cipher, so data must be padded to 16 byte boundaries
       o An incorrect initialization vector will only cause the first
         block to be corrupt; all other blocks will be intact
       o A corrupt bit in the cipher text will cause a block to be
         corrupted, and the next block to be inverted, but all other
         blocks will be intact.

   Security Notes:
       o This method (and CTR) ARE recommended.

   Also see:
       o https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Cipher-block_chaining_.28CBC.29
       o See NIST SP800-38A (http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf); section 6.2'''

    # 设置名称为 "Cipher-Block Chaining (CBC)"
    name = "Cipher-Block Chaining (CBC)"

    # 初始化方法，接受密钥和初始化向量（IV）
    def __init__(self, key, iv = None):
        # 如果 IV 为空，则初始化上一个密文块为 16 个 0
        if iv is None:
            self._last_cipherblock = [ 0 ] * 16
        # 如果 IV 长度不为 16，则抛出数值错误
        elif len(iv) != 16:
            raise ValueError('initialization vector must be 16 bytes')
        else:
            self._last_cipherblock = _string_to_bytes(iv)

        # 调用父类的初始化方法，传入密钥
        AESBlockModeOfOperation.__init__(self, key)

    # 加密方法，接受明文
    def encrypt(self, plaintext):
        # 如果明文长度不为 16，则抛出数值错误
        if len(plaintext) != 16:
            raise ValueError('plaintext block must be 16 bytes')

        # 将明文转换为字节流
        plaintext = _string_to_bytes(plaintext)
        # 计算预密文块
        precipherblock = [ (p ^ l) for (p, l) in zip(plaintext, self._last_cipherblock) ]
        # 使用 AES 加密预密文块，更新上一个密文块
        self._last_cipherblock = self._aes.encrypt(precipherblock)

        # 返回上一个密文块
        return _bytes_to_string(self._last_cipherblock)

    # 解密方法，接受密文
    def decrypt(self, ciphertext):
        # 如果密文长度不为 16，则抛出数值错误
        if len(ciphertext) != 16:
            raise ValueError('ciphertext block must be 16 bytes')

        # 将密文转换为字节流
        cipherblock = _string_to_bytes(ciphertext)
        # 计算明文
        plaintext = [ (p ^ l) for (p, l) in zip(self._aes.decrypt(cipherblock), self._last_cipherblock) ]
        # 更新上一个密文块
        self._last_cipherblock = cipherblock

        # 返回明文
        return _bytes_to_string(plaintext)
class AESModeOfOperationCFB(AESSegmentModeOfOperation):
    '''AES Cipher Feedback Mode of Operation.

       o A stream-cipher, so input does not need to be padded to blocks,
         but does need to be padded to segment_size

    Also see:
       o https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Cipher_feedback_.28CFB.29
       o See NIST SP800-38A (http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf); section 6.3'''

    # 设置模式名称为 "Cipher Feedback (CFB)"
    name = "Cipher Feedback (CFB)"

    # 初始化方法，接受密钥、初始化向量和段大小作为参数
    def __init__(self, key, iv, segment_size = 1):
        # 如果段大小为0，则设置为1
        if segment_size == 0: segment_size = 1

        # 如果初始化向量为None，则将移位寄存器初始化为16个0
        if iv is None:
            self._shift_register = [ 0 ] * 16
        # 如果初始化向量长度不为16，则抛出数值错误
        elif len(iv) != 16:
            raise ValueError('initialization vector must be 16 bytes')
        else:
          self._shift_register = _string_to_bytes(iv)

        # 设置段字节数
        self._segment_bytes = segment_size

        # 调用父类的初始化方法，传入密钥
        AESBlockModeOfOperation.__init__(self, key)

    # 定义段字节数的属性
    segment_bytes = property(lambda s: s._segment_bytes)

    # 加密方法，接受明文作为参数
    def encrypt(self, plaintext):
        # 如果明文长度不是段大小的整数倍，则抛出数值错误
        if len(plaintext) % self._segment_bytes != 0:
            raise ValueError('plaintext block must be a multiple of segment_size')

        # 将明文转换为字节数组
        plaintext = _string_to_bytes(plaintext)

        # 将明文分成段
        encrypted = [ ]
        for i in xrange(0, len(plaintext), self._segment_bytes):
            plaintext_segment = plaintext[i: i + self._segment_bytes]
            xor_segment = self._aes.encrypt(self._shift_register)[:len(plaintext_segment)]
            cipher_segment = [ (p ^ x) for (p, x) in zip(plaintext_segment, xor_segment) ]

            # 将顶部位移出去，将密文加入
            self._shift_register = _concat_list(self._shift_register[len(cipher_segment):], cipher_segment)

            encrypted.extend(cipher_segment)

        return _bytes_to_string(encrypted)
    # 解密函数，接受密文作为参数
    def decrypt(self, ciphertext):
        # 如果密文长度不是段大小的整数倍，则抛出数值错误
        if len(ciphertext) % self._segment_bytes != 0:
            raise ValueError('ciphertext block must be a multiple of segment_size')

        # 将密文转换为字节流
        ciphertext = _string_to_bytes(ciphertext)

        # 将密文分成段
        decrypted = [ ]
        for i in xrange(0, len(ciphertext), self._segment_bytes):
            # 获取当前段的密文
            cipher_segment = ciphertext[i: i + self._segment_bytes]
            # 使用移位寄存器加密当前段的长度个字节
            xor_segment = self._aes.encrypt(self._shift_register)[:len(cipher_segment)]
            # 对当前段的密文和加密结果进行异或操作，得到明文段
            plaintext_segment = [ (p ^ x) for (p, x) in zip(cipher_segment, xor_segment) ]

            # 将移位寄存器的高位移出，将密文移入
            self._shift_register = _concat_list(self._shift_register[len(cipher_segment):], cipher_segment)

            # 将解密的明文段添加到解密结果中
            decrypted.extend(plaintext_segment)

        # 将解密结果转换为字符串并返回
        return _bytes_to_string(decrypted)
# 定义 AES 输出反馈模式的操作类，继承自 AES 流模式的操作类
class AESModeOfOperationOFB(AESStreamModeOfOperation):
    '''AES Output Feedback Mode of Operation.

       o A stream-cipher, so input does not need to be padded to blocks,
         allowing arbitrary length data.
       o A bit twiddled in the cipher text, twiddles the same bit in the
         same bit in the plain text, which can be useful for error
         correction techniques.

    Also see:
       o https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Output_feedback_.28OFB.29
       o See NIST SP800-38A (http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf); section 6.4'''

    # 设置模式名称为 "Output Feedback (OFB)"
    name = "Output Feedback (OFB)"

    # 初始化方法，接受密钥和初始化向量（可选）
    def __init__(self, key, iv = None):
        # 如果没有提供初始化向量，则使用全零的 16 字节数组
        if iv is None:
            self._last_precipherblock = [ 0 ] * 16
        # 如果提供的初始化向量长度不为 16 字节，则抛出数值错误
        elif len(iv) != 16:
            raise ValueError('initialization vector must be 16 bytes')
        else:
          self._last_precipherblock = _string_to_bytes(iv)

        self._remaining_block = [ ]

        # 调用父类的初始化方法，传入密钥
        AESBlockModeOfOperation.__init__(self, key)

    # 加密方法，接受明文输入
    def encrypt(self, plaintext):
        encrypted = [ ]
        # 遍历明文的每个字节
        for p in _string_to_bytes(plaintext):
            # 如果剩余块为空
            if len(self._remaining_block) == 0:
                # 使用 AES 加密上一个预加密块
                self._remaining_block = self._aes.encrypt(self._last_precipherblock)
                self._last_precipherblock = [ ]
            # 取出剩余块的第一个字节
            precipherbyte = self._remaining_block.pop(0)
            # 将明文字节与剩余块字节进行异或运算
            cipherbyte = p ^ precipherbyte
            encrypted.append(cipherbyte)

        return _bytes_to_string(encrypted)

    # 解密方法，接受密文输入
    def decrypt(self, ciphertext):
        # AES-OFB 是对称的，所以解密和加密相同
        return self.encrypt(ciphertext)



class AESModeOfOperationCTR(AESStreamModeOfOperation):
    # AES Counter Mode of Operation.
    # AES 计数器模式操作

       o A stream-cipher, so input does not need to be padded to blocks,
         allowing arbitrary length data.
       # 流密码，因此输入不需要填充到块中，允许任意长度的数据。

       o The counter must be the same size as the key size (ie. len(key))
       # 计数器必须与密钥大小相同（即 len(key)）

       o Each block independant of the other, so a corrupt byte will not
         damage future blocks.
       # 每个块都是独立的，因此损坏的字节不会损坏未来的块。

       o Each block has a unique counter value associated with it, which
         contributes to the encrypted value, so no data patterns are
         leaked.
       # 每个块都有与之关联的唯一计数器值，这有助于加密值，因此不会泄漏数据模式。

       o Also known as: Counter Mode (CM), Integer Counter Mode (ICM) and
         Segmented Integer Counter (SIC
       # 也称为：计数器模式（CM），整数计数器模式（ICM）和分段整数计数器（SIC）

   Security Notes:
       o This method (and CBC) ARE recommended.
       # 推荐使用此方法（和 CBC）。

       o Each message block is associated with a counter value which must be
         unique for ALL messages with the same key. Otherwise security may be
         compromised.
       # 每个消息块都与一个计数器值相关联，该值必须对于具有相同密钥的所有消息是唯一的。否则安全性可能会受到威胁。

    Also see:

       o https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_.28CTR.29
       # 参见：https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_.28CTR.29

       o See NIST SP800-38A (http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf); section 6.5
         and Appendix B for managing the initial counter
       # 参见 NIST SP800-38A（http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf）；第 6.5 节和附录 B 管理初始计数器

    name = "Counter (CTR)"

    def __init__(self, key, counter = None):
        # 初始化方法，接受密钥和计数器作为参数
        AESBlockModeOfOperation.__init__(self, key)

        if counter is None:
            counter = Counter()

        self._counter = counter
        self._remaining_counter = [ ]

    def encrypt(self, plaintext):
        # 加密方法，接受明文作为参数
        while len(self._remaining_counter) < len(plaintext):
            self._remaining_counter += self._aes.encrypt(self._counter.value)
            self._counter.increment()

        plaintext = _string_to_bytes(plaintext)

        encrypted = [ (p ^ c) for (p, c) in zip(plaintext, self._remaining_counter) ]
        self._remaining_counter = self._remaining_counter[len(encrypted):]

        return _bytes_to_string(encrypted)

    def decrypt(self, crypttext):
        # 解密方法，接受密文作为参数
        # AES-CTR is symetric
        return self.encrypt(crypttext)
# 创建一个简单的查找表，用于将每种模式映射到相应的操作类
AESModesOfOperation = dict(
    ctr = AESModeOfOperationCTR,  # 将 'ctr' 映射到 AESModeOfOperationCTR 类
    cbc = AESModeOfOperationCBC,  # 将 'cbc' 映射到 AESModeOfOperationCBC 类
    cfb = AESModeOfOperationCFB,  # 将 'cfb' 映射到 AESModeOfOperationCFB 类
    ecb = AESModeOfOperationECB,  # 将 'ecb' 映射到 AESModeOfOperationECB 类
    ofb = AESModeOfOperationOFB,  # 将 'ofb' 映射到 AESModeOfOperationOFB 类
)
```