# `ZeroNet\src\lib\pyaes\blockfeeder.py`

```py
# 导入所需的模块和函数
from .aes import AESBlockModeOfOperation, AESSegmentModeOfOperation, AESStreamModeOfOperation
from .util import append_PKCS7_padding, strip_PKCS7_padding, to_bufferable

# 定义两种填充方式
PADDING_NONE       = 'none'
PADDING_DEFAULT    = 'default'
# @TODO: Ciphertext stealing and explicit PKCS#7
# PADDING_CIPHERTEXT_STEALING
# PADDING_PKCS7

# ECB and CBC are block-only ciphers

# 检查是否可以处理指定大小的数据块，如果大于等于16则返回16，否则返回0
def _block_can_consume(self, size):
    if size >= 16: return 16
    return 0

# 在填充后，可能会有多个数据块
def _block_final_encrypt(self, data, padding = PADDING_DEFAULT):
    if padding == PADDING_DEFAULT:
        data = append_PKCS7_padding(data)

    elif padding == PADDING_NONE:
        if len(data) != 16:
            raise Exception('invalid data length for final block')
    else:
        raise Exception('invalid padding option')

    if len(data) == 32:
        return self.encrypt(data[:16]) + self.encrypt(data[16:])

    return self.encrypt(data)

def _block_final_decrypt(self, data, padding = PADDING_DEFAULT):
    if padding == PADDING_DEFAULT:
        return strip_PKCS7_padding(self.decrypt(data))

    if padding == PADDING_NONE:
        if len(data) != 16:
            raise Exception('invalid data length for final block')
        return self.decrypt(data)

    raise Exception('invalid padding option')

AESBlockModeOfOperation._can_consume = _block_can_consume
AESBlockModeOfOperation._final_encrypt = _block_final_encrypt
AESBlockModeOfOperation._final_decrypt = _block_final_decrypt

# CFB is a segment cipher

# 检查是否可以处理指定大小的数据块，返回可以处理的数据块大小
def _segment_can_consume(self, size):
    return self.segment_bytes * int(size // self.segment_bytes)

# CFB可以处理最后一个非段大小的数据块，使用剩余的密码块
def _segment_final_encrypt(self, data, padding = PADDING_DEFAULT):
    if padding != PADDING_DEFAULT:
        raise Exception('invalid padding option')

    faux_padding = (chr(0) * (self.segment_bytes - (len(data) % self.segment_bytes)))
    padded = data + to_bufferable(faux_padding)
    return self.encrypt(padded)[:len(data)]

# CFB可以处理最后一个非段大小的数据块，使用剩余的密码块
def _segment_final_decrypt(self, data, padding = PADDING_DEFAULT):
    # 如果填充选项不是默认值，则抛出异常
    if padding != PADDING_DEFAULT:
        raise Exception('invalid padding option')

    # 计算需要填充的字节数，补充相应数量的空字符
    faux_padding = (chr(0) * (self.segment_bytes - (len(data) % self.segment_bytes)))
    # 将数据和填充后的内容拼接在一起
    padded = data + to_bufferable(faux_padding)
    # 对填充后的数据进行解密，并截取原始数据长度的部分返回
    return self.decrypt(padded)[:len(data)]
# 设置 AESSegmentModeOfOperation 类的 _can_consume 属性为 _segment_can_consume 方法
AESSegmentModeOfOperation._can_consume = _segment_can_consume
# 设置 AESSegmentModeOfOperation 类的 _final_encrypt 属性为 _segment_final_encrypt 方法
AESSegmentModeOfOperation._final_encrypt = _segment_final_encrypt
# 设置 AESSegmentModeOfOperation 类的 _final_decrypt 属性为 _segment_final_decrypt 方法

# 定义 _stream_can_consume 方法，用于流密码的处理，返回指定大小的数据
def _stream_can_consume(self, size):
    return size

# 定义 _stream_final_encrypt 方法，用于流密码的最终加密，根据填充选项进行加密
def _stream_final_encrypt(self, data, padding = PADDING_DEFAULT):
    if padding not in [PADDING_NONE, PADDING_DEFAULT]:
        raise Exception('invalid padding option')
    return self.encrypt(data)

# 定义 _stream_final_decrypt 方法，用于流密码的最终解密，根据填充选项进行解密
def _stream_final_decrypt(self, data, padding = PADDING_DEFAULT):
    if padding not in [PADDING_NONE, PADDING_DEFAULT]:
        raise Exception('invalid padding option')
    return self.decrypt(data)

# 设置 AESStreamModeOfOperation 类的 _can_consume 属性为 _stream_can_consume 方法
AESStreamModeOfOperation._can_consume = _stream_can_consume
# 设置 AESStreamModeOfOperation 类的 _final_encrypt 属性为 _stream_final_encrypt 方法
AESStreamModeOfOperation._final_encrypt = _stream_final_encrypt
# 设置 AESStreamModeOfOperation 类的 _final_decrypt 属性为 _stream_final_decrypt 方法

# 定义 BlockFeeder 类，用于处理字节流的分块和填充
class BlockFeeder(object):
    '''The super-class for objects to handle chunking a stream of bytes
       into the appropriate block size for the underlying mode of operation
       and applying (or stripping) padding, as necessary.'''
    # 初始化方法，接受模式、输入数据、最终处理方法和填充选项
    def __init__(self, mode, feed, final, padding = PADDING_DEFAULT):
        self._mode = mode
        self._feed = feed
        self._final = final
        self._buffer = to_bufferable("")  # 初始化缓冲区为空字符串
        self._padding = padding  # 设置填充选项为传入的参数
    # 定义一个 feed 方法，用于提供要加密（或解密）的字节，并返回可能来自此次或之前调用 feed 的任何字节
    # 用 None 或空字符串调用以刷新操作模式，并返回任何最终字节；不能再调用 feed
    def feed(self, data = None):
        if self._buffer is None:
            # 如果缓冲区已经为空，则抛出数值错误
            raise ValueError('already finished feeder')

        # 如果 data 为 None，则进行最终处理；处理我们一直保留的多余字节
        if data is None:
            result = self._final(self._buffer, self._padding)
            self._buffer = None
            return result

        # 将 data 转换为可缓冲的数据，并添加到缓冲区中
        self._buffer += to_bufferable(data)

        # 我们保留 16 个字节，以便确定填充
        result = to_bufferable('')
        while len(self._buffer) > 16:
            # 确定可以处理的字节数
            can_consume = self._mode._can_consume(len(self._buffer) - 16)
            # 如果没有可以处理的字节数，则跳出循环
            if can_consume == 0: break
            # 处理缓冲区中的数据，并将结果添加到 result 中
            result += self._feed(self._buffer[:can_consume])
            # 从缓冲区中移除已处理的数据
            self._buffer = self._buffer[can_consume:]

        return result
class Encrypter(BlockFeeder):
    'Accepts bytes of plaintext and returns encrypted ciphertext.'

    def __init__(self, mode, padding = PADDING_DEFAULT):
        # 调用父类的初始化方法，传入加密模式、加密函数、最终加密函数和填充方式
        BlockFeeder.__init__(self, mode, mode.encrypt, mode._final_encrypt, padding)


class Decrypter(BlockFeeder):
    'Accepts bytes of ciphertext and returns decrypted plaintext.'

    def __init__(self, mode, padding = PADDING_DEFAULT):
        # 调用父类的初始化方法，传入解密模式、解密函数、最终解密函数和填充方式
        BlockFeeder.__init__(self, mode, mode.decrypt, mode._final_decrypt, padding)


# 8kb blocks
BLOCK_SIZE = (1 << 13)

def _feed_stream(feeder, in_stream, out_stream, block_size = BLOCK_SIZE):
    'Uses feeder to read and convert from in_stream and write to out_stream.'

    # 循环读取输入流的数据，使用feeder进行转换，并写入输出流
    while True:
        chunk = in_stream.read(block_size)
        if not chunk:
            break
        converted = feeder.feed(chunk)
        out_stream.write(converted)
    converted = feeder.feed()
    out_stream.write(converted)


def encrypt_stream(mode, in_stream, out_stream, block_size = BLOCK_SIZE, padding = PADDING_DEFAULT):
    'Encrypts a stream of bytes from in_stream to out_stream using mode.'

    # 创建Encrypter对象
    encrypter = Encrypter(mode, padding = padding)
    # 调用_feed_stream函数进行加密
    _feed_stream(encrypter, in_stream, out_stream, block_size)


def decrypt_stream(mode, in_stream, out_stream, block_size = BLOCK_SIZE, padding = PADDING_DEFAULT):
    'Decrypts a stream of bytes from in_stream to out_stream using mode.'

    # 创建Decrypter对象
    decrypter = Decrypter(mode, padding = padding)
    # 调用_feed_stream函数进行解密
    _feed_stream(decrypter, in_stream, out_stream, block_size)
```