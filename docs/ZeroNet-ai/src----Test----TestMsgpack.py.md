# `ZeroNet\src\Test\TestMsgpack.py`

```
# 导入所需的模块
import io
import os
import msgpack
import pytest
from Config import config
from util import Msgpack
from collections import OrderedDict

# 定义测试类 TestMsgpack
class TestMsgpack:
    # 定义测试数据
    test_data = OrderedDict(
        sorted({"cmd": "fileGet", "bin": b'p\x81zDhL\xf0O\xd0\xaf', "params": {"site": "1Site"}, "utf8": b'\xc3\xa1rv\xc3\xadzt\xc5\xb1r\xc5\x91'.decode("utf8"), "list": [b'p\x81zDhL\xf0O\xd0\xaf', b'p\x81zDhL\xf0O\xd0\xaf']}.items())
    )

    # 定义测试打包方法
    def testPacking(self):
        # 断言打包后的数据是否符合预期
        assert Msgpack.pack(self.test_data) == b'\x85\xa3bin\xc4\np\x81zDhL\xf0O\xd0\xaf\xa3cmd\xa7fileGet\xa4list\x92\xc4\np\x81zDhL\xf0O\xd0\xaf\xc4\np\x81zDhL\xf0O\xd0\xaf\xa6params\x81\xa4site\xa51Site\xa4utf8\xad\xc3\xa1rv\xc3\xadzt\xc5\xb1r\xc5\x91'
        # 断言打包后的数据是否符合预期，使用不同的参数
        assert Msgpack.pack(self.test_data, use_bin_type=False) == b'\x85\xa3bin\xaap\x81zDhL\xf0O\xd0\xaf\xa3cmd\xa7fileGet\xa4list\x92\xaap\x81zDhL\xf0O\xd0\xaf\xaap\x81zDhL\xf0O\xd0\xaf\xa6params\x81\xa4site\xa51Site\xa4utf8\xad\xc3\xa1rv\xc3\xadzt\xc5\xb1r\xc5\x91'

    # 定义测试解包方法
    def testUnpackinkg(self):
        # 断言解包后的数据是否与原始数据相同
        assert Msgpack.unpack(Msgpack.pack(self.test_data)) == self.test_data

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试
    @pytest.mark.parametrize("unpacker_class", [msgpack.Unpacker, msgpack.fallback.Unpacker])
    # 定义测试解包器方法
    def testUnpacker(self, unpacker_class):
        # 根据参数化传入的解包器类创建解包器对象
        unpacker = unpacker_class(raw=False)

        # 打包测试数据并使用二进制类型
        data = msgpack.packb(self.test_data, use_bin_type=True)
        data += msgpack.packb(self.test_data, use_bin_type=True)

        # 创建空列表存储消息
        messages = []
        # 遍历打包后的数据
        for char in data:
            # 将每个字符传入解包器
            unpacker.feed(bytes([char]))
            # 遍历解包器的消息
            for message in unpacker:
                # 将消息添加到列表中
                messages.append(message)

        # 断言消息列表的长度是否为2
        assert len(messages) == 2
        # 断言消息列表中的消息与测试数据相同
        assert messages[0] == self.test_data
        assert messages[0] == messages[1]
    # 测试流式处理
    def testStreaming(self):
        # 生成随机的二进制数据
        bin_data = os.urandom(20)
        # 创建一个 Msgpack 文件部分对象，以只读方式打开文件
        f = Msgpack.FilePart("%s/users.json" % config.data_dir, "rb")
        # 设置读取的字节数
        f.read_bytes = 30

        # 创建一个包含文件部分对象和二进制数据的字典
        data = {"cmd": "response", "body": f, "bin": bin_data}

        # 创建一个字节流对象
        out_buff = io.BytesIO()
        # 将数据流式写入字节流对象
        Msgpack.stream(data, out_buff.write)
        # 将字节流对象指针移动到开头
        out_buff.seek(0)

        # 创建一个包含文件部分对象和二进制数据的字典
        data_packb = {
            "cmd": "response",
            "body": open("%s/users.json" % config.data_dir, "rb").read(30),
            "bin": bin_data
        }

        # 将字节流对象指针移动到开头
        out_buff.seek(0)
        # 解包字节流对象中的数据
        data_unpacked = Msgpack.unpack(out_buff.read())
        # 断言解包后的数据与预期的数据相等
        assert data_unpacked == data_packb
        # 断言解包后的数据中的命令字段为 "response"
        assert data_unpacked["cmd"] == "response"
        # 断言解包后的数据中的 body 字段类型为 bytes
        assert type(data_unpacked["body"]) == bytes

    # 测试向后兼容性
    def testBackwardCompatibility(self):
        # 创建一个空字典
        packed = {}
        # 使用不同的参数对测试数据进行打包
        packed["py3"] = Msgpack.pack(self.test_data, use_bin_type=False)
        packed["py3_bin"] = Msgpack.pack(self.test_data, use_bin_type=True)
        # 遍历打包后的数据
        for key, val in packed.items():
            # 解包数据
            unpacked = Msgpack.unpack(val)
            # 断言解包后的 utf8 字段类型为 str
            type(unpacked["utf8"]) == str
            # 断言解包后的 bin 字段类型为 bytes

        # 使用 use_bin_type=False 进行打包（ZeroNet 0.7.0 之前的版本）
        # 解包数据
        unpacked = Msgpack.unpack(packed["py3"], decode=True)
        # 断言解包后的 utf8 字段类型为 str
        type(unpacked["utf8"]) == str
        # 断言解包后的 bin 字段类型为 bytes
        type(unpacked["bin"]) == bytes
        # 断言解包后的 utf8 字段长度为 9
        assert len(unpacked["utf8"]) == 9
        # 断言解包后的 bin 字段长度为 10
        assert len(unpacked["bin"]) == 10
        # 尝试将二进制数据按照 utf-8 解码，预期会抛出 UnicodeDecodeError
        with pytest.raises(UnicodeDecodeError) as err:
            unpacked = Msgpack.unpack(packed["py3"], decode=False)

        # 使用 use_bin_type=True 进行打包
        # 解包数据
        unpacked = Msgpack.unpack(packed["py3_bin"], decode=False)
        # 断言解包后的 utf8 字段类型为 str
        type(unpacked["utf8"]) == str
        # 断言解包后的 bin 字段类型为 bytes
        type(unpacked["bin"]) == bytes
        # 断言解包后的 utf8 字段长度为 9
        assert len(unpacked["utf8"]) == 9
        # 断言解包后的 bin 字段长度为 10
        assert len(unpacked["bin"]) == 10
```