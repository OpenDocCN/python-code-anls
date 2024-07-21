# `.\pytorch\test\test_serialization.py`

```py
# Owner(s): ["module: serialization"]

# 导入必要的模块和库
import copy  # 导入复制操作相关的模块
import gc  # 导入垃圾回收模块
import gzip  # 导入处理 gzip 文件的模块
import io  # 导入处理 IO 操作的模块
import os  # 导入操作系统相关的模块
import pickle  # 导入 pickle 序列化模块
import platform  # 导入获取平台信息的模块
import shutil  # 导入文件和目录操作相关的模块
import sys  # 导入系统相关的模块
import tempfile  # 导入处理临时文件和目录的模块
import unittest  # 导入单元测试框架模块
import warnings  # 导入警告处理模块
import zipfile  # 导入处理 ZIP 文件的模块
from collections import namedtuple, OrderedDict  # 导入具名元组和有序字典
from copy import deepcopy  # 导入深度复制函数
from itertools import product  # 导入迭代工具模块中的笛卡尔积函数

from pathlib import Path  # 导入处理路径的模块

import torch  # 导入 PyTorch 深度学习框架
from torch._utils import _rebuild_tensor  # 导入内部工具函数，用于重建张量
from torch._utils_internal import get_file_path_2  # 导入内部工具函数，获取文件路径
from torch.serialization import (  # 导入 PyTorch 序列化模块的相关函数和类
    check_module_version_greater_or_equal,  # 检查模块版本是否大于等于给定版本
    get_default_load_endianness,  # 获取默认加载字节顺序
    LoadEndianness,  # 加载字节顺序枚举
    set_default_load_endianness,  # 设置默认加载字节顺序
    SourceChangeWarning,  # 源码变更警告类
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests  # 导入设备类型测试工具函数
from torch.testing._internal.common_dtype import all_types_and_complex_and  # 导入数据类型测试工具函数
from torch.testing._internal.common_utils import (  # 导入常用测试工具函数
    AlwaysWarnTypedStorageRemoval,  # 总是警告类型存储移除类
    BytesIOContext,  # 字节流上下文管理器
    download_file,  # 下载文件函数
    instantiate_parametrized_tests,  # 实例化参数化测试
    IS_FBCODE,  # 是否为 FBCODE 环境
    IS_FILESYSTEM_UTF8_ENCODING,  # 是否支持文件系统 UTF-8 编码
    IS_WINDOWS,  # 是否在 Windows 环境下
    parametrize,  # 参数化装饰器
    run_tests,  # 运行测试函数
    serialTest,  # 序列化测试类
    skipIfTorchDynamo,  # 如果是 Torch Dynamo 环境则跳过测试
    TemporaryDirectoryName,  # 临时目录名类
    TemporaryFileName,  # 临时文件名类
    TEST_DILL,  # 测试 dill 序列化
    TestCase,  # 测试用例基类
)
from torch.testing._internal.two_tensor import TwoTensor  # 导入双张量测试工具类

from torch.utils._import_utils import import_dill  # 导入导入 dill 模块的工具函数

# 如果不在 Windows 环境下，导入 mmap 模块的两个常量
if not IS_WINDOWS:
    from mmap import MAP_PRIVATE, MAP_SHARED
else:
    MAP_SHARED, MAP_PRIVATE = None, None

# 下面的测试用例代码大部分源自于 `test/test_torch.py`，详细信息请查看具体的 Git 提交记录，
# 可以访问链接 https://github.com/pytorch/pytorch/blame/9a2691f2fc948b9792686085b493c61793c2de30/test/test_torch.py

# 导入 dill 模块，并检查其版本是否至少为 0.3.1
dill = import_dill()
HAS_DILL_AT_LEAST_0_3_1 = dill is not None and check_module_version_greater_or_equal(dill, (0, 3, 1))

# 初始化变量 `can_retrieve_source`，用于检测是否能够检索源代码
can_retrieve_source = True
with warnings.catch_warnings(record=True) as warns:
    with tempfile.NamedTemporaryFile() as checkpoint:
        x = torch.save(torch.nn.Module(), checkpoint)  # 将 PyTorch 模块保存到临时文件中
        for warn in warns:
            if "Couldn't retrieve source code" in warn.message.args[0]:
                can_retrieve_source = False  # 如果警告中包含无法检索源代码的信息，则设置为 False
                break

# 定义一个模拟文件类 `FilelikeMock`
class FilelikeMock:
    def __init__(self, data, has_fileno=True, has_readinto=False):
        if has_readinto:
            self.readinto = self.readinto_opt  # 如果需要支持 readinto 方法，则使用优化后的版本
        if has_fileno:
            self.fileno = self.fileno_opt  # 如果需要支持 fileno 方法，则使用优化后的版本

        self.calls = set()  # 初始化调用方法集合
        self.bytesio = io.BytesIO(data)  # 使用给定的数据创建字节流对象

        # 定义追踪方法调用的函数
        def trace(fn, name):
            def result(*args, **kwargs):
                self.calls.add(name)  # 记录方法调用名称到调用集合中
                return fn(*args, **kwargs)
            return result

        # 对字节流对象的读写等方法进行追踪
        for attr in ['read', 'readline', 'seek', 'tell', 'write', 'flush']:
            traced_fn = trace(getattr(self.bytesio, attr), attr)
            setattr(self, attr, traced_fn)

    def fileno_opt(self):
        raise io.UnsupportedOperation('Not a real file')  # 如果调用了不支持的 fileno 方法，则抛出异常
    # 将'readinto'添加到调用集合中，表示readinto方法被调用过
    def readinto_opt(self, view):
        self.calls.add('readinto')
        # 调用BytesIO对象的readinto方法，将数据读入到提供的视图中，并返回读取的字节数
        return self.bytesio.readinto(view)
    
    # 检查给定名称是否在调用集合中，用于判断特定方法是否被调用过
    def was_called(self, name):
        return name in self.calls
class SerializationMixin:
    # 定义一个序列化的Mixin类，提供了一些与序列化相关的方法

    def _test_serialization_data(self):
        # 生成测试数据的方法
        a = [torch.randn(5, 5).float() for i in range(2)]
        # 创建两个5x5的随机浮点数张量列表a

        b = [a[i % 2] for i in range(4)]  # 0-3
        # 根据循环生成的索引，创建长度为4的列表b，其中包含a列表中的张量

        b += [a[0].storage()]  # 4
        # 将a列表中第一个张量的存储(storage)添加到b列表中的第4个位置

        b += [a[0].reshape(-1)[1:4].storage()]  # 5
        # 将a列表中第一个张量经过reshape后的部分（从索引1到3）的存储添加到b列表中的第5个位置

        b += [torch.arange(1, 11).int()]  # 6
        # 创建一个从1到10的整数张量，并添加到b列表的第6个位置

        t1 = torch.FloatTensor().set_(a[0].reshape(-1)[1:4].clone().storage(), 0, (3,), (1,))
        # 创建一个新的浮点数张量t1，使用a列表中第一个张量reshape后的部分（从索引1到3）的克隆存储
        # 设置其偏移为0，大小为(3,)，步长为(1,)

        t2 = torch.FloatTensor().set_(a[0].reshape(-1)[1:4].clone().storage(), 0, (3,), (1,))
        # 创建另一个新的浮点数张量t2，使用a列表中第一个张量reshape后的部分（从索引1到3）的克隆存储
        # 设置其偏移为0，大小为(3,)，步长为(1,)

        b += [(t1.storage(), t1.storage(), t2.storage())]  # 7
        # 将t1和t2的存储分别作为元组添加到b列表的第7个位置

        b += [a[0].reshape(-1)[0:2].storage()]  # 8
        # 将a列表中第一个张量reshape后的部分（从索引0到1）的存储添加到b列表的第8个位置

        return b
        # 返回生成的列表b作为测试数据

    def _test_serialization_assert(self, b, c):
        # 对测试数据进行断言的方法

        self.assertEqual(b, c, atol=0, rtol=0)
        # 使用断言检查b和c是否相等，绝对误差(atol)和相对误差(rtol)均为0

        self.assertTrue(isinstance(c[0], torch.FloatTensor))
        # 使用断言检查c列表中第一个元素是否为torch.FloatTensor类型

        self.assertTrue(isinstance(c[1], torch.FloatTensor))
        # 使用断言检查c列表中第二个元素是否为torch.FloatTensor类型

        self.assertTrue(isinstance(c[2], torch.FloatTensor))
        # 使用断言检查c列表中第三个元素是否为torch.FloatTensor类型

        self.assertTrue(isinstance(c[3], torch.FloatTensor))
        # 使用断言检查c列表中第四个元素是否为torch.FloatTensor类型

        self.assertTrue(isinstance(c[4], torch.storage.TypedStorage))
        # 使用断言检查c列表中第五个元素是否为torch.storage.TypedStorage类型

        self.assertEqual(c[4].dtype, torch.float)
        # 使用断言检查c列表中第五个元素的数据类型是否为torch.float

        c[0].fill_(10)
        # 将c列表中第一个元素填充为10

        self.assertEqual(c[0], c[2], atol=0, rtol=0)
        # 使用断言检查c列表中第一个元素和第三个元素是否相等，绝对误差和相对误差均为0

        self.assertEqual(c[4], torch.FloatStorage(25).fill_(10), atol=0, rtol=0)
        # 使用断言检查c列表中第五个元素是否与torch.FloatStorage(25).fill_(10)相等，绝对误差和相对误差均为0

        c[1].fill_(20)
        # 将c列表中第二个元素填充为20

        self.assertEqual(c[1], c[3], atol=0, rtol=0)
        # 使用断言检查c列表中第二个元素和第四个元素是否相等，绝对误差和相对误差均为0

        # 由于没有直接切片存储的方法，我不得不采用这种绕过的方式
        for i in range(4):
            self.assertEqual(c[4][i + 1], c[5][i])
        # 使用循环检查c列表中第五个元素和第六个元素的部分存储是否相等

        # 检查序列化相同存储视图对象时，是否解析为一个对象而不是两个（反之亦然）
        views = c[7]
        self.assertEqual(views[0]._cdata, views[1]._cdata)
        # 使用断言检查views列表中第一个元素和第二个元素的_cdata属性是否相等

        self.assertEqual(views[0], views[2])
        # 使用断言检查views列表中第一个元素和第三个元素是否相等

        self.assertNotEqual(views[0]._cdata, views[2]._cdata)
        # 使用断言检查views列表中第一个元素的_cdata属性和第三个元素的_cdata属性是否不相等

        rootview = c[8]
        # 将c列表中第九个元素赋值给rootview

        self.assertEqual(rootview.data_ptr(), c[0].data_ptr())
        # 使用断言检查rootview的数据指针和c列表中第一个元素的数据指针是否相等

    def test_serialization_zipfile_utils(self):
        # 测试序列化文件工具的方法

        data = {
            'a': b'12039810948234589',
            'b': b'1239081209484958',
            'c/d': b'94589480984058'
        }
        # 创建包含键值对的数据字典

        def test(name_or_buffer):
            # 定义一个测试函数，接受文件名或缓冲区作为参数

            with torch.serialization._open_zipfile_writer(name_or_buffer) as zip_file:
                # 使用torch.serialization._open_zipfile_writer打开一个zip文件写入器
                for key in data:
                    zip_file.write_record(key, data[key], len(data[key]))
                # 遍历数据字典，将每个键值对写入zip文件中

            if hasattr(name_or_buffer, 'seek'):
                name_or_buffer.seek(0)
            # 如果name_or_buffer具有'seek'方法，则将其指针移动到文件开头

            with torch.serialization._open_zipfile_reader(name_or_buffer) as zip_file:
                # 使用torch.serialization._open_zipfile_reader打开一个zip文件阅读器
                for key in data:
                    actual = zip_file.get_record(key)
                    expected = data[key]
                    self.assertEqual(expected, actual)
                # 遍历数据字典，读取每个键值对，并使用断言检查读取的值与期望的值是否相等

        with tempfile.NamedTemporaryFile() as f:
            test(f)
        # 使用临时命名文件执行测试

        with TemporaryFileName() as fname:
            test(fname)
        # 使用临时文件名执行测试

        test(io.BytesIO())
        # 使用字节流执行测试
    # 定义一个私有方法，用于测试序列化功能，支持只保存权重的选项
    def _test_serialization(self, weights_only):
        # 使用真实文件进行序列化测试
        # 获取测试数据
        b = self._test_serialization_data()
        
        # 使用临时命名文件进行保存和加载测试
        with tempfile.NamedTemporaryFile() as f:
            # 将数据 b 保存到文件 f 中
            torch.save(b, f)
            # 将文件指针移动到文件开头
            f.seek(0)
            # 从文件 f 中加载数据 c
            c = torch.load(f, weights_only=weights_only)
            # 断言加载的数据 c 与原始数据 b 相等
            self._test_serialization_assert(b, c)
        
        # 使用临时文件名进行保存和加载测试
        with TemporaryFileName() as fname:
            # 将数据 b 保存到文件 fname 中
            torch.save(b, fname)
            # 从文件 fname 中加载数据 c
            c = torch.load(fname, weights_only=weights_only)
            # 断言加载的数据 c 与原始数据 b 相等
            self._test_serialization_assert(b, c)
        
        # 测试字节数组/字符串的非 ASCII 编码
        # 下面的字节串是在 Python 2.7.12 和 PyTorch 0.4.1 中通过序列化生成的
        # 包含了一些 UTF-8 字符的字节 (即 `utf8_str.encode('utf-8')`).
        serialized = (
            b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9\x03.'
            b'\x80\x02}q\x01(U\x10protocol_versionq\x02M\xe9\x03U\n'
            b'type_sizesq\x03}q\x04(U\x03intq\x05K\x04U\x05shortq\x06K\x02U'
            b'\x04longq\x07K\x04uU\rlittle_endianq\x08\x88u.\x80\x02]q'
            b'\x01(U\x0e\xc5\xbc\xc4\x85\xc4\x85\xc3\xb3\xc5\xbc\xc4\x85'
            b'\xc5\xbcq\x02ctorch._utils\n_rebuild_tensor_v2\nq\x03((U'
            b'\x07storageq\x04ctorch\nFloatStorage\nq\x05U\x0845640624q'
            b'\x06U\x03cpuq\x07\x8a\x01\x01NtQK\x00K\x01\x85K\x01\x85'
            b'\x89NtRq\x08K\x02e.\x80\x02]q\x01U\x0845640624q\x02a.\x01\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        )
        
        # 将字节串 serialized 转换为 BytesIO 对象
        buf = io.BytesIO(serialized)
        # 将 UTF-8 编码的字节 utf8_bytes 解码成字符串 utf8_str
        utf8_bytes = b'\xc5\xbc\xc4\x85\xc4\x85\xc3\xb3\xc5\xbc\xc4\x85\xc5\xbc'
        utf8_str = utf8_bytes.decode('utf-8')
        
        # 从 buf 中加载数据 loaded_utf8，使用 weights_only 参数，并指定编码为 UTF-8
        loaded_utf8 = torch.load(buf, weights_only=weights_only, encoding='utf-8')
        # 断言加载的数据 loaded_utf8 与预期结果 [utf8_str, torch.zeros(1, dtype=torch.float), 2] 相等
        self.assertEqual(loaded_utf8, [utf8_str, torch.zeros(1, dtype=torch.float), 2])
        
        # 将 buf 的文件指针移动到开头
        buf.seek(0)
        # 从 buf 中加载数据 loaded_bytes，使用 weights_only 参数，并指定编码为 bytes
        loaded_bytes = torch.load(buf, weights_only=weights_only, encoding='bytes')
        # 断言加载的数据 loaded_bytes 与预期结果 [utf8_bytes, torch.zeros(1, dtype=torch.float), 2] 相等
        self.assertEqual(loaded_bytes, [utf8_bytes, torch.zeros(1, dtype=torch.float), 2])

    # 测试序列化功能，调用 _test_serialization 方法并传入 False
    def test_serialization(self):
        self._test_serialization(False)

    # 安全的测试序列化功能，调用 _test_serialization 方法并传入 True
    def test_serialization_safe(self):
        self._test_serialization(True)

    # 使用类似文件对象进行测试序列化的功能
    def test_serialization_filelike(self):
        # 获取测试数据
        b = self._test_serialization_data()
        
        # 使用 BytesIOContext 上下文管理器创建字节流对象 f
        with BytesIOContext() as f:
            # 将数据 b 保存到字节流对象 f 中
            torch.save(b, f)
            # 将字节流对象 f 的文件指针移动到开头
            f.seek(0)
            # 从字节流对象 f 中加载数据 c
            c = torch.load(f)
        
        # 断言加载的数据 c 与原始数据 b 相等
        self._test_serialization_assert(b, c)
    # 定义测试方法：测试伪造的 ZIP 序列化
    def test_serialization_fake_zip(self):
        # 准备测试数据，包含 ZIP 文件的头部信息
        data = [
            ord('P'),
            ord('K'),
            5,
            6
        ]
        # 添加 100 个字节的零填充数据
        for i in range(0, 100):
            data.append(0)
        # 使用 Torch 创建一个无符号整数张量
        t = torch.tensor(data, dtype=torch.uint8)

        # 使用临时文件来保存序列化后的张量
        with tempfile.NamedTemporaryFile() as f:
            # 将张量保存到临时文件中
            torch.save(t, f)

            # 检查文件是否是有效的 ZIP 文件
            self.assertTrue(zipfile.is_zipfile(f))
            # 检查文件不是 Torch 的序列化文件
            self.assertFalse(torch.serialization._is_zipfile(f))
            # 将文件指针移到文件开头
            f.seek(0)
            # 从文件中加载数据，检查加载的数据与原始数据是否一致
            self.assertEqual(torch.load(f), t)

    # 定义测试方法：测试使用 gzip 文件进行序列化
    def test_serialization_gzip(self):
        # 调用辅助方法获取测试数据
        b = self._test_serialization_data()
        # 创建两个临时文件用于保存数据
        f1 = tempfile.NamedTemporaryFile(delete=False)
        f2 = tempfile.NamedTemporaryFile(delete=False)
        # 将数据保存到第一个临时文件中
        torch.save(b, f1)
        # 打开第一个文件并以 gzip 格式写入到第二个文件中
        with open(f1.name, 'rb') as f_in, gzip.open(f2.name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        # 使用 gzip 打开第二个文件并加载数据
        with gzip.open(f2.name, 'rb') as f:
            c = torch.load(f)
        # 调用辅助方法检查加载的数据与原始数据是否一致
        self._test_serialization_assert(b, c)

    # 跳过测试条件：如果 dill 未安装或版本不符合要求
    @unittest.skipIf(
        not TEST_DILL or HAS_DILL_AT_LEAST_0_3_1,
        '"dill" not found or is correct version'
    )
    # 定义测试方法：测试 dill 版本不支持的情况
    def test_serialization_dill_version_not_supported(self):
        # 生成一个 5x5 的随机张量
        x = torch.randn(5, 5)

        # 使用临时文件保存张量，并期望抛出 ValueError 异常
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaisesRegex(ValueError, 'supports dill >='):
                torch.save(x, f, pickle_module=dill)
            # 将文件指针移到文件开头
            f.seek(0)
            with self.assertRaisesRegex(ValueError, 'supports dill >='):
                # 从文件中加载数据，期望抛出 ValueError 异常
                x2 = torch.load(f, pickle_module=dill, encoding='utf-8')

    # 定义测试方法：测试 pickle 模块抛出异常的情况
    def test_pickle_module(self):
        # 定义一个自定义的 Unpickler 类，加载时抛出 RuntimeError 异常
        class ThrowingUnpickler(pickle.Unpickler):
            def load(self, *args, **kwargs):
                raise RuntimeError("rumpelstiltskin")

        # 定义一个包含抛出异常 Unpickler 的类
        class ThrowingModule:
            Unpickler = ThrowingUnpickler
            load = ThrowingUnpickler.load

        # 生成一个 3x3 的单位矩阵张量
        x = torch.eye(3)
        # 使用临时文件保存张量
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            # 从文件中加载数据，期望抛出 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, "rumpelstiltskin"):
                torch.load(f, pickle_module=ThrowingModule)
            f.seek(0)
            # 再次加载数据，检查加载的数据与原始数据是否一致
            z = torch.load(f)
        # 断言加载的数据与原始数据是否一致
        self.assertEqual(x, z)

    # 跳过测试条件：如果 dill 未安装或版本不符合要求
    @unittest.skipIf(
        not TEST_DILL or not HAS_DILL_AT_LEAST_0_3_1,
        '"dill" not found or not correct version'
    )
    # 定义测试函数，测试使用 dill 序列化和反序列化 PyTorch 张量对象
    def test_serialization_dill(self):
        # 创建一个大小为 5x5 的随机张量 x
        x = torch.randn(5, 5)

        # 使用临时文件来保存序列化后的张量 x
        with tempfile.NamedTemporaryFile() as f:
            # 使用 dill 模块将张量 x 序列化并保存到文件 f 中
            torch.save(x, f, pickle_module=dill)
            # 将文件指针移动到文件开头
            f.seek(0)
            # 从文件 f 中加载并反序列化张量 x2，使用 dill 模块，指定编码为 utf-8
            x2 = torch.load(f, pickle_module=dill, encoding='utf-8')
            # 检查 x2 是否与 x 是同一类型的对象
            self.assertIsInstance(x2, type(x))
            # 检查 x2 是否与 x 的值相等
            self.assertEqual(x, x2)
            # 将文件指针移动到文件开头
            f.seek(0)
            # 再次从文件 f 中加载并反序列化张量 x3，使用 dill 模块
            x3 = torch.load(f, pickle_module=dill)
            # 检查 x3 是否与 x 是同一类型的对象
            self.assertIsInstance(x3, type(x))
            # 检查 x3 是否与 x 的值相等
            self.assertEqual(x, x3)

    # 定义测试函数，测试在 gzip 文件中偏移量序列化和反序列化 PyTorch 张量对象
    def test_serialization_offset_gzip(self):
        # 创建一个大小为 5x5 的随机张量 a
        a = torch.randn(5, 5)
        # 定义整数 i
        i = 41
        # 创建两个临时文件 f1 和 f2，并设置 delete=False 防止删除
        f1 = tempfile.NamedTemporaryFile(delete=False)
        f2 = tempfile.NamedTemporaryFile(delete=False)
        
        # 将整数 i 和张量 a 以 pickle 格式依次写入文件 f1
        with open(f1.name, 'wb') as f:
            pickle.dump(i, f)
            torch.save(a, f)
        
        # 打开文件 f1 和创建的 gzip 文件 f2，将 f1 的内容复制到 f2
        with open(f1.name, 'rb') as f_in, gzip.open(f2.name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        
        # 打开 gzip 文件 f2，并从中加载整数 j 和张量 b
        with gzip.open(f2.name, 'rb') as f:
            j = pickle.load(f)
            b = torch.load(f)
        
        # 检查张量 a 和 b 是否相等
        self.assertTrue(torch.equal(a, b))
        # 检查整数 i 和 j 是否相等
        self.assertEqual(i, j)

    # 定义内部测试函数，测试稀疏张量的序列化和反序列化
    def _test_serialization_sparse(self, weights_only):
        # 定义内部函数，用于测试指定类型的稀疏张量的序列化和反序列化
        def _test_serialization(conversion):
            # 创建一个大小为 3x3 的零张量 x，并将其中一个元素设为 1
            x = torch.zeros(3, 3)
            x[1][1] = 1
            # 将张量 x 转换为稀疏张量格式 conversion，并保存到临时文件 f 中
            with tempfile.NamedTemporaryFile() as f:
                torch.save({"tensor": x}, f)
                f.seek(0)
                # 从文件 f 中加载并反序列化稀疏张量 y，根据 weights_only 参数确定是否只加载权重
                y = torch.load(f, weights_only=weights_only)
                # 检查反序列化后的稀疏张量 y["tensor"] 是否与原始张量 x 相等
                self.assertEqual(x, y["tensor"], exact_is_coalesced=True)
        
        # 分别使用 lambda 表达式调用 _test_serialization 函数，测试不同类型的稀疏张量
        _test_serialization(lambda x: x.to_sparse())
        _test_serialization(lambda x: x.to_sparse_csr())
        _test_serialization(lambda x: x.to_sparse_csc())
        _test_serialization(lambda x: x.to_sparse_bsr((1, 1)))
        _test_serialization(lambda x: x.to_sparse_bsc((1, 1)))

    # 定义测试函数，测试非安全模式下的稀疏张量序列化和反序列化
    def test_serialization_sparse(self):
        # 调用 _test_serialization_sparse 函数，传入 False，表示非安全模式
        self._test_serialization_sparse(False)

    # 定义测试函数，测试安全模式下的稀疏张量序列化和反序列化
    def test_serialization_sparse_safe(self):
        # 调用 _test_serialization_sparse 函数，传入 True，表示安全模式
        self._test_serialization_sparse(True)
    def test_serialization_sparse_invalid(self):
        x = torch.zeros(3, 3)
        x[1][1] = 1
        x = x.to_sparse()

        class TensorSerializationSpoofer:
            def __init__(self, tensor):
                self.tensor = tensor

            def __reduce_ex__(self, proto):
                # 复制稀疏张量的索引
                invalid_indices = self.tensor._indices().clone()
                # 修改索引的第一个维度的第一个元素为 3，制造不一致情况
                invalid_indices[0][0] = 3
                # 返回用于重建稀疏张量的函数和参数
                return (
                    torch._utils._rebuild_sparse_tensor,
                    (
                        self.tensor.layout,
                        (
                            invalid_indices,
                            self.tensor._values(),
                            self.tensor.size())))

        # 使用临时文件来保存序列化后的张量
        with tempfile.NamedTemporaryFile() as f:
            # 将 spoofed 对象保存到文件 f 中
            torch.save({"spoofed": TensorSerializationSpoofer(x)}, f)
            f.seek(0)
            # 加载文件 f 并期望抛出运行时错误，错误消息为 "size is inconsistent with indices"
            with self.assertRaisesRegex(
                    RuntimeError,
                    "size is inconsistent with indices"):
                y = torch.load(f)

    def _test_serialization_sparse_compressed_invalid(self,
                                                      conversion,
                                                      get_compressed_indices,
                                                      get_plain_indices):
        x = torch.zeros(3, 3)
        x[1][1] = 1
        # 应用转换函数将稀疏张量转换为压缩格式
        x = conversion(x)

        class TensorSerializationSpoofer:
            def __init__(self, tensor):
                self.tensor = tensor

            def __reduce_ex__(self, proto):
                # 获取稀疏张量的压缩索引并进行克隆
                invalid_compressed_indices = get_compressed_indices(self.tensor).clone()
                # 修改压缩索引的第一个元素为 3，制造不一致情况
                invalid_compressed_indices[0] = 3
                # 返回用于重建稀疏张量的函数和参数
                return (
                    torch._utils._rebuild_sparse_tensor,
                    (
                        self.tensor.layout,
                        (
                            invalid_compressed_indices,
                            get_plain_indices(self.tensor),
                            self.tensor.values(),
                            self.tensor.size())))

        # 根据不同的布局类型确定压缩索引的名称
        if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
            compressed_indices_name = 'crow_indices'
        else:
            compressed_indices_name = 'ccol_indices'

        # 使用临时文件来保存序列化后的张量
        with tempfile.NamedTemporaryFile() as f:
            # 将 spoofed 对象保存到文件 f 中
            torch.save({"spoofed": TensorSerializationSpoofer(x)}, f)
            f.seek(0)
            # 加载文件 f 并期望抛出运行时错误，错误消息中包含对应的压缩索引名称
            with self.assertRaisesRegex(
                    RuntimeError,
                    f"`{compressed_indices_name}[[]..., 0[]] == 0` is not satisfied."):
                y = torch.load(f)

    def test_serialization_sparse_csr_invalid(self):
        # 调用 _test_serialization_sparse_compressed_invalid 方法测试 CSR 格式的稀疏张量序列化异常
        self._test_serialization_sparse_compressed_invalid(
            torch.Tensor.to_sparse_csr, torch.Tensor.crow_indices, torch.Tensor.col_indices)
    # 测试稀疏矩阵 CSC 格式的序列化是否有效
    def test_serialization_sparse_csc_invalid(self):
        # 调用 _test_serialization_sparse_compressed_invalid 方法，传入相关函数和参数
        self._test_serialization_sparse_compressed_invalid(
            torch.Tensor.to_sparse_csc,  # 使用 Torch 的 to_sparse_csc 方法
            torch.Tensor.ccol_indices,   # Torch Tensor 的列索引
            torch.Tensor.row_indices     # Torch Tensor 的行索引
        )
    
    # 测试稀疏矩阵 BSR 格式的序列化是否有效
    def test_serialization_sparse_bsr_invalid(self):
        # 调用 _test_serialization_sparse_compressed_invalid 方法，传入相关函数和参数
        self._test_serialization_sparse_compressed_invalid(
            lambda x: x.to_sparse_bsr((1, 1)),  # 使用 Torch 的 to_sparse_bsr 方法
            torch.Tensor.crow_indices,         # Torch Tensor 的行偏移索引
            torch.Tensor.col_indices           # Torch Tensor 的列偏移索引
        )
    
    # 测试稀疏矩阵 BSC 格式的序列化是否有效
    def test_serialization_sparse_bsc_invalid(self):
        # 调用 _test_serialization_sparse_compressed_invalid 方法，传入相关函数和参数
        self._test_serialization_sparse_compressed_invalid(
            lambda x: x.to_sparse_bsc((1, 1)),  # 使用 Torch 的 to_sparse_bsc 方法
            torch.Tensor.ccol_indices,         # Torch Tensor 的列偏移索引
            torch.Tensor.row_indices           # Torch Tensor 的行偏移索引
        )
    
    # 测试设备对象的序列化和反序列化是否正确
    def test_serialize_device(self):
        # 定义设备的字符串表示形式列表
        device_str = ['cpu', 'cpu:0', 'cuda', 'cuda:0']
        # 创建相应的 Torch 设备对象列表
        device_obj = [torch.device(d) for d in device_str]
        # 遍历设备对象列表
        for device in device_obj:
            # 对设备对象进行深拷贝
            device_copied = copy.deepcopy(device)
            # 使用断言检查原始设备对象和深拷贝后的设备对象是否相等
            self.assertEqual(device, device_copied)
    # 定义一个测试方法，用于验证反向兼容性的序列化操作，可以选择是否仅保存权重
    def _test_serialization_backwards_compat(self, weights_only):
        # 创建包含浮点数的二维张量列表 a，共两个张量
        a = [torch.arange(1 + i, 26 + i).view(5, 5).float() for i in range(2)]
        # 创建列表 b，其中包含张量 a 中的部分元素和其存储
        b = [a[i % 2] for i in range(4)]
        b += [a[0].storage()]
        b += [a[0].reshape(-1)[1:4].clone().storage()]
        # 下载并加载测试数据的路径
        path = download_file('https://download.pytorch.org/test_data/legacy_serialized.pt')
        # 从路径加载数据，根据 weights_only 参数选择是否仅加载权重
        c = torch.load(path, weights_only=weights_only)
        # 断言 b 和 c 相等，精确度为 0
        self.assertEqual(b, c, atol=0, rtol=0)
        # 断言 c 的前四个元素为 torch.FloatTensor 类型
        self.assertTrue(isinstance(c[0], torch.FloatTensor))
        self.assertTrue(isinstance(c[1], torch.FloatTensor))
        self.assertTrue(isinstance(c[2], torch.FloatTensor))
        self.assertTrue(isinstance(c[3], torch.FloatTensor))
        # 断言 c 的第五个元素为 torch.storage.TypedStorage 类型，数据类型为 torch.float32
        self.assertTrue(isinstance(c[4], torch.storage.TypedStorage))
        self.assertEqual(c[4].dtype, torch.float32)
        # 将 c 的第一个元素填充为 10，并与 c 的第三个元素进行比较
        c[0].fill_(10)
        self.assertEqual(c[0], c[2], atol=0, rtol=0)
        # 断言 c 的第四个元素与创建一个填充为 10 的 torch.FloatStorage 对象相等
        self.assertEqual(c[4], torch.FloatStorage(25).fill_(10), atol=0, rtol=0)
        # 将 c 的第二个元素填充为 20，并与 c 的第四个元素进行比较
        c[1].fill_(20)
        self.assertEqual(c[1], c[3], atol=0, rtol=0)

        # 测试一些旧的张量序列化机制
        # 定义旧张量的基类
        class OldTensorBase:
            def __init__(self, new_tensor):
                self.new_tensor = new_tensor

            def __getstate__(self):
                return (self.new_tensor.storage(),
                        self.new_tensor.storage_offset(),
                        tuple(self.new_tensor.size()),
                        self.new_tensor.stride())

        # 定义旧张量版本1
        class OldTensorV1(OldTensorBase):
            def __reduce__(self):
                return (torch.Tensor, (), self.__getstate__())

        # 定义旧张量版本2
        class OldTensorV2(OldTensorBase):
            def __reduce__(self):
                return (_rebuild_tensor, self.__getstate__())

        # 创建一个步幅为 [9, 3] 的张量 x，并按旧张量类别进行循环测试
        x = torch.randn(30).as_strided([2, 3], [9, 3], 2)
        for old_cls in [OldTensorV1, OldTensorV2]:
            # 使用临时文件保存并加载张量 x，并与原始张量 x 进行比较
            with tempfile.NamedTemporaryFile() as f:
                old_x = old_cls(x)
                torch.save(old_x, f)
                f.seek(0)
                load_x = torch.load(f, weights_only=weights_only)
                self.assertEqual(x.storage(), load_x.storage())
                self.assertEqual(x.storage_offset(), load_x.storage_offset())
                self.assertEqual(x.size(), load_x.size())
                self.assertEqual(x.stride(), load_x.stride())

    # 测试不带权重的反向兼容性序列化操作
    def test_serialization_backwards_compat(self):
        self._test_serialization_backwards_compat(False)

    # 测试带有安全机制的反向兼容性序列化操作
    def test_serialization_backwards_compat_safe(self):
        self._test_serialization_backwards_compat(True)

    # 测试序列化保存时的警告
    def test_serialization_save_warnings(self):
        # 捕获警告信息
        with warnings.catch_warnings(record=True) as warns:
            with tempfile.NamedTemporaryFile() as checkpoint:
                # 将一个线性模型保存到临时文件，并断言没有警告信息产生
                x = torch.save(torch.nn.Linear(2, 3), checkpoint)
                self.assertEqual(len(warns), 0)
    # 定义一个测试方法，用于测试序列化对象的位置映射功能
    def test_serialization_map_location(self):
        # 下载测试文件并获取其路径
        test_file_path = download_file('https://download.pytorch.org/test_data/gpu_tensors.pt')

        # 定义一个简单的位置映射函数，直接返回输入的存储对象
        def map_location(storage, loc):
            return storage

        # 定义生成不同设备类型的位置映射列表的函数
        def generate_map_locations(device_type):
            return [
                {'cuda:0': device_type + ':0'},  # 返回一个映射字典
                device_type,                    # 返回设备类型字符串
                device_type + ':0',             # 返回特定设备字符串
                torch.device(device_type),      # 返回一个 Torch 设备对象
                torch.device(device_type, 0)    # 返回一个带索引的 Torch 设备对象
            ]

        # 定义加载文件内容并返回字节流的函数
        def load_bytes():
            with open(test_file_path, 'rb') as f:
                return io.BytesIO(f.read())

        # 定义文件对象或字节流生成器的列表
        fileobject_lambdas = [lambda: test_file_path, load_bytes]

        # 定义针对 CPU 设备的位置映射列表
        cpu_map_locations = [
            map_location,                   # 使用之前定义的映射函数
            {'cuda:0': 'cpu'},              # 硬编码的 CUDA 到 CPU 映射字典
            'cpu',                          # 字符串形式的 CPU 设备
            torch.device('cpu'),            # Torch CPU 设备对象
        ]

        # 定义针对 GPU 设备的位置映射列表
        gpu_0_map_locations = generate_map_locations('cuda')
        
        # 定义最后一个 GPU 设备的位置映射列表
        gpu_last_map_locations = [
            f'cuda:{torch.cuda.device_count() - 1}',  # 使用字符串格式化计算最后一个 CUDA 设备的索引
        ]

        # 定义针对 XPU 设备的位置映射列表
        xpu_0_map_locations = generate_map_locations('xpu')
        
        # 定义最后一个 XPU 设备的位置映射列表
        xpu_last_map_locations = [
            f'xpu:{torch.xpu.device_count() - 1}',  # 使用字符串格式化计算最后一个 XPU 设备的索引
        ]

        # 定义检查位置映射的函数
        def check_map_locations(map_locations, dtype, intended_device):
            for fileobject_lambda in fileobject_lambdas:
                for map_location in map_locations:
                    # 加载序列化对象，并使用指定的位置映射
                    tensor = torch.load(fileobject_lambda(), map_location=map_location)

                    # 断言张量的设备与预期设备一致
                    self.assertEqual(tensor.device, intended_device)
                    # 断言张量的数据类型与预期类型一致
                    self.assertEqual(tensor.dtype, dtype)
                    # 断言张量的内容与预期的内容一致
                    self.assertEqual(tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, device=intended_device))

        # 对 CPU 设备的位置映射进行测试
        check_map_locations(cpu_map_locations, torch.float, torch.device('cpu'))

        # 如果 CUDA 可用，则对 GPU 设备的位置映射进行测试
        if torch.cuda.is_available():
            check_map_locations(gpu_0_map_locations, torch.float, torch.device('cuda', 0))
            check_map_locations(
                gpu_last_map_locations,
                torch.float,
                torch.device('cuda', torch.cuda.device_count() - 1)
            )

        # 如果 XPU 可用，则对 XPU 设备的位置映射进行测试
        if torch.xpu.is_available():
            check_map_locations(xpu_0_map_locations, torch.float, torch.device('xpu', 0))
            check_map_locations(
                xpu_last_map_locations,
                torch.float,
                torch.device('xpu', torch.xpu.device_count() - 1)
            )

    # 标记测试方法，如果当前环境只支持 CPU，则跳过测试
    @unittest.skipIf(torch.cuda.is_available(), "Testing torch.load on CPU-only machine")
    def test_load_nonexistent_device(self):
        # Setup: create a serialized file object with a 'cuda:0' restore location
        # The following was generated by saving a torch.randn(2, device='cuda') tensor.
        serialized = (b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9'
                      b'\x03.\x80\x02}q\x00(X\x10\x00\x00\x00protocol_versionq'
                      b'\x01M\xe9\x03X\r\x00\x00\x00little_endianq\x02\x88X\n'
                      b'\x00\x00\x00type_sizesq\x03}q\x04(X\x05\x00\x00\x00shortq'
                      b'\x05K\x02X\x03\x00\x00\x00intq\x06K\x04X\x04\x00\x00\x00'
                      b'longq\x07K\x04uu.\x80\x02ctorch._utils\n_rebuild_tensor_v2'
                      b'\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage'
                      b'\nq\x02X\x0e\x00\x00\x0094919395964320q\x03X\x06\x00\x00'
                      b'\x00cuda:0q\x04K\x02Ntq\x05QK\x00K\x02\x85q\x06K\x01\x85q'
                      b'\x07\x89Ntq\x08Rq\t.\x80\x02]q\x00X\x0e\x00\x00\x00'
                      b'94919395964320q\x01a.\x02\x00\x00\x00\x00\x00\x00\x00\xbb'
                      b'\x1f\x82\xbe\xea\x81\xd1>')

        # 将序列化数据装载到字节流缓冲区
        buf = io.BytesIO(serialized)

        # 准备捕获的错误信息，表明尝试在 CUDA 设备上反序列化对象
        error_msg = r'Attempting to deserialize object on a CUDA device'
        
        # 使用断言确保在加载时抛出预期的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, error_msg):
            _ = torch.load(buf)

    @unittest.skipIf((3, 8, 0) <= sys.version_info < (3, 8, 2), "See https://bugs.python.org/issue39681")
    def test_serialization_filelike_api_requirements(self):
        # 创建一个模拟的文件对象，不支持 readinto 方法
        filemock = FilelikeMock(b'', has_readinto=False)
        
        # 创建一个随机张量并将其保存到 filemock 对象中
        tensor = torch.randn(3, 5)
        torch.save(tensor, filemock)
        
        # 期望的方法集合，确保 filemock 对象支持写入和刷新操作
        expected_superset = {'write', 'flush'}
        self.assertTrue(expected_superset.issuperset(filemock.calls))

        # 在保存和加载之间进行重置
        filemock.seek(0)
        filemock.calls.clear()

        # 加载保存的张量，并验证 filemock 对象支持读取、读取行、查找和告知操作
        _ = torch.load(filemock)
        expected_superset = {'read', 'readline', 'seek', 'tell'}
        self.assertTrue(expected_superset.issuperset(filemock.calls))

    def _test_serialization_filelike(self, tensor, mock, desc):
        # 创建一个模拟的文件对象，用于保存张量数据
        f = mock(b'')
        
        # 将张量保存到文件对象 f 中
        torch.save(tensor, f)
        
        # 将文件对象的读取指针移动到开头，并将其内容读取到 data 中
        f.seek(0)
        data = mock(f.read())

        # 准备消息字符串，表明使用模拟文件对象进行文件式序列化
        msg = 'filelike serialization with {}'

        # 加载数据并与原始张量进行比较，确保它们相等
        b = torch.load(data)
        self.assertTrue(torch.equal(tensor, b), msg.format(desc))

    @unittest.skipIf((3, 8, 0) <= sys.version_info < (3, 8, 2), "See https://bugs.python.org/issue39681")
    def test_serialization_filelike_missing_attrs(self):
        # 测试文件类对象缺少属性的边缘情况。
        # Python 的 io 文档建议这些属性应该存在并抛出 io.UnsupportedOperation 异常，但并非总是如此。
        mocks = [
            ('no readinto', lambda x: FilelikeMock(x)),
            ('has readinto', lambda x: FilelikeMock(x, has_readinto=True)),
            ('no fileno', lambda x: FilelikeMock(x, has_fileno=False)),
        ]

        to_serialize = torch.randn(3, 10)
        for desc, mock in mocks:
            # 对于每个模拟情况，执行文件类对象序列化测试
            self._test_serialization_filelike(to_serialize, mock, desc)

    @unittest.skipIf((3, 8, 0) <= sys.version_info < (3, 8, 2), "See https://bugs.python.org/issue39681")
    def test_serialization_filelike_stress(self):
        a = torch.randn(11 * (2 ** 9) + 1, 5 * (2 ** 9))

        # 这个测试应该会多次调用 Python 的 read 方法
        self._test_serialization_filelike(a, lambda x: FilelikeMock(x, has_readinto=False),
                                          'read() stress test')
        self._test_serialization_filelike(a, lambda x: FilelikeMock(x, has_readinto=True),
                                          'readinto() stress test')

    def test_serialization_filelike_uses_readinto(self):
        # 为了最大效率，在读取文件类对象时，确保使用 C API 调用 readinto 而不是 read。
        a = torch.randn(5, 4)

        f = io.BytesIO()
        torch.save(a, f)
        f.seek(0)
        data = FilelikeMock(f.read(), has_readinto=True)

        b = torch.load(data)
        self.assertTrue(data.was_called('readinto'))

    def test_serialization_filelike_exceptions(self):
        # 尝试将数据序列化到不具有 write 方法或有损坏方法的缓冲区，并确保不会导致中止。
        # 参见 https://github.com/pytorch/pytorch/issues/87997
        x = torch.rand(10)
        with self.assertRaises(AttributeError):
            # 尝试将字符串序列化为张量
            torch.save('foo', x)
        x.write = "bar"
        x.flush = "baz"
        with self.assertRaises(TypeError):
            # 尝试使用具有 write 属性的字符串序列化张量
            torch.save('foo', x)
        x.write = str.__add__
        x.flush = str.__mul__
        with self.assertRaises(TypeError):
            # 尝试使用错误的可调用写入属性将字符串序列化为张量
            torch.save('foo', x)
        s_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        s = torch.CharStorage(s_data)
        with self.assertRaises(AttributeError):
            # 尝试将列表序列化为 CharStorage
            torch.save(s_data, s)
        x = torch.randint(10, (3, 3), dtype=torch.float).cpu().numpy()
        with self.assertRaises(AttributeError):
            # 尝试将 ndarray 序列化为 ndarray
            torch.save(x, x)
    def test_serialization_storage_slice(self):
        # 测试序列化和存储切片的功能

        # 预生成的序列化数据
        serialized = (b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9\x03'
                      b'.\x80\x02}q\x00(X\n\x00\x00\x00type_sizesq\x01}q\x02(X\x03'
                      b'\x00\x00\x00intq\x03K\x04X\x05\x00\x00\x00shortq\x04K\x02X'
                      b'\x04\x00\x00\x00longq\x05K\x04uX\x10\x00\x00\x00protocol_versionq'
                      b'\x06M\xe9\x03X\r\x00\x00\x00little_endianq\x07\x88u.\x80\x02'
                      b'(X\x07\x00\x00\x00storageq\x00ctorch\nFloatStorage\nq\x01X\x0e'
                      b'\x00\x00\x0094279043900432q\x02X\x03\x00\x00\x00cpuq\x03K\x02'
                      b'X\x0e\x00\x00\x0094279029750368q\x04K\x00K\x01\x87q\x05tq\x06'
                      b'Q(h\x00h\x01X\x0e\x00\x00\x0094279043900432q\x07h\x03K\x02X'
                      b'\x0e\x00\x00\x0094279029750432q\x08K\x01K\x01\x87q\ttq\nQ'
                      b'\x86q\x0b.\x80\x02]q\x00X\x0e\x00\x00\x0094279043900432q'
                      b'\x01a.\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                      b'\x00\x00\x00\x00')

        # 将序列化数据包装成字节流
        buf = io.BytesIO(serialized)
        # 从字节流中加载数据
        (s1, s2) = torch.load(buf)

        # 断言：验证加载后的第一个切片值为0
        self.assertEqual(s1[0], 0)
        # 断言：验证加载后的第二个切片值为0
        self.assertEqual(s2[0], 0)
        # 断言：验证两个切片的数据指针地址相差4
        self.assertEqual(s1.data_ptr() + 4, s2.data_ptr())

    def test_load_unicode_error_msg(self):
        # 测试加载包含Unicode数据的Pickle文件时的错误消息

        # 下载包含Python 2模块和Unicode数据的文件
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        # 使用ascii编码加载文件应该引发UnicodeDecodeError异常
        self.assertRaises(UnicodeDecodeError, lambda: torch.load(path, encoding='ascii'))

    def test_load_python2_unicode_module(self):
        # 测试加载包含Unicode数据的Pickle文件时的警告消息

        # 下载包含Unicode数据的文件
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        # 使用警告捕获器记录警告消息
        with warnings.catch_warnings(record=True) as w:
            # 验证加载过程不返回None
            self.assertIsNotNone(torch.load(path))

    def test_load_error_msg(self):
        # 测试加载过程中的错误消息

        # 预期的错误消息正则表达式
        expected_err_msg = (".*You can only torch.load from a file that is seekable. " +
                            "Please pre-load the data into a buffer like io.BytesIO and " +
                            "try to load from it instead.")

        # 创建模拟文件对象，删除seek和tell方法
        resource = FilelikeMock(data=b"data")
        delattr(resource, "tell")
        delattr(resource, "seek")
        # 使用assertRaisesRegex断言捕获指定异常和消息
        with self.assertRaisesRegex(AttributeError, expected_err_msg):
            torch.load(resource)
    # 定义测试函数，用于测试在不同设备和数据类型下保存和加载操作的正确性
    def test_save_different_dtype_unallocated(self):
        # 初始化设备列表，至少包含 'cpu'，如果有 GPU 则添加 'cuda'
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        # 定义保存和加载检查函数，用于验证保存和加载的正确性
        def save_load_check(a, b):
            # 使用 BytesIO 创建一个内存文件对象
            with io.BytesIO() as f:
                # 将张量 a 和 b 保存到内存文件中
                torch.save([a, b], f)
                # 将文件指针移动到文件开头
                f.seek(0)
                # 从内存文件中加载数据到 a_loaded 和 b_loaded
                a_loaded, b_loaded = torch.load(f)
            # 断言加载的张量与保存的张量相等
            self.assertEqual(a, a_loaded)
            self.assertEqual(b, b_loaded)

        # 遍历设备和所有的数据类型组合
        for device, dtype in product(devices, all_types_and_complex_and(torch.half,
                                                                        torch.bfloat16, torch.bool)):
            # 创建一个空张量 a，指定设备和数据类型
            a = torch.tensor([], dtype=dtype, device=device)

            # 对于每种其他数据类型，创建一个 TypedStorage 对象 s，并进行保存和加载检查
            for other_dtype in all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool):
                s = torch.TypedStorage(
                    wrap_storage=a.storage().untyped(),
                    dtype=other_dtype)
                save_load_check(a, s)
                save_load_check(a.storage(), s)
                # 创建一个空张量 b，指定设备和当前的数据类型，进行保存和加载检查
                b = torch.tensor([], dtype=other_dtype, device=device)
                save_load_check(a, b)

    # 测试在不同数据类型下保存操作会触发特定的运行时错误
    def test_save_different_dtype_error(self):
        # 定义错误消息，用于断言特定的运行时错误
        error_msg = r"Cannot save multiple tensors or storages that view the same data as different types"

        # 初始化设备列表，至少包含 'cpu'，如果有 GPU 则添加 'cuda'
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        # 遍历设备列表
        for device in devices:
            # 创建一个复数张量 a，指定设备和数据类型
            a = torch.randn(10, dtype=torch.complex128, device=device)
            # 创建一个 BytesIO 对象 f
            f = io.BytesIO()

            # 使用 assertRaisesRegex 断言保存多个张量或存储时会触发错误消息
            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a, a.imag], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a.storage(), a.imag], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a, a.imag.storage()], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a.storage(), a.imag.storage()], f)

            # 创建一个张量 a，指定设备，并转换为 TypedStorage 对象 s_bytes
            a = torch.randn(10, device=device)
            s_bytes = torch.TypedStorage(
                wrap_storage=a.storage().untyped(),
                dtype=torch.uint8)

            # 使用 assertRaisesRegex 断言保存多个张量或存储时会触发错误消息
            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a, s_bytes], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a.storage(), s_bytes], f)

    # 测试基本数据类型的安全加载操作
    def test_safe_load_basic_types(self):
        # 使用临时文件创建一个文件对象 f
        with tempfile.NamedTemporaryFile() as f:
            # 定义要保存的数据字典
            data = {"int": 123, "str": "world", "float": 3.14, "bool": False}
            # 将数据保存到临时文件中
            torch.save(data, f)
            # 将文件指针移动到文件开头
            f.seek(0)
            # 从临时文件中加载数据，仅加载权重数据，并进行断言验证
            loaded_data = torch.load(f, weights_only=True)
            self.assertEqual(data, loaded_data)
class serialization_method:
    # 定义一个序列化方法的类
    def __init__(self, use_zip):
        # 初始化方法，接受一个参数 use_zip，表示是否使用 ZIP 文件格式
        self.use_zip = use_zip
        self.torch_save = torch.save  # 保存 torch.save 方法的引用

    def __enter__(self, *args, **kwargs):
        # 进入上下文时执行的方法
        def wrapper(*args, **kwargs):
            # 包装器函数，用于拦截 torch.save 的调用并修改其行为
            if '_use_new_zipfile_serialization' in kwargs:
                # 如果用户尝试手动设置 '_use_new_zipfile_serialization' 参数，抛出异常
                raise RuntimeError("Cannot set method manually")
            kwargs['_use_new_zipfile_serialization'] = self.use_zip
            return self.torch_save(*args, **kwargs)  # 调用原始的 torch.save 方法

        torch.save = wrapper  # 替换 torch.save 方法为 wrapper 函数

    def __exit__(self, *args, **kwargs):
        # 离开上下文时执行的方法，恢复原始的 torch.save 方法
        torch.save = self.torch_save

# 定义一个命名元组 Point，包含 x 和 y 两个字段
Point = namedtuple('Point', ['x', 'y'])

class ClassThatUsesBuildInstruction:
    # 包含使用 BUILD 指令的类示例
    def __init__(self, num):
        self.num = num

    def __reduce_ex__(self, proto):
        # 定义 __reduce_ex__ 方法以支持 pickle 序列化，返回类名、初始化参数和状态字典
        # 在这里设置状态字典，会触发 pickle 使用 BUILD 指令
        return ClassThatUsesBuildInstruction, (self.num,), {'foo': 'bar'}


@unittest.skipIf(IS_WINDOWS, "NamedTemporaryFile on windows")
class TestBothSerialization(TestCase):
    # 测试两种序列化方式的兼容性
    @parametrize("weights_only", (True, False))
    def test_serialization_new_format_old_format_compat(self, device, weights_only):
        # 测试函数，比较新旧格式的序列化结果
        x = [torch.ones(200, 200, device=device) for i in range(30)]

        def test(f_new, f_old):
            # 内部测试函数，分别使用新旧格式保存和加载数据
            torch.save(x, f_new, _use_new_zipfile_serialization=True)
            f_new.seek(0)
            x_new_load = torch.load(f_new, weights_only=weights_only)
            self.assertEqual(x, x_new_load)

            torch.save(x, f_old, _use_new_zipfile_serialization=False)
            f_old.seek(0)
            x_old_load = torch.load(f_old, weights_only=weights_only)
            self.assertEqual(x_old_load, x_new_load)

        # 使用 AlwaysWarnTypedStorageRemoval 上下文管理器和捕获警告
        with AlwaysWarnTypedStorageRemoval(True), warnings.catch_warnings(record=True) as w:
            with tempfile.NamedTemporaryFile() as f_new, tempfile.NamedTemporaryFile() as f_old:
                test(f_new, f_old)
            self.assertTrue(len(w) == 0, msg=f"Expected no warnings but got {[str(x) for x in w]}")


class TestOldSerialization(TestCase, SerializationMixin):
    # 测试旧版序列化的类，混合 SerializationMixin
    # 在 Python 2.7 上，如果警告模块传递的警告与已有的相同，则不会再次引发该警告。
    # 定义一个测试函数，用于测试序列化容器的功能
    def _test_serialization_container(self, unique_key, filecontext_lambda):

        # 创建临时模块名称，确保唯一性
        tmpmodule_name = f'tmpmodule{unique_key}'

        # 定义导入模块的函数
        def import_module(name, filename):
            import importlib.util
            # 根据文件路径创建模块的规范
            spec = importlib.util.spec_from_file_location(name, filename)
            # 根据规范创建模块对象
            module = importlib.util.module_from_spec(spec)
            # 执行模块对象的代码
            spec.loader.exec_module(module)
            # 将模块添加到系统模块列表中
            sys.modules[module.__name__] = module
            return module

        # 使用文件上下文管理器创建检查点
        with filecontext_lambda() as checkpoint:
            # 获取网络1的文件路径
            fname = get_file_path_2(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'testing',
                                    '_internal', 'data', 'network1.py')
            # 导入指定文件作为临时模块
            module = import_module(tmpmodule_name, fname)
            # 将网络结构保存到检查点
            torch.save(module.Net(), checkpoint)

            # 验证检查点可以加载且不会出现有关不安全加载的警告
            checkpoint.seek(0)
            with warnings.catch_warnings(record=True) as w:
                loaded = torch.load(checkpoint)
                self.assertTrue(isinstance(loaded, module.Net))
                # 如果可以检索源代码，验证警告信息
                if can_retrieve_source:
                    self.assertEqual(len(w), 1)
                    self.assertEqual(w[0].category, FutureWarning)
                    self.assertTrue("You are using `torch.load` with `weights_only=False`" in str(w[0].message))

            # 替换模块为不同的源代码
            fname = get_file_path_2(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'testing',
                                    '_internal', 'data', 'network2.py')
            # 导入第二个网络结构的文件作为临时模块
            module = import_module(tmpmodule_name, fname)
            # 重新定位检查点的位置
            checkpoint.seek(0)
            with warnings.catch_warnings(record=True) as w:
                loaded = torch.load(checkpoint)
                self.assertTrue(isinstance(loaded, module.Net))
                # 如果可以检索源代码，验证警告信息
                if can_retrieve_source:
                    self.assertEqual(len(w), 2)
                    self.assertEqual(w[0].category, FutureWarning)
                    self.assertEqual(w[1].category, SourceChangeWarning)

    # 测试序列化容器功能的方法，调用 _test_serialization_container 进行测试
    def test_serialization_container(self):
        self._test_serialization_container('file', tempfile.NamedTemporaryFile)

    # 测试文件样式序列化容器功能的方法，调用 _test_serialization_container 进行测试
    def test_serialization_container_filelike(self):
        self._test_serialization_container('filelike', BytesIOContext)
    # 定义一个测试方法，用于测试序列化偏移量的功能
    def test_serialization_offset(self):
        # 创建一个大小为 5x5 的随机张量 a
        a = torch.randn(5, 5)
        # 创建一个大小为 1024x1024x512 的随机张量 b，数据类型为 float32
        b = torch.randn(1024, 1024, 512, dtype=torch.float32)
        # 创建一个输入通道为 1，输出通道为 1，卷积核大小为 (1, 3) 的卷积层 m
        m = torch.nn.Conv2d(1, 1, (1, 3))
        # 定义整数变量 i 和 j，分别赋值为 41 和 43
        i, j = 41, 43
        # 使用临时命名文件对象 f 进行处理
        with tempfile.NamedTemporaryFile() as f:
            # 将变量 i 序列化并写入文件 f
            pickle.dump(i, f)
            # 将张量 a 保存到文件 f
            torch.save(a, f)
            # 将变量 j 序列化并写入文件 f
            pickle.dump(j, f)
            # 将张量 b 保存到文件 f
            torch.save(b, f)
            # 将卷积层 m 保存到文件 f
            torch.save(m, f)
            # 断言文件指针当前位置大于 2GB
            self.assertTrue(f.tell() > 2 * 1024 * 1024 * 1024)
            # 将文件指针移到文件开头
            f.seek(0)
            # 从文件 f 中加载并反序列化变量 i
            i_loaded = pickle.load(f)
            # 从文件 f 中加载张量 a
            a_loaded = torch.load(f)
            # 从文件 f 中加载并反序列化变量 j
            j_loaded = pickle.load(f)
            # 从文件 f 中加载张量 b
            b_loaded = torch.load(f)
            # 从文件 f 中加载卷积层 m
            m_loaded = torch.load(f)
        # 断言张量 a 和加载后的张量 a 相等
        self.assertTrue(torch.equal(a, a_loaded))
        # 断言张量 b 和加载后的张量 b 相等
        self.assertTrue(torch.equal(b, b_loaded))
        # 断言加载后的卷积层 m 和原始卷积层 m 的核大小相等
        self.assertTrue(m.kernel_size == m_loaded.kernel_size)
        # 断言变量 i 和加载后的变量 i 相等
        self.assertEqual(i, i_loaded)
        # 断言变量 j 和加载后的变量 j 相等
        self.assertEqual(j, j_loaded)

    # 使用参数化装饰器定义另一个测试方法，用于测试文件类似对象上的序列化偏移量
    @parametrize('weights_only', (True, False))
    def test_serialization_offset_filelike(self, weights_only):
        # 创建一个大小为 5x5 的随机张量 a
        a = torch.randn(5, 5)
        # 创建一个大小为 1024x1024x512 的随机张量 b，数据类型为 float32
        b = torch.randn(1024, 1024, 512, dtype=torch.float32)
        # 定义整数变量 i 和 j，分别赋值为 41 和 43
        i, j = 41, 43
        # 使用 BytesIOContext 上下文管理器创建一个字节流对象 f
        with BytesIOContext() as f:
            # 将变量 i 序列化并写入字节流 f
            pickle.dump(i, f)
            # 将张量 a 保存到字节流 f
            torch.save(a, f)
            # 将变量 j 序列化并写入字节流 f
            pickle.dump(j, f)
            # 将张量 b 保存到字节流 f
            torch.save(b, f)
            # 断言字节流 f 的当前位置大于 2GB
            self.assertTrue(f.tell() > 2 * 1024 * 1024 * 1024)
            # 将字节流 f 的读写位置移动到开头
            f.seek(0)
            # 从字节流 f 中加载并反序列化变量 i
            i_loaded = pickle.load(f)
            # 从字节流 f 中加载张量 a，根据 weights_only 参数决定是否仅加载权重
            a_loaded = torch.load(f, weights_only=weights_only)
            # 从字节流 f 中加载并反序列化变量 j
            j_loaded = pickle.load(f)
            # 从字节流 f 中加载张量 b，根据 weights_only 参数决定是否仅加载权重
            b_loaded = torch.load(f, weights_only=weights_only)
        # 断言张量 a 和加载后的张量 a 相等
        self.assertTrue(torch.equal(a, a_loaded))
        # 断言张量 b 和加载后的张量 b 相等
        self.assertEqual(b, b_loaded)
        # 断言变量 i 和加载后的变量 i 相等
        self.assertEqual(i, i_loaded)
        # 断言变量 j 和加载后的变量 j 相等
        self.assertEqual(j, j_loaded)

    # 重写 run 方法以运行父类的 run 方法，同时指定序列化方法不使用 ZIP 格式
    def run(self, *args, **kwargs):
        # 使用 serialization_method 上下文管理器禁用 ZIP 格式
        with serialization_method(use_zip=False):
            # 调用父类的 run 方法并传入参数和关键字参数
            return super().run(*args, **kwargs)
# 测试类 TestSerialization 继承自 TestCase 和 SerializationMixin
class TestSerialization(TestCase, SerializationMixin):

    # 使用参数化装饰器 parametrize 来执行两次测试，分别测试 weights_only 为 True 和 False 的情况
    @parametrize('weights_only', (True, False))
    def test_serialization_zipfile(self, weights_only):
        # 获取测试序列化数据
        data = self._test_serialization_data()

        # 定义内部测试函数 test，接受一个文件名或文件对象作为参数
        def test(name_or_buffer):
            # 使用 Torch 的 save 方法将数据保存到指定的文件名或缓冲区中
            torch.save(data, name_or_buffer)

            # 如果 name_or_buffer 具有 seek 方法（即是文件对象），则将其指针移动到文件开头
            if hasattr(name_or_buffer, 'seek'):
                name_or_buffer.seek(0)

            # 使用 Torch 的 load 方法从文件名或缓冲区中加载数据，根据 weights_only 参数选择是否仅加载权重
            result = torch.load(name_or_buffer, weights_only=weights_only)
            
            # 使用 TestCase 中的 assertEqual 方法断言加载的结果与原始数据相等
            self.assertEqual(result, data)

        # 使用临时命名文件进行测试
        with tempfile.NamedTemporaryFile() as f:
            test(f)

        # 使用临时文件名进行测试
        with TemporaryFileName() as fname:
            test(fname)

        # 如果文件系统支持 UTF-8 编码，则使用带有特定后缀的临时目录名和文件名进行测试
        if IS_FILESYSTEM_UTF8_ENCODING:
            with TemporaryDirectoryName(suffix='\u975eASCII\u30d1\u30b9') as dname:
                with TemporaryFileName(dir=dname) as fname:
                    test(fname)

        # 使用 BytesIO 进行测试
        test(io.BytesIO())

    # 测试函数，确保大型 zip64 序列化正常工作
    @serialTest()
    def test_serialization_2gb_file(self):
        # 执行垃圾回收以尽可能释放内存
        gc.collect()
        # 创建一个大型的 Conv2d 模型
        big_model = torch.nn.Conv2d(20000, 3200, kernel_size=3)

        # 使用 BytesIOContext 作为文件对象进行测试
        with BytesIOContext() as f:
            # 使用 Torch 的 save 方法将大模型的状态字典保存到文件对象中
            torch.save(big_model.state_dict(), f)
            f.seek(0)
            # 使用 Torch 的 load 方法加载文件对象中的数据
            state = torch.load(f)

    # 使用参数化装饰器 parametrize 来执行两次测试，分别测试 weights_only 为 True 和 False 的情况
    @parametrize('weights_only', (True, False))
    def test_pathlike_serialization(self, weights_only):
        # 创建一个 Conv2d 模型
        model = torch.nn.Conv2d(20, 3200, kernel_size=3)

        # 使用临时文件名进行测试
        with TemporaryFileName() as fname:
            # 创建路径对象 Path
            path = Path(fname)
            # 使用 Torch 的 save 方法将模型的状态字典保存到路径指定的文件中
            torch.save(model.state_dict(), path)
            # 使用 Torch 的 load 方法从路径加载数据，根据 weights_only 参数选择是否仅加载权重
            torch.load(path, weights_only=weights_only)

    # 使用参数化装饰器 parametrize 来执行两次测试，分别测试 weights_only 为 True 和 False 的情况
    def test_meta_serialization(self, weights_only):
        # 创建一个带有元数据的大型 Conv2d 模型
        big_model = torch.nn.Conv2d(20000, 320000, kernel_size=3, device='meta')

        # 使用 BytesIOContext 作为文件对象进行测试
        with BytesIOContext() as f:
            # 使用 Torch 的 save 方法将大模型的状态字典保存到文件对象中
            torch.save(big_model.state_dict(), f)
            f.seek(0)
            # 使用 Torch 的 load 方法从文件对象加载数据，根据 weights_only 参数选择是否仅加载权重
            state = torch.load(f, weights_only=weights_only)

        # 使用 TestCase 中的 assertEqual 方法断言加载的状态字典中权重的大小与大模型的权重大小相等
        self.assertEqual(state['weight'].size(), big_model.weight.size())
    # 定义测试方法，用于测试学习率调度器的序列化功能
    def test_lr_scheduler_serialization(self):
        # 创建一个随机张量并指定需要梯度计算
        sgd = torch.optim.SGD([
            torch.tensor(torch.randn(100, 100, 2000), requires_grad=True)
        ], lr=0.1, momentum=0.9)
        # 创建一个OneCycleLR类型的学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(sgd, 6.0, total_steps=10)

        # 使用BytesIOContext上下文管理器打开一个字节流文件
        with BytesIOContext() as f:
            # 将学习率调度器的状态字典保存到文件流中
            torch.save(lr_scheduler.state_dict(), f)
            # 定位文件流到末尾并获取文件大小
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # 将文件流指针重置到文件开头
            f.seek(0)
            # 从文件流中加载学习率调度器的状态
            lr_scheduler_state = torch.load(f)

        # 断言保存的状态字典中的'base_lrs'与原始学习率调度器的一致
        self.assertEqual(lr_scheduler_state['base_lrs'], lr_scheduler.base_lrs)
        # 检查是否存在'anneal_func'字段，若存在则断言其不绑定到特定对象
        if 'anneal_func' in lr_scheduler_state:
            self.assertFalse(hasattr(lr_scheduler_state['anneal_func'], '__self__'))  # check method is not bound
        else:
            self.assertTrue('_anneal_func_type' in lr_scheduler_state)
        # 断言文件大小小于1MB
        self.assertTrue(size < 1024 * 1024)  # Must be less than 1MB

    # 参数化测试方法，用于测试Python属性的序列化
    @parametrize('weights_only', (True, False))
    def test_serialization_python_attr(self, weights_only):
        # 内部测试方法，用于测试对象属性的保存和加载
        def _test_save_load_attr(t):
            t.foo = 'foo'
            t.pi = 3.14

            # 使用BytesIOContext上下文管理器打开一个字节流文件
            with BytesIOContext() as f:
                # 将对象t保存到文件流中
                torch.save(t, f)
                # 将文件流指针重置到文件开头
                f.seek(0)
                # 从文件流中加载对象，根据weights_only参数决定是否仅加载权重
                loaded_t = torch.load(f, weights_only=weights_only)

            # 断言加载的对象与原始对象t相等
            self.assertEqual(t, loaded_t)
            # 断言加载的对象属性与原始对象的属性相等
            self.assertEqual(t.foo, loaded_t.foo)
            self.assertEqual(t.pi, loaded_t.pi)

        # 创建一个3x3的零张量t，并进行属性序列化测试
        t = torch.zeros(3, 3)
        _test_save_load_attr(t)
        # 将t转换为参数对象并进行属性序列化测试
        _test_save_load_attr(torch.nn.Parameter(t))

    # 测试方法，用于测试仅权重加载时的断言行为
    def test_weights_only_assert(self):
        # 定义一个HelloWorld类，实现__reduce__方法返回print函数和参数元组
        class HelloWorld:
            def __reduce__(self):
                return (print, ("Hello World!",))

        # 使用BytesIOContext上下文管理器打开一个字节流文件
        with BytesIOContext() as f:
            # 将HelloWorld实例保存到文件流中
            torch.save(HelloWorld(), f)
            # 将文件流指针重置到文件开头
            f.seek(0)
            # 尝试使用不安全模式加载，预期返回None
            self.assertIsNone(torch.load(f, weights_only=False))
            # 将文件流指针重置到文件开头
            f.seek(0)
            # 尝试使用安全模式加载，预期引发UnpicklingError异常
            with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported global: GLOBAL builtins.print"):
                torch.load(f, weights_only=True)
            try:
                # 添加print函数到安全全局白名单，将文件流指针重置到文件开头
                torch.serialization.add_safe_globals([print])
                f.seek(0)
                # 使用安全模式加载，应成功加载
                torch.load(f, weights_only=True)
            finally:
                # 清除安全全局白名单
                torch.serialization.clear_safe_globals()
    # 定义测试函数，测试在安全全局对象的情况下使用 weights_only 选项加载数据

        # 创建一个 Point 对象 p，并使用 NEWOBJ 指令进行序列化
        p = Point(x=1, y=2)
        # 使用 BytesIOContext 上下文管理器创建字节流 f
        with BytesIOContext() as f:
            # 将 p 对象保存到 f 中
            torch.save(p, f)
            # 将文件指针移动到文件开头
            f.seek(0)
            # 断言捕获到 UnpicklingError 异常，验证 GLOBAL __main__.Point 默认情况下不是允许的全局对象
            with self.assertRaisesRegex(pickle.UnpicklingError,
                                        "GLOBAL __main__.Point was not an allowed global by default"):
                # 使用 weights_only=True 选项加载 f 中的数据
                torch.load(f, weights_only=True)
            # 将文件指针移动到文件开头
            f.seek(0)
            try:
                # 将 Point 类添加到安全全局对象列表中
                torch.serialization.add_safe_globals([Point])
                # 使用 weights_only=True 选项加载 f 中的数据
                loaded_p = torch.load(f, weights_only=True)
                # 断言加载的数据与原始数据 p 相等
                self.assertEqual(loaded_p, p)
            finally:
                # 清除安全全局对象列表中的内容
                torch.serialization.clear_safe_globals()

    # 定义测试函数，测试在安全全局对象的情况下使用 build 指令加载数据
    def test_weights_only_safe_globals_build(self):
        # 计数器初始化为 0
        counter = 0

        # 定义 fake_set_state 函数，用于模拟 setstate 方法
        def fake_set_state(obj, *args):
            nonlocal counter
            counter += 1

        # 创建一个 ClassThatUsesBuildInstruction 的实例 c
        c = ClassThatUsesBuildInstruction(2)
        # 使用 BytesIOContext 上下文管理器创建字节流 f
        with BytesIOContext() as f:
            # 将 c 对象保存到 f 中
            torch.save(c, f)
            # 将文件指针移动到文件开头
            f.seek(0)
            # 断言捕获到 UnpicklingError 异常，验证 GLOBAL __main__.ClassThatUsesBuildInstruction 默认情况下不是允许的全局对象
            with self.assertRaisesRegex(pickle.UnpicklingError,
                                        "GLOBAL __main__.ClassThatUsesBuildInstruction was not an allowed global by default"):
                # 使用 weights_only=True 选项加载 f 中的数据
                torch.load(f, weights_only=True)
            try:
                # 将 ClassThatUsesBuildInstruction 类添加到安全全局对象列表中
                torch.serialization.add_safe_globals([ClassThatUsesBuildInstruction])
                # 将文件指针移动到文件开头
                f.seek(0)
                # 使用 weights_only=True 选项加载 f 中的数据
                loaded_c = torch.load(f, weights_only=True)
                # 断言 loaded_c 对象的属性 num 等于 2
                self.assertEqual(loaded_c.num, 2)
                # 断言 loaded_c 对象的属性 foo 等于 'bar'
                self.assertEqual(loaded_c.foo, 'bar')
                # 将 fake_set_state 函数赋值给 ClassThatUsesBuildInstruction 的 __setstate__ 方法
                ClassThatUsesBuildInstruction.__setstate__ = fake_set_state
                # 将文件指针移动到文件开头
                f.seek(0)
                # 使用 weights_only=True 选项加载 f 中的数据
                loaded_c = torch.load(f, weights_only=True)
                # 断言 loaded_c 对象的属性 num 等于 2
                self.assertEqual(loaded_c.num, 2)
                # 断言 counter 的值为 1
                self.assertEqual(counter, 1)
                # 断言 loaded_c 对象不包含属性 'foo'
                self.assertFalse(hasattr(loaded_c, 'foo'))
            finally:
                # 清除安全全局对象列表中的内容
                torch.serialization.clear_safe_globals()
                # 将 ClassThatUsesBuildInstruction 的 __setstate__ 方法恢复为 None

    # 使用参数化装饰器定义测试函数，测试在不同 unsafe_global 参数下使用 weights_only 选项加载数据
    @parametrize("unsafe_global", [True, False])
    def test_weights_only_error(self, unsafe_global):
        # 创建一个包含 TwoTensor 实例的字典 sd
        sd = {'t': TwoTensor(torch.randn(2), torch.randn(2))}
        # 根据 unsafe_global 参数选择序列化协议
        pickle_protocol = torch.serialization.DEFAULT_PROTOCOL if unsafe_global else 5
        # 使用 BytesIOContext 上下文管理器创建字节流 f
        with BytesIOContext() as f:
            # 将 sd 字典保存到 f 中，使用指定的 pickle_protocol
            torch.save(sd, f, pickle_protocol=pickle_protocol)
            # 将文件指针移动到文件开头
            f.seek(0)
            if unsafe_global:
                # 断言捕获到 UnpicklingError 异常，提示使用 torch.serialization.add_safe_globals([TwoTensor]) 方法进行允许列表设置
                with self.assertRaisesRegex(pickle.UnpicklingError,
                                            r"use `torch.serialization.add_safe_globals\(\[TwoTensor\]\)` to allowlist"):
                    # 使用 weights_only=True 选项加载 f 中的数据
                    torch.load(f, weights_only=True)
            else:
                # 断言捕获到 UnpicklingError 异常，提示文件中包含的对象需要反馈以支持 `weights_only=True` 选项
                with self.assertRaisesRegex(pickle.UnpicklingError,
                                            "file an issue with the following so that we can make `weights_only=True`"):
                    # 使用 weights_only=True 选项加载 f 中的数据
                    torch.load(f, weights_only=True)
    # 参数化测试，测试函数会被多次调用，每次传入不同的 weights_only 参数值（False 和 True）
    @parametrize('weights_only', (False, True))
    def test_serialization_math_bits(self, weights_only):
        # 生成一个随机复数张量 t
        t = torch.randn(1, dtype=torch.cfloat)

        # 定义内部函数 _save_load_check，用于保存和加载检查
        def _save_load_check(t):
            # 使用 BytesIOContext 创建一个文件对象 f
            with BytesIOContext() as f:
                # 将张量 t 保存到文件对象 f 中
                torch.save(t, f)
                f.seek(0)
                # 使用 unsafe load 加载张量，并断言加载的结果与原始张量 t 相等
                self.assertEqual(torch.load(f, weights_only=weights_only), t)

        # 对张量 t 进行共轭操作得到 t_conj，并调用 _save_load_check 进行保存加载检查
        t_conj = torch.conj(t)
        _save_load_check(t_conj)

        # 对张量 t 进行取负视图操作得到 t_neg，并调用 _save_load_check 进行保存加载检查
        t_neg = torch._neg_view(t)
        _save_load_check(t_neg)

        # 对张量 t 进行共轭和取负视图操作得到 t_n_c，并调用 _save_load_check 进行保存加载检查
        t_n_c = torch._neg_view(torch.conj(t))
        _save_load_check(t_n_c)

    # 参数化测试，测试函数会被多次调用，每次传入不同的 weights_only 参数值（False 和 True）
    @parametrize('weights_only', (False, True))
    def test_serialization_efficient_zerotensor(self, weights_only):
        # 创建一个零张量，用于测试，目前不支持序列化 ZeroTensor，因为它还未对外公开
        t = torch._efficientzerotensor((4, 5))

        # 定义内部函数 _save_load_check，用于保存和加载检查
        def _save_load_check(t):
            # 使用 BytesIOContext 创建一个文件对象 f
            with BytesIOContext() as f:
                # 将张量 t 保存到文件对象 f 中
                torch.save(t, f)
                f.seek(0)
                # 使用 unsafe load 加载张量，并断言加载的结果与原始张量 t 相等
                self.assertEqual(torch.load(f, weights_only=weights_only), t)

        # 由于 ZeroTensor 不支持序列化，预期 _save_load_check(t) 会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, 'ZeroTensor is not serializable'):
            _save_load_check(t)

    # 测试序列化和反序列化字节顺序标记
    def test_serialization_byteorder_mark(self):
        # 创建一个 LSTM 模型和输入数据
        lstm = torch.nn.LSTM(3, 3)
        inputs = [torch.randn(1, 3) for _ in range(5)]
        inputs = torch.cat(inputs).view(len(inputs), 1, -1)
        hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # 清空隐藏状态

        # 创建一个字节流对象 databuffer
        databuffer = io.BytesIO()
        # 将 LSTM 模型的状态字典保存到 databuffer 中
        torch.save(lstm.state_dict(), databuffer)
        databuffer.seek(0)

        # 使用 torch.serialization._open_zipfile_reader 打开 databuffer 的 ZIP 文件阅读器
        with torch.serialization._open_zipfile_reader(databuffer) as zip_file:
            byteordername = 'byteorder'
            # 断言 ZIP 文件中存在名为 byteorder 的记录
            self.assertTrue(zip_file.has_record(byteordername))
            # 获取名为 byteorder 的记录数据
            byteorderdata = zip_file.get_record(byteordername)
            # 断言 byteorderdata 数据是 'little' 或 'big' 中的一个
            self.assertTrue(byteorderdata in [b'little', b'big'])
            # 断言 byteorderdata 解码后与系统的字节顺序相同
            self.assertEqual(byteorderdata.decode(), sys.byteorder)

    # 跳过测试，仅在平台为 s390x 时执行，同时参数化 path_type 和 weights_only
    @unittest.skipIf(platform.machine() != 's390x', "s390x-specific test")
    @parametrize('path_type', (str, Path))
    @parametrize('weights_only', (True, False))
    @unittest.skipIf(IS_WINDOWS, "NamedTemporaryFile on windows")
    # 定义一个测试函数，用于测试使用 mmap 加载的序列化数据
    def test_serialization_mmap_loading(self, weights_only, path_type):
        # 定义一个简单的神经网络模型
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(3, 1024)
                self.fc2 = torch.nn.Linear(1024, 5)

            def forward(self, input):
                return self.fc2(self.fc1(input))

        # 使用临时文件名上下文管理器创建一个临时文件
        with TemporaryFileName() as f:
            # 将临时文件名转换为指定类型的路径
            f = path_type(f)
            # 创建 DummyModel 的状态字典并保存到临时文件
            state_dict = DummyModel().state_dict()
            torch.save(state_dict, f)
            # 使用 mmap 方式加载模型状态字典
            result = torch.load(f, mmap=True, weights_only=weights_only)
            # 不使用 mmap 方式加载模型状态字典
            result_non_mmap = torch.load(f, mmap=False, weights_only=weights_only)

        # 创建使用 mmap 加载的模型并加载状态字典
        model_mmap_state_dict = DummyModel()
        model_mmap_state_dict.load_state_dict(result)
        # 创建不使用 mmap 加载的模型并加载状态字典
        model_non_mmap_state_dict = DummyModel()
        model_non_mmap_state_dict.load_state_dict(result_non_mmap)
        # 创建输入张量
        input = torch.randn(4, 3)
        # 断言使用 mmap 加载的模型和不使用 mmap 加载的模型的输出相同
        self.assertEqual(model_mmap_state_dict(input), model_non_mmap_state_dict(input.clone()))

    # 如果 CUDA 不可用或者运行环境是 Windows，则跳过该测试
    @unittest.skipIf(not torch.cuda.is_available() or IS_WINDOWS,
                     "CUDA is unavailable or NamedTemporaryFile on Windows")
    # 定义测试函数，测试带有 map_location 参数的 mmap 加载序列化数据
    def test_serialization_mmap_loading_with_map_location(self):
        # 定义一个简单的神经网络模型
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(3, 1024)
                self.fc2 = torch.nn.Linear(1024, 5)

            def forward(self, input):
                return self.fc2(self.fc1(input))

        # 使用临时文件名上下文管理器创建一个临时文件
        with TemporaryFileName() as f:
            # 在 CUDA 设备上创建一个 DummyModel 实例
            with torch.device('cuda'):
                m = DummyModel()
            # 获取模型的状态字典并保存到临时文件
            state_dict = m.state_dict()
            torch.save(state_dict, f)
            # 使用 mmap 方式加载模型状态字典
            result = torch.load(f, mmap=True)
            # 断言加载的每个张量都位于 CUDA 设备上
            for v in result.values():
                self.assertTrue(v.is_cuda)
    # 定义测试方法，用于测试序列化 mmap 加载选项
    def test_serialization_mmap_loading_options(self):
        # 如果运行环境是 Windows
        if IS_WINDOWS:
            # 断言抛出 RuntimeError，并验证错误消息
            with self.assertRaisesRegex(RuntimeError, "Changing the default mmap options is currently not supported"):
                torch.serialization.set_default_mmap_options(2)
            # 返回结束测试
            return
        
        # 创建一个线性模型
        m = torch.nn.Linear(3, 5)
        # 获取模型的状态字典
        sd = m.state_dict()
        
        # 使用临时文件保存模型状态字典
        with tempfile.NamedTemporaryFile() as f:
            torch.save(sd, f)
            
            # 加载保存的模型状态字典，启用 mmap 模式
            sd_loaded = torch.load(f.name, mmap=True)
            # 修改加载的模型状态字典中的权重数据
            sd_loaded['weight'][0][0] = 0
            # 再次加载相同文件，验证未修改的数据是否保持不变
            sd_loaded2 = torch.load(f.name, mmap=True)
            self.assertEqual(sd_loaded2['weight'], sd['weight'])
            
            # 设置默认 mmap 选项为 MAP_SHARED，允许修改文件
            torch.serialization.set_default_mmap_options(MAP_SHARED)
            try:
                # 重新加载文件，验证是否可以修改加载的数据
                sd_loaded = torch.load(f.name, mmap=True)
                sd_loaded['weight'][0][0] = 0
                sd_loaded2 = torch.load(f.name, mmap=True)
                # 验证修改后的权重数据是否已更新
                self.assertNotEqual(sd_loaded2['weight'], sd['weight'])
                self.assertEqual(sd_loaded2['weight'][0][0].item(), 0)
                self.assertEqual(sd_loaded2['weight'], sd_loaded['weight'])
            finally:
                # 最终恢复默认 mmap 选项为 MAP_PRIVATE
                torch.serialization.set_default_mmap_options(MAP_PRIVATE)

    # 参数化测试，测试不同的数据类型和仅权重的加载选项
    @parametrize('dtype', (torch.float8_e5m2, torch.float8_e4m3fn, torch.complex32))
    @parametrize('weights_only', (True, False))
    def test_serialization_dtype(self, dtype, weights_only):
        """ Tests that newer dtypes can be serialized using `_rebuild_tensor_v3` """
        # 使用临时文件进行测试
        with tempfile.NamedTemporaryFile() as f:
            # 创建一个张量 x，并将其序列化保存到文件中
            x = torch.arange(0.0, 100.0).to(dtype=dtype)
            torch.save({'x': x, 'even': x[0::2], 'odd': x[1::2]}, f)
            f.seek(0)
            # 从文件中加载数据，根据 weights_only 参数选择是否仅加载权重
            y = torch.load(f, weights_only=weights_only)
            self.assertEqual(y['x'], x)
            
            # 验证加载的 odd 和 even 张量是否为视图
            y['odd'][0] = torch.tensor(0.25, dtype=dtype)
            y['even'][0] = torch.tensor(-0.25, dtype=dtype)
            self.assertEqual(y['x'][:2].to(dtype=torch.float32), torch.tensor([-0.25, 0.25]))

    # 参数化测试，测试文件名的跳过装饰器条件
    @parametrize('filename', (True, False))
    @unittest.skipIf(IS_WINDOWS, "NamedTemporaryFile on windows")
    @unittest.skipIf(IS_FBCODE, "miniz version differs between fbcode and oss")
    def test_filewriter_metadata_writing(self, filename):
        # 获取线性模型的状态字典
        sd = torch.nn.Linear(3, 5).state_dict()
        # 计算权重和偏置的字节大小
        weight_nbytes = sd['weight'].untyped_storage().nbytes()
        bias_nbytes = sd['bias'].untyped_storage().nbytes()
        
        # 根据文件名是否存在选择文件创建函数
        # 如果有 filename，则使用 TemporaryFileName 创建临时文件名字符串
        # 如果没有 filename，则使用 tempfile.NamedTemporaryFile 创建临时文件对象
        file_creation_func = TemporaryFileName if filename else tempfile.NamedTemporaryFile

        # 使用创建函数生成两个临时文件对象 f 和 g
        with file_creation_func() as f, file_creation_func() as g:
            # 将状态字典 sd 保存到文件 f 中
            torch.save(sd, f)
            if not filename:
                f.seek(0)
            
            # 从文件 f 中提取 'data.pkl' 用于虚假检查点
            with torch.serialization._open_file_like(f, 'rb') as opened_file:
                with torch.serialization._open_zipfile_reader(opened_file) as zip_file:
                    # 从 ZIP 文件中获取 'data.pkl' 的内容并存储在 data_file 中
                    data_file = io.BytesIO(zip_file.get_record('data.pkl'))
                    # 获取 'data/0' 和 'data/1' 的偏移量
                    data_0_offset = zip_file.get_record_offset('data/0')
                    data_1_offset = zip_file.get_record_offset('data/1')

            # 在打开的文件（根据是否有 filename 决定是文件对象 f 还是其名称）中写入空数据 '0' * weight_nbytes 到 'data/0' 和 'data/1'
            with open(f if filename else f.name, 'rb+') as opened_f:
                opened_f.seek(data_0_offset)
                opened_f.write(b'0' * weight_nbytes)
                opened_f.seek(data_1_offset)
                opened_f.write(b'0' * bias_nbytes)

            # 使用 g 文件对象的 ZIP 文件写入器进行操作
            with torch.serialization._open_zipfile_writer(g) as zip_file:
                # 获取 data_file 的值并写入 'data.pkl' 记录
                data_value = data_file.getvalue()
                zip_file.write_record('data.pkl', data_value, len(data_value))
                # 写入系统字节顺序到 'byteorder' 记录
                zip_file.write_record('byteorder', sys.byteorder, len(sys.byteorder))
                # 为 'data/0' 和 'data/1' 的数据记录写入元数据信息（存储字节大小）
                zip_file.write_record_metadata('data/0', weight_nbytes)
                zip_file.write_record_metadata('data/1', bias_nbytes)

            if not filename:
                # 如果没有 filename，则将 f 和 g 的指针位置设置为文件开头
                f.seek(0)
                g.seek(0)
            
            # 从 g 中加载状态字典，并从 f 中加载参考状态字典
            sd_loaded = torch.load(g)
            sd_loaded_ref = torch.load(f)
            # 断言加载的状态字典与参考状态字典相等
            self.assertEqual(sd_loaded, sd_loaded_ref)

    def run(self, *args, **kwargs):
        # 使用 ZIP 文件序列化方法运行父类的 run 方法
        with serialization_method(use_zip=True):
            return super().run(*args, **kwargs)
class TestWrapperSubclass(torch.Tensor):
    elem: torch.Tensor
    __slots__ = ['elem', 'other']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # 封装的张量（TestSubclass）只是一个元张量，不持有任何内存（元张量通常是从中创建子类的首选张量类型）...
        r = torch.Tensor._make_subclass(cls, elem.to('meta'), elem.requires_grad)
        # ...真正的张量作为元素存储在张量上。
        r.elem = elem
        return r

    def clone(self):
        # 创建并返回当前实例的克隆副本
        return type(self)(self.elem.clone())


class TestGetStateSubclass(torch.Tensor):
    elem: torch.Tensor
    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # 封装的张量（TestSubclass）只是一个元张量，不持有任何内存（元张量通常是从中创建子类的首选张量类型）...
        r = torch.Tensor._make_subclass(cls, elem.to('meta'), elem.requires_grad)
        # ...真正的张量作为元素存储在张量上。
        r.elem = elem
        return r

    def __getstate__(self):
        # 返回当前实例的序列化状态，包括标记、elem属性和自定义的__dict__属性
        return ("foo", getattr(self, "elem", None), self.__dict__)

    def __setstate__(self, state):
        # 根据给定的状态恢复实例的状态
        marker, self.elem, self.__dict__ = state
        if not marker == "foo":
            # 如果标记不匹配，抛出运行时错误
            raise RuntimeError("Invalid state for TestGetStateSubclass")
        # 添加一个新的属性作为重新加载的标记
        self.reloaded = True


class TestEmptySubclass(torch.Tensor):
    # TestEmptySubclass 类继承自 torch.Tensor，但没有定义任何新的属性或方法
    ...


class TestSubclassSerialization(TestCase):
    def test_tensor_subclass_wrapper_serialization(self):
        # 创建一个包装了随机张量的 TestWrapperSubclass 实例
        wrapped_tensor = torch.rand(2)
        my_tensor = TestWrapperSubclass(wrapped_tensor)

        foo_val = "bar"
        # 添加一个自定义属性 foo 到 my_tensor 上
        my_tensor.foo = foo_val
        # 断言自定义属性 foo 的值与预期值相等
        self.assertEqual(my_tensor.foo, foo_val)

        with BytesIOContext() as f:
            # 将 my_tensor 序列化并保存到字节流 f 中
            torch.save(my_tensor, f)
            f.seek(0)
            # 从字节流中加载数据并反序列化为 new_tensor
            new_tensor = torch.load(f)

        # 断言 new_tensor 是 TestWrapperSubclass 的实例
        self.assertIsInstance(new_tensor, TestWrapperSubclass)
        # 断言 new_tensor 的 elem 属性与 my_tensor 的 elem 属性相等
        self.assertEqual(new_tensor.elem, my_tensor.elem)
        # 断言 new_tensor 的 foo 属性与 my_tensor 的 foo 属性相等
        self.assertEqual(new_tensor.foo, foo_val)

    def test_tensor_subclass_getstate_overwrite(self):
        # 创建一个包装了随机张量的 TestGetStateSubclass 实例
        wrapped_tensor = torch.rand(2)
        my_tensor = TestGetStateSubclass(wrapped_tensor)

        foo_val = "bar"
        # 添加一个自定义属性 foo 到 my_tensor 上
        my_tensor.foo = foo_val
        # 断言自定义属性 foo 的值与预期值相等
        self.assertEqual(my_tensor.foo, foo_val)

        with BytesIOContext() as f:
            # 将 my_tensor 序列化并保存到字节流 f 中
            torch.save(my_tensor, f)
            f.seek(0)
            # 从字节流中加载数据并反序列化为 new_tensor
            new_tensor = torch.load(f)

        # 断言 new_tensor 是 TestGetStateSubclass 的实例
        self.assertIsInstance(new_tensor, TestGetStateSubclass)
        # 断言 new_tensor 的 elem 属性与 my_tensor 的 elem 属性相等
        self.assertEqual(new_tensor.elem, my_tensor.elem)
        # 断言 new_tensor 的 foo 属性与 my_tensor 的 foo 属性相等
        self.assertEqual(new_tensor.foo, foo_val)
        # 断言 new_tensor 的 reloaded 属性为 True
        self.assertTrue(new_tensor.reloaded)
    # 定义一个测试方法，用于测试深度复制张量子类对象
    def test_tensor_subclass_deepcopy(self):
        # 创建一个包装了随机张量的子类对象
        wrapped_tensor = torch.rand(2)
        my_tensor = TestWrapperSubclass(wrapped_tensor)

        # 设置一个字符串值 foo_val，并将其作为属性 foo 添加到 my_tensor 中
        foo_val = "bar"
        my_tensor.foo = foo_val
        # 断言 my_tensor 的属性 foo 等于 foo_val
        self.assertEqual(my_tensor.foo, foo_val)

        # 对 my_tensor 进行深度复制，生成一个新对象 new_tensor
        new_tensor = deepcopy(my_tensor)

        # 断言 new_tensor 是 TestWrapperSubclass 的实例
        self.assertIsInstance(new_tensor, TestWrapperSubclass)
        # 断言 new_tensor 的元素 elem 等于 my_tensor 的元素 elem
        self.assertEqual(new_tensor.elem, my_tensor.elem)
        # 断言 new_tensor 的属性 foo 等于 foo_val
        self.assertEqual(new_tensor.foo, foo_val)

    # 参数化装饰器，定义了一个测试方法，用于测试深度复制克隆的张量
    @parametrize('requires_grad', (True, False))
    def test_cloned_deepcopy(self, requires_grad):
        # 创建一个随机张量 my_tensor，根据参数 requires_grad 和 device 进行设置
        my_tensor = torch.rand(2, requires_grad=requires_grad, device='meta')

        # 对 my_tensor 进行深度复制，生成一个新对象 new_tensor
        new_tensor = deepcopy(my_tensor)

        # 断言 new_tensor 的 requires_grad 属性等于 my_tensor 的 requires_grad 属性
        self.assertEqual(new_tensor.requires_grad, my_tensor.requires_grad)

    # 定义一个测试方法，用于测试空子类对象的序列化
    def test_empty_class_serialization(self):
        # 创建一个 TestEmptySubclass 的对象 tensor，包含一个单元素列表 [1.]
        tensor = TestEmptySubclass([1.])
        # 使用 copy.copy 对 tensor 进行浅复制，生成一个新对象 tensor2
        tensor2 = copy.copy(tensor)

        # 使用 BytesIOContext 上下文管理器 f 进行 torch 对象的序列化和反序列化测试
        with BytesIOContext() as f:
            torch.save(tensor, f)
            f.seek(0)
            tensor2 = torch.load(f)

        # 创建一个空的 TestEmptySubclass 的对象 tensor
        tensor = TestEmptySubclass()
        # 使用 copy.copy 对 tensor 进行浅复制，生成一个新对象 tensor2
        # 注意 tensor.data_ptr() == 0 的情况
        tensor2 = copy.copy(tensor)

        # 使用 BytesIOContext 上下文管理器 f 进行 torch 对象的序列化和反序列化测试
        with BytesIOContext() as f:
            torch.save(tensor, f)
            f.seek(0)
            tensor2 = torch.load(f)

    # 装饰器 @skipIfTorchDynamo，条件是当 Torch Dynamo 模式下运行时跳过测试
    @skipIfTorchDynamo("name 'SYNTHETIC_LOCAL' is not defined")
    def test_safe_globals_for_weights_only(self):
        '''
        Tests import semantic for tensor subclass and the {add/get/clear}_safe_globals APIs
        '''
        # 创建一个包含两个随机张量的 TwoTensor 实例
        t = TwoTensor(torch.randn(2, 3), torch.randn(2, 3))
        # 将 TwoTensor 实例 t 包装成 torch.nn.Parameter 对象 p
        p = torch.nn.Parameter(t)
        # 创建有序字典 sd，包含键值对 ('t', t) 和 ('p', p)
        sd = OrderedDict([('t', t), ('p', p)])

        # 使用临时文件保存 sd 对象
        with tempfile.NamedTemporaryFile() as f:
            torch.save(sd, f)

            # 当使用 weights_only=True 加载 tensor 子类时应该失败，
            # 因为 tensor 子类不在 safe_globals 中
            with self.assertRaisesRegex(pickle.UnpicklingError,
                                        "Unsupported global: GLOBAL torch.testing._internal.two_tensor.TwoTensor"):
                f.seek(0)
                sd = torch.load(f, weights_only=True)

            # 如果类标记为 safe，则加载 tensor 子类应该成功
            f.seek(0)
            try:
                torch.serialization.add_safe_globals([TwoTensor])
                self.assertTrue(torch.serialization.get_safe_globals() == [TwoTensor])
                sd = torch.load(f, weights_only=True)
                self.assertEqual(sd['t'], t)
                self.assertEqual(sd['p'], p)

                # 清除 safe globals 后应该再次失败
                torch.serialization.clear_safe_globals()
                f.seek(0)
                with self.assertRaisesRegex(pickle.UnpicklingError,
                                            "Unsupported global: GLOBAL torch.testing._internal.two_tensor.TwoTensor"):
                    torch.load(f, weights_only=True)
            finally:
                torch.serialization.clear_safe_globals()

    @unittest.skipIf(not torch.cuda.is_available(), "map_location loads to cuda")
    def test_tensor_subclass_map_location(self):
        # 创建一个包含两个随机张量的 TwoTensor 实例 t
        t = TwoTensor(torch.randn(2, 3), torch.randn(2, 3))
        # 创建字典 sd，包含键 't' 和值 t
        sd = {'t': t}

        # 使用临时文件保存 sd 对象
        with TemporaryFileName() as f:
            torch.save(sd, f)
            
            # 使用 map_location=torch.device('cuda:0') 加载文件应该将张量加载到 GPU 上
            sd_loaded = torch.load(f, map_location=torch.device('cuda:0'))
            self.assertTrue(sd_loaded['t'].device == torch.device('cuda:0'))
            self.assertTrue(sd_loaded['t'].a.device == torch.device('cuda:0'))
            self.assertTrue(sd_loaded['t'].b.device == torch.device('cuda:0'))
            
            # 确保在多次 torch.load 调用时 map_location 不会传播
            sd_loaded = torch.load(f)
            self.assertTrue(sd_loaded['t'].device == torch.device('cpu'))
            self.assertTrue(sd_loaded['t'].a.device == torch.device('cpu'))
            self.assertTrue(sd_loaded['t'].b.device == torch.device('cpu'))
# 调用函数 instantiate_device_type_tests，并传入 TestBothSerialization 类及其全局作用域的变量集合，用于实例化设备类型测试
instantiate_device_type_tests(TestBothSerialization, globals())

# 调用函数 instantiate_parametrized_tests，并传入 TestSubclassSerialization 类，用于实例化参数化测试
instantiate_parametrized_tests(TestSubclassSerialization)

# 调用函数 instantiate_parametrized_tests，并传入 TestOldSerialization 类，用于实例化参数化测试
instantiate_parametrized_tests(TestOldSerialization)

# 调用函数 instantiate_parametrized_tests，并传入 TestSerialization 类，用于实例化参数化测试
instantiate_parametrized_tests(TestSerialization)

# 检查当前脚本是否作为主程序运行
if __name__ == '__main__':
    # 运行测试函数
    run_tests()
```