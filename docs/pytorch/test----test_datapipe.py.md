# `.\pytorch\test\test_datapipe.py`

```py
# mypy: ignore-errors

# Owner(s): ["module: dataloader"]

# 导入所需的标准库和第三方库
import copy  # 导入复制对象的标准库模块
import itertools  # 导入创建迭代器的标准库模块
import os  # 导入操作系统相关功能的标准库模块
import os.path  # 导入操作文件路径相关功能的标准库模块
import pickle  # 导入序列化和反序列化 Python 对象的标准库模块
import pydoc  # 导入文档生成和查找的标准库模块
import random  # 导入生成伪随机数的标准库模块
import sys  # 导入系统特定参数和功能的标准库模块
import tempfile  # 导入创建临时文件和目录的标准库模块
import warnings  # 导入警告处理的标准库模块
from functools import partial  # 导入函数工具的标准库模块中的 partial 函数
from typing import (  # 导入类型提示相关的标准库模块中的多个类型
    Any,
    Awaitable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

if not TYPE_CHECKING:
    # 如果不是类型检查模式，则从 typing_extensions 导入 NamedTuple
    from typing_extensions import NamedTuple
else:
    # 如果是类型检查模式，则直接从 typing 导入 NamedTuple
    from typing import NamedTuple

import operator  # 导入操作符操作函数的标准库模块
from unittest import skipIf  # 导入跳过测试条件的单元测试库模块

import numpy as np  # 导入数值计算库 numpy

import torch  # 导入深度学习框架 PyTorch
import torch.nn as nn  # 导入神经网络模块 nn
import torch.utils.data.datapipes as dp  # 导入 PyTorch 数据管道模块 dp
import torch.utils.data.graph  # 导入 PyTorch 数据图模块
import torch.utils.data.graph_settings  # 导入 PyTorch 数据图设置模块
from torch.testing._internal.common_utils import (  # 导入 PyTorch 内部测试工具模块中的多个功能
    run_tests,
    skipIfNoDill,
    skipIfTorchDynamo,
    suppress_warnings,
    TEST_DILL,
    TestCase,
)
from torch.utils._import_utils import import_dill  # 导入 PyTorch 导入工具模块中的 import_dill 函数
from torch.utils.data import (  # 导入 PyTorch 数据处理工具模块中的多个功能
    argument_validation,
    DataChunk,
    DataLoader,
    IterDataPipe,
    MapDataPipe,
    RandomSampler,
    runtime_validation,
    runtime_validation_disabled,
)
from torch.utils.data.datapipes.dataframe import (  # 导入 PyTorch 数据管道模块中的 dataframe 相关功能
    CaptureDataFrame,
    dataframe_wrapper as df_wrapper,
)
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES  # 导入 PyTorch 数据管道模块中的分片优先级常量
from torch.utils.data.datapipes.utils.common import StreamWrapper  # 导入 PyTorch 数据管道模块中的流包装器
from torch.utils.data.datapipes.utils.decoder import (  # 导入 PyTorch 数据管道模块中的解码器相关功能
    basichandlers as decoder_basichandlers,
)
from torch.utils.data.datapipes.utils.snapshot import (  # 导入 PyTorch 数据管道模块中的快照恢复功能
    _simple_graph_snapshot_restoration,
)
from torch.utils.data.graph import traverse_dps  # 导入 PyTorch 数据图模块中的 traverse_dps 函数

# 导入 dill 库，用于对象序列化
dill = import_dill()
# 检查是否导入了 dill 库
HAS_DILL = TEST_DILL

try:
    import pandas  # type: ignore[import]  # noqa: F401 F403

    HAS_PANDAS = True
except ImportError:
    # 如果导入 pandas 失败，则设置为 False
    HAS_PANDAS = False
# 根据是否有 pandas 库，设置跳过测试条件
skipIfNoDataFrames = skipIf(not HAS_PANDAS, "no dataframes (pandas)")

# 设置类型检查的跳过条件，用于解决类型提示中的 bug
skipTyping = skipIf(True, "TODO: Fix typing bug")
T_co = TypeVar("T_co", covariant=True)


def create_temp_dir_and_files():
    # 创建临时目录和文件，这些文件将在 tearDown() 中释放和删除
    # 添加 `noqa: P201` 来避免 mypy 在此函数中未释放目录句柄时的警告
    temp_dir = tempfile.TemporaryDirectory()  # 创建临时目录对象
    temp_dir_path = temp_dir.name  # 获取临时目录的路径
    with tempfile.NamedTemporaryFile(
        dir=temp_dir_path, delete=False, suffix=".txt"
    ) as f:
        temp_file1_name = f.name  # 创建带有指定后缀的临时文本文件
    with tempfile.NamedTemporaryFile(
        dir=temp_dir_path, delete=False, suffix=".byte"
    ) as f:
        temp_file2_name = f.name  # 创建带有指定后缀的临时二进制文件
    with tempfile.NamedTemporaryFile(
        dir=temp_dir_path, delete=False, suffix=".empty"
    ) as f:
        temp_file3_name = f.name  # 创建带有指定后缀的空文件

    with open(temp_file1_name, "w") as f1:
        f1.write("0123456789abcdef")  # 向文本文件中写入内容
    with open(temp_file2_name, "wb") as f2:
        f2.write(b"0123456789abcdef")  # 向二进制文件中写入内容
    # 创建一个临时子目录，位于指定的临时目录路径下
    temp_sub_dir = tempfile.TemporaryDirectory(dir=temp_dir_path)  # noqa: P201
    # 获取临时子目录的路径名
    temp_sub_dir_path = temp_sub_dir.name
    # 在临时子目录中创建一个命名临时文件，文件名以".txt"为后缀，不会在关闭时删除
    with tempfile.NamedTemporaryFile(
        dir=temp_sub_dir_path, delete=False, suffix=".txt"
    ) as f:
        # 获取第一个临时文件的完整路径名
        temp_sub_file1_name = f.name
    # 在临时子目录中创建一个命名临时文件，文件名以".byte"为后缀，不会在关闭时删除
    with tempfile.NamedTemporaryFile(
        dir=temp_sub_dir_path, delete=False, suffix=".byte"
    ) as f:
        # 获取第二个临时文件的完整路径名
        temp_sub_file2_name = f.name

    # 打开第一个临时文件，以写入模式，并写入固定的字符串内容
    with open(temp_sub_file1_name, "w") as f1:
        f1.write("0123456789abcdef")
    # 打开第二个临时文件，以二进制写入模式，并写入固定的字节内容
    with open(temp_sub_file2_name, "wb") as f2:
        f2.write(b"0123456789abcdef")

    # 返回一个包含两个元组的列表，每个元组包含相关的临时目录和文件名信息
    return [
        (temp_dir, temp_file1_name, temp_file2_name, temp_file3_name),
        (temp_sub_dir, temp_sub_file1_name, temp_sub_file2_name),
    ]
# 定义一个函数 reset_after_n_next_calls，接受一个数据管道 datapipe 和一个整数 n，并返回一个元组
# 元组包含两个列表：
#   1. 在重置之前从数据管道中获取的 n 个元素组成的列表
#   2. 在重置后从数据管道中获取的所有元素组成的列表
def reset_after_n_next_calls(
    datapipe: Union[IterDataPipe[T_co], MapDataPipe[T_co]], n: int
) -> Tuple[List[T_co], List[T_co]]:
    # 创建数据管道的迭代器
    it = iter(datapipe)
    # 存储重置前的元素列表
    res_before_reset = []
    # 获取数据管道中的前 n 个元素
    for _ in range(n):
        res_before_reset.append(next(it))
    # 返回元组，第一个元素是重置前的元素列表，第二个元素是重置后的所有元素列表
    return res_before_reset, list(datapipe)


# 定义一个函数 odd_or_even，接受一个整数 x，返回 x 的奇偶性，奇数返回 1，偶数返回 0
def odd_or_even(x: int) -> int:
    return x % 2


# 测试类 TestDataChunk，继承自 TestCase
class TestDataChunk(TestCase):
    # 设置测试环境
    def setUp(self):
        # 创建一个随机排列的整数列表
        self.elements = list(range(10))
        random.shuffle(self.elements)
        # 创建 DataChunk 对象，存储随机排列的整数列表
        self.chunk: DataChunk[int] = DataChunk(self.elements)

    # 测试索引获取功能
    def test_getitem(self):
        # 断言 DataChunk 对象按索引访问与原始列表一致
        for i in range(10):
            self.assertEqual(self.elements[i], self.chunk[i])

    # 测试迭代功能
    def test_iter(self):
        # 断言迭代 DataChunk 对象与迭代原始列表一致
        for ele, dc in zip(self.elements, iter(self.chunk)):
            self.assertEqual(ele, dc)

    # 测试长度功能
    def test_len(self):
        # 断言 DataChunk 对象长度与原始列表长度一致
        self.assertEqual(len(self.elements), len(self.chunk))

    # 测试转换为字符串功能
    def test_as_string(self):
        # 断言 DataChunk 对象转换为字符串与原始列表转换为字符串一致
        self.assertEqual(str(self.chunk), str(self.elements))

        # 创建一个包含原始列表的列表
        batch = [self.elements] * 3
        # 创建一个包含 DataChunk 对象的列表
        chunks: List[DataChunk[int]] = [DataChunk(self.elements)] * 3
        # 断言这两个列表转换为字符串后一致
        self.assertEqual(str(batch), str(chunks))

    # 测试排序功能
    def test_sort(self):
        # 创建 DataChunk 对象
        chunk: DataChunk[int] = DataChunk(self.elements)
        # 对 DataChunk 对象进行排序
        chunk.sort()
        # 断言排序后的对象仍然是 DataChunk 类型
        self.assertTrue(isinstance(chunk, DataChunk))
        # 断言排序后的 DataChunk 对象与已排序的索引列表一致
        for i, d in enumerate(chunk):
            self.assertEqual(i, d)

    # 测试反转功能
    def test_reverse(self):
        # 创建 DataChunk 对象
        chunk: DataChunk[int] = DataChunk(self.elements)
        # 反转 DataChunk 对象
        chunk.reverse()
        # 断言反转后的对象仍然是 DataChunk 类型
        self.assertTrue(isinstance(chunk, DataChunk))
        # 断言反转后的 DataChunk 对象与反转后的索引列表一致
        for i in range(10):
            self.assertEqual(chunk[i], self.elements[9 - i])

    # 测试随机打乱功能
    def test_random_shuffle(self):
        # 创建一个整数列表
        elements = list(range(10))
        # 创建 DataChunk 对象
        chunk: DataChunk[int] = DataChunk(elements)

        # 创建一个固定种子的随机数生成器
        rng = random.Random(0)
        # 使用生成器对 DataChunk 对象进行随机打乱
        rng.shuffle(chunk)

        # 创建一个新的随机数生成器，与前面的种子相同
        rng = random.Random(0)
        # 使用新生成器对整数列表进行随机打乱
        rng.shuffle(elements)

        # 断言打乱后的 DataChunk 对象与打乱后的整数列表一致
        self.assertEqual(chunk, elements)


# 测试类 TestStreamWrapper，继承自 TestCase
class TestStreamWrapper(TestCase):
    # 内部类 _FakeFD，模拟文件描述符
    class _FakeFD:
        # 初始化方法，接受文件路径 filepath
        def __init__(self, filepath):
            self.filepath = filepath
            self.opened = False  # 表示文件是否已打开
            self.closed = False  # 表示文件是否已关闭

        # 打开文件方法
        def open(self):
            self.opened = True

        # 读取文件方法
        def read(self):
            # 如果文件已打开，返回连接的字符串
            if self.opened:
                return "".join(self)
            else:
                # 如果文件未打开，抛出 OSError 异常
                raise OSError("Cannot read from un-opened file descriptor")

        # 迭代器方法，模拟文件内容
        def __iter__(self):
            for i in range(5):
                yield str(i)

        # 关闭文件方法
        def close(self):
            # 如果文件已打开，将状态改为关闭
            if self.opened:
                self.opened = False
                self.closed = True

        # 返回类的字符串表示
        def __repr__(self):
            return "FakeFD"
    # 测试用例：验证 StreamWrapper 类的实例在调用 dir() 方法后是否包含指定的 API 方法
    def test_dir(self):
        # 创建一个空字符串的假文件描述符对象
        fd = TestStreamWrapper._FakeFD("")
        # 使用假文件描述符对象创建 StreamWrapper 包装器对象
        wrap_fd = StreamWrapper(fd)

        # 获得包装器对象的所有属性和方法名，并转换为集合
        s = set(dir(wrap_fd))
        # 遍历预期包含的 API 方法列表
        for api in ["open", "read", "close"]:
            # 断言每个 API 方法名是否在集合中
            self.assertTrue(api in s)

    # 装饰器跳过条件：当未使用 Torch Dynamo 时执行测试
    @skipIfTorchDynamo()
    # 测试用例：验证 StreamWrapper 类的基本 API 功能
    def test_api(self):
        # 创建一个空字符串的假文件描述符对象
        fd = TestStreamWrapper._FakeFD("")
        # 使用假文件描述符对象创建 StreamWrapper 包装器对象
        wrap_fd = StreamWrapper(fd)

        # 断言文件描述符对象尚未打开
        self.assertFalse(fd.opened)
        # 断言文件描述符对象尚未关闭
        self.assertFalse(fd.closed)
        # 使用上下文管理器断言在未打开时调用 read() 方法会引发 IOError 异常
        with self.assertRaisesRegex(IOError, "Cannot read from"):
            wrap_fd.read()

        # 打开包装器对象
        wrap_fd.open()
        # 断言文件描述符对象已打开
        self.assertTrue(fd.opened)
        # 断言读取数据与预期值相等
        self.assertEqual("01234", wrap_fd.read())

        # 删除包装器对象
        del wrap_fd
        # 断言文件描述符对象已关闭
        self.assertFalse(fd.opened)
        # 断言文件描述符对象已关闭
        self.assertTrue(fd.closed)

    # 测试用例：验证 StreamWrapper 对象的 pickle 序列化和反序列化行为
    def test_pickle(self):
        # 使用临时文件进行上下文管理
        with tempfile.TemporaryFile() as f:
            # 使用上下文管理器断言尝试对文件对象进行 pickle 序列化会引发 TypeError 异常
            with self.assertRaises(TypeError) as ctx1:
                pickle.dumps(f)

            # 使用文件对象创建 StreamWrapper 包装器对象
            wrap_f = StreamWrapper(f)
            # 使用上下文管理器断言尝试对包装器对象进行 pickle 序列化会引发 TypeError 异常
            with self.assertRaises(TypeError) as ctx2:
                pickle.dumps(wrap_f)

            # 断言两个异常对象的字符串表示相等
            self.assertEqual(str(ctx1.exception), str(ctx2.exception))

        # 创建一个空字符串的假文件描述符对象
        fd = TestStreamWrapper._FakeFD("")
        # 使用假文件描述符对象创建 StreamWrapper 包装器对象
        wrap_fd = StreamWrapper(fd)
        # 使用 pickle 进行序列化和反序列化，并将结果赋给临时变量
        _ = pickle.loads(pickle.dumps(wrap_fd))

    # 测试用例：验证 StreamWrapper 对象的字符串表示形式
    def test_repr(self):
        # 创建一个空字符串的假文件描述符对象
        fd = TestStreamWrapper._FakeFD("")
        # 使用假文件描述符对象创建 StreamWrapper 包装器对象
        wrap_fd = StreamWrapper(fd)
        # 断言包装器对象的字符串表示形式与预期值相等
        self.assertEqual(str(wrap_fd), "StreamWrapper<FakeFD>")

        # 使用临时文件进行上下文管理
        with tempfile.TemporaryFile() as f:
            # 使用文件对象创建 StreamWrapper 包装器对象
            wrap_f = StreamWrapper(f)
            # 断言包装器对象的字符串表示形式与预期值相等（包含文件对象的字符串表示形式）
            self.assertEqual(str(wrap_f), "StreamWrapper<" + str(f) + ">")
# 定义一个测试类 TestIterableDataPipeBasic，继承自 TestCase，用于测试可迭代数据管道的基本功能
class TestIterableDataPipeBasic(TestCase):

    # 在每个测试方法执行前调用，设置临时目录和文件
    def setUp(self):
        # 调用 create_temp_dir_and_files 函数创建临时目录和文件
        ret = create_temp_dir_and_files()
        # 将返回结果解包并赋值给对应的实例变量
        self.temp_dir = ret[0][0]
        self.temp_files = ret[0][1:]
        self.temp_sub_dir = ret[1][0]
        self.temp_sub_files = ret[1][1:]

    # 在每个测试方法执行后调用，清理临时目录和文件
    def tearDown(self):
        try:
            # 清理临时子目录
            self.temp_sub_dir.cleanup()
            # 清理临时主目录
            self.temp_dir.cleanup()
        except Exception as e:
            # 如果清理失败，发出警告并记录异常信息
            warnings.warn(
                f"TestIterableDatasetBasic was not able to cleanup temp dir due to {str(e)}"
            )

    # 测试方法：测试 FileLister 类的基本迭代功能
    def test_listdirfiles_iterable_datapipe(self):
        # 获取临时目录的路径名
        temp_dir = self.temp_dir.name
        # 创建 IterDataPipe 对象 datapipe，用于列出 temp_dir 目录下的文件列表
        datapipe: IterDataPipe = dp.iter.FileLister(temp_dir, "")

        # 初始化计数器
        count = 0
        # 遍历 datapipe 中的每个 pathname
        for pathname in datapipe:
            count = count + 1
            # 断言 pathname 存在于 self.temp_files 中
            self.assertTrue(pathname in self.temp_files)
        # 断言计数器值等于 self.temp_files 的长度
        self.assertEqual(count, len(self.temp_files))

        # 重置计数器
        count = 0
        # 创建支持递归的 FileLister 对象 datapipe
        datapipe = dp.iter.FileLister(temp_dir, "", recursive=True)
        # 遍历 datapipe 中的每个 pathname
        for pathname in datapipe:
            count = count + 1
            # 断言 pathname 存在于 self.temp_files 或 self.temp_sub_files 中
            self.assertTrue(
                (pathname in self.temp_files) or (pathname in self.temp_sub_files)
            )
        # 断言计数器值等于 self.temp_files 和 self.temp_sub_files 总长度
        self.assertEqual(count, len(self.temp_files) + len(self.temp_sub_files))

        # 获取 self.temp_files 的副本
        temp_files = self.temp_files
        # 创建 FileLister 对象 datapipe，传入 temp_dir 和 temp_files
        datapipe = dp.iter.FileLister([temp_dir, *temp_files])
        # 重置计数器
        count = 0
        # 遍历 datapipe 中的每个 pathname
        for pathname in datapipe:
            count += 1
            # 断言 pathname 存在于 self.temp_files 中
            self.assertTrue(pathname in self.temp_files)
        # 断言计数器值等于 self.temp_files 的两倍长度
        self.assertEqual(count, 2 * len(self.temp_files))

        # 测试函数式 API
        # 调用 list_files 方法，获取返回的 datapipe
        datapipe = datapipe.list_files()
        # 重置计数器
        count = 0
        # 遍历 datapipe 中的每个 pathname
        for pathname in datapipe:
            count += 1
            # 断言 pathname 存在于 self.temp_files 中
            self.assertTrue(pathname in self.temp_files)
        # 断言计数器值等于 self.temp_files 的两倍长度
        self.assertEqual(count, 2 * len(self.temp_files))

    # 测试方法：测试 FileLister 类的确定性迭代功能
    def test_listdirfilesdeterministic_iterable_datapipe(self):
        # 获取临时目录的路径名
        temp_dir = self.temp_dir.name

        # 创建 FileLister 对象 datapipe，用于列出 temp_dir 目录下的文件列表
        datapipe = dp.iter.FileLister(temp_dir, "")
        # 断言两次对 datapipe 的迭代结果应该一致
        self.assertEqual(list(datapipe), list(datapipe))

        # 创建支持递归的 FileLister 对象 datapipe
        datapipe = dp.iter.FileLister(temp_dir, "", recursive=True)
        # 断言两次对 datapipe 的迭代结果应该一致
        self.assertEqual(list(datapipe), list(datapipe))
    # 定义测试方法：测试从磁盘打开文件的可迭代数据管道
    def test_openfilesfromdisk_iterable_datapipe(self):
        # 直接导入 Torch 数据管道中的 FileLister 和 FileOpener 类
        from torch.utils.data.datapipes.iter import FileLister, FileOpener

        # 获取临时目录的路径
        temp_dir = self.temp_dir.name
        # 创建 FileLister 数据管道，指定临时目录和空字符串作为过滤器
        datapipe1 = FileLister(temp_dir, "")
        # 创建 FileOpener 数据管道，以二进制模式打开 FileLister 数据管道
        datapipe2 = FileOpener(datapipe1, mode="b")

        # 初始化计数器
        count = 0
        # 遍历 datapipe2 数据管道中的记录
        for rec in datapipe2:
            # 计数器加一
            count = count + 1
            # 断言记录中的第一个元素（文件路径）存在于 self.temp_files 中
            self.assertTrue(rec[0] in self.temp_files)
            # 使用标准 Python 打开文件并断言其内容与记录中的第二个元素（文件内容对象）的内容相同
            with open(rec[0], "rb") as f:
                self.assertEqual(rec[1].read(), f.read())
                # 关闭记录中的文件内容对象
                rec[1].close()
        # 断言遍历记录的数量等于 self.temp_files 的长度
        self.assertEqual(count, len(self.temp_files))

        # 使用函数式 API 打开文件
        datapipe3 = datapipe1.open_files(mode="b")

        # 初始化计数器
        count = 0
        # 再次遍历 datapipe3 数据管道中的记录
        for rec in datapipe3:
            # 计数器加一
            count = count + 1
            # 断言记录中的第一个元素（文件路径）存在于 self.temp_files 中
            self.assertTrue(rec[0] in self.temp_files)
            # 使用标准 Python 打开文件并断言其内容与记录中的第二个元素（文件内容对象）的内容相同
            with open(rec[0], "rb") as f:
                self.assertEqual(rec[1].read(), f.read())
                # 关闭记录中的文件内容对象
                rec[1].close()
        # 断言遍历记录的数量等于 self.temp_files 的长度
        self.assertEqual(count, len(self.temp_files))

        # 测试 __len__ 方法是否会引发 TypeError 异常
        with self.assertRaises(TypeError):
            len(datapipe3)
    def test_routeddecoder_iterable_datapipe(self):
        # 获取临时目录的名称
        temp_dir = self.temp_dir.name
        # 构建临时 PNG 文件的路径
        temp_pngfile_pathname = os.path.join(temp_dir, "test_png.png")
        # 创建示例 PNG 数据
        png_data = np.array(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=np.single,
        )
        # 将 PNG 数据保存到文件中
        np.save(temp_pngfile_pathname, png_data)
        # 创建文件列表数据管道，指定目录和文件匹配模式
        datapipe1 = dp.iter.FileLister(temp_dir, ["*.png", "*.txt"])
        # 创建文件打开数据管道，以二进制模式打开文件
        datapipe2 = dp.iter.FileOpener(datapipe1, mode="b")

        def _png_decoder(extension, data):
            # PNG 解码器函数，根据扩展名判断是否为 PNG 文件，如果是则加载数据
            if extension != "png":
                return None
            return np.load(data)

        def _helper(prior_dp, dp, channel_first=False):
            # 辅助函数，用于比较和验证数据管道中的数据
            # 检查前置数据管道中的字节流是否已关闭
            for inp in prior_dp:
                self.assertFalse(inp[1].closed)
            # 遍历前置数据管道和当前数据管道的数据记录进行比较
            for inp, rec in zip(prior_dp, dp):
                ext = os.path.splitext(rec[0])[1]
                if ext == ".png":
                    # 如果是 PNG 文件，验证其数据是否与预期一致
                    expected = np.array(
                        [
                            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        ],
                        dtype=np.single,
                    )
                    if channel_first:
                        expected = expected.transpose(2, 0, 1)
                    self.assertEqual(rec[1], expected)
                else:
                    # 如果不是 PNG 文件，以 UTF-8 编码读取文件内容并验证
                    with open(rec[0], "rb") as f:
                        self.assertEqual(rec[1], f.read().decode("utf-8"))
                # 验证对应的字节流在解码后是否已关闭
                self.assertTrue(inp[1].closed)

        # 缓存当前数据管道中的所有数据记录
        cached = list(datapipe2)
        # 使用 PNG 解码器创建路由解码器数据管道
        with warnings.catch_warnings(record=True) as wa:
            datapipe3 = dp.iter.RoutedDecoder(cached, _png_decoder)
        # 向路由解码器添加基础解码处理器
        datapipe3.add_handler(decoder_basichandlers)
        # 使用辅助函数验证数据管道 datapipe3
        _helper(cached, datapipe3)

        # 再次缓存当前数据管道中的所有数据记录
        cached = list(datapipe2)
        # 使用基础解码处理器创建路由解码器数据管道 datapipe4
        with warnings.catch_warnings(record=True) as wa:
            datapipe4 = dp.iter.RoutedDecoder(cached, decoder_basichandlers)
        # 向路由解码器添加 PNG 解码器作为处理器
        datapipe4.add_handler(_png_decoder)
        # 使用辅助函数验证数据管道 datapipe4，指定通道顺序为先通道后样本
        _helper(cached, datapipe4, channel_first=True)
    def test_groupby_iterable_datapipe(self):
        file_list = [
            "a.png",
            "b.png",
            "c.json",
            "a.json",
            "c.png",
            "b.json",
            "d.png",
            "d.json",
            "e.png",
            "f.json",
            "g.png",
            "f.png",
            "g.json",
            "e.json",
            "h.txt",
            "h.json",
        ]
        
        import io  # 导入 io 模块

        # 创建一个包含文件名和 BytesIO 对象的列表，每个文件名对应的 BytesIO 包含 b"12345abcde" 数据
        datapipe1 = dp.iter.IterableWrapper(
            [(filename, io.BytesIO(b"12345abcde")) for filename in file_list]
        )

        # 定义分组函数，根据文件路径获取基本文件名（不带后缀）
        def group_fn(data):
            filepath, _ = data
            return os.path.basename(filepath).split(".")[0]

        # 使用分组函数对 datapipe1 中的数据进行分组，每组包含两个元素
        datapipe2 = dp.iter.Grouper(datapipe1, group_key_fn=group_fn, group_size=2)

        # 定义排序函数，对文件名进行逆序排序
        def order_fn(data):
            data.sort(key=lambda f: f[0], reverse=True)
            return data

        # 对 datapipe2 应用排序函数，并将结果存储在 datapipe3 中
        datapipe3 = dp.iter.Mapper(datapipe2, fn=order_fn)  # type: ignore[var-annotated]

        # 期望的结果列表，包含元组，每个元组包含两个文件名
        expected_result = [
            ("a.png", "a.json"),
            ("c.png", "c.json"),
            ("b.png", "b.json"),
            ("d.png", "d.json"),
            ("f.png", "f.json"),
            ("g.png", "g.json"),
            ("e.png", "e.json"),
            ("h.txt", "h.json"),
        ]

        count = 0
        # 遍历 datapipe3 和 expected_result，进行断言验证
        for rec, expected in zip(datapipe3, expected_result):
            count = count + 1
            # 断言第一个文件名的基本名称与期望一致
            self.assertEqual(os.path.basename(rec[0][0]), expected[0])
            # 断言第二个文件名的基本名称与期望一致
            self.assertEqual(os.path.basename(rec[1][0]), expected[1])
            # 对每个文件的 BytesIO 对象进行断言，验证其内容为 b"12345abcde"，并关闭文件流
            for i in [0, 1]:
                self.assertEqual(rec[i][1].read(), b"12345abcde")
                rec[i][1].close()
        # 最终断言遍历次数为 8
        self.assertEqual(count, 8)

        # 测试 keep_key 选项
        datapipe4 = dp.iter.Grouper(
            datapipe1, group_key_fn=group_fn, keep_key=True, group_size=2
        )

        # 重新定义排序函数，对数据的第二个元素进行文件名逆序排序
        def order_fn(data):
            data[1].sort(key=lambda f: f[0], reverse=True)
            return data

        # 对 datapipe4 应用新的排序函数，并将结果存储在 datapipe5 中
        datapipe5 = dp.iter.Mapper(datapipe4, fn=order_fn)  # type: ignore[var-annotated]

        # 新的期望结果列表，包含元组，每个元组的第一个元素是基本文件名，第二个元素是原始文件名元组
        expected_result = [
            ("a", ("a.png", "a.json")),
            ("c", ("c.png", "c.json")),
            ("b", ("b.png", "b.json")),
            ("d", ("d.png", "d.json")),
            ("f", ("f.png", "f.json")),
            ("g", ("g.png", "g.json")),
            ("e", ("e.png", "e.json")),
            ("h", ("h.txt", "h.json")),
        ]

        count = 0
        # 遍历 datapipe5 和 expected_result，进行断言验证
        for rec, expected in zip(datapipe5, expected_result):
            count = count + 1
            # 断言第一个元素与期望的基本文件名一致
            self.assertEqual(rec[0], expected[0])
            # 断言第二个元素的第一个文件名与期望一致
            self.assertEqual(rec[1][0][0], expected[1][0])
            # 断言第二个元素的第二个文件名与期望一致
            self.assertEqual(rec[1][1][0], expected[1][1])
            # 对每个文件的 BytesIO 对象进行断言，验证其内容为 b"12345abcde"，并关闭文件流
            for i in [0, 1]:
                self.assertEqual(rec[1][i][1].read(), b"12345abcde")
                rec[1][i][1].close()
        # 最终断言遍历次数为 8
        self.assertEqual(count, 8)
    def test_demux_mux_datapipe(self):
        # 创建一个包含10个数字的NumbersDataset对象
        numbers = NumbersDataset(10)
        
        # 对数据进行demux操作，按照x % 2的规则分成两部分，n1和n2
        n1, n2 = numbers.demux(2, lambda x: x % 2)
        
        # 断言n1中的内容是否为[0, 2, 4, 6, 8]
        self.assertEqual([0, 2, 4, 6, 8], list(n1))
        
        # 断言n2中的内容是否为[1, 3, 5, 7, 9]
        self.assertEqual([1, 3, 5, 7, 9], list(n2))

        # Functional Test: demux and mux works sequentially as expected
        # 再次创建一个包含10个数字的NumbersDataset对象
        numbers = NumbersDataset(10)
        
        # 对数据进行demux操作，按照x % 3的规则分成三部分，n1、n2和n3
        n1, n2, n3 = numbers.demux(3, lambda x: x % 3)
        
        # 将n1、n2和n3进行mux操作，合并成一个数据流n
        n = n1.mux(n2, n3)
        
        # 断言n中的内容是否为[0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(list(range(9)), list(n))

        # Functional Test: Uneven DataPipes
        # 创建一个包含不均匀数据的IterableWrapper对象
        source_numbers = list(range(0, 10)) + [10, 12]
        numbers_dp = dp.iter.IterableWrapper(source_numbers)
        
        # 对数据进行demux操作，按照x % 2的规则分成两部分，n1和n2
        n1, n2 = numbers_dp.demux(2, lambda x: x % 2)
        
        # 断言n1中的内容是否为[0, 2, 4, 6, 8, 10, 12]
        self.assertEqual([0, 2, 4, 6, 8, 10, 12], list(n1))
        
        # 断言n2中的内容是否为[1, 3, 5, 7, 9]
        self.assertEqual([1, 3, 5, 7, 9], list(n2))
        
        # 将n1和n2进行mux操作，合并成一个数据流n
        n = n1.mux(n2)
        
        # 断言n中的内容是否为[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(list(range(10)), list(n))

    @suppress_warnings  # Suppress warning for lambda fn
    def test_map_with_col_file_handle_datapipe(self):
        # 获取临时目录的名称
        temp_dir = self.temp_dir.name
        
        # 创建一个FileLister对象，用于列出临时目录中的文件
        datapipe1 = dp.iter.FileLister(temp_dir, "")
        
        # 创建一个FileOpener对象，用于打开datapipe1中的文件
        datapipe2 = dp.iter.FileOpener(datapipe1)

        def _helper(datapipe):
            # 对datapipe进行map操作，对每个元素执行x.read()函数，input_col=1
            dp1 = datapipe.map(lambda x: x.read(), input_col=1)
            
            # 对datapipe进行map操作，对每个元素执行(x[0], x[1].read())的映射
            dp2 = datapipe.map(lambda x: (x[0], x[1].read()))
            
            # 断言dp1和dp2中的内容是否一致
            self.assertEqual(list(dp1), list(dp2))

        # tuple
        # 调用_helper函数处理datapipe2
        _helper(datapipe2)
        
        # list
        # 对datapipe2进行map操作，将每个元素转换为list类型
        datapipe3 = datapipe2.map(lambda x: list(x))
        
        # 再次调用_helper函数处理datapipe3
        _helper(datapipe3)
# 如果没有数据框架，则跳过测试
@skipIfNoDataFrames
class TestCaptureDataFrame(TestCase):
    # 创建并返回一个包含数据的新数据框架
    def get_new_df(self):
        return df_wrapper.create_dataframe([[1, 2]], columns=["a", "b"])

    # 比较捕获数据框架和即时计算的结果
    def compare_capture_and_eager(self, operations):
        cdf = CaptureDataFrame()
        cdf = operations(cdf)  # 对捕获数据框架应用操作
        df = self.get_new_df()
        cdf = cdf.apply_ops(df)  # 应用操作到捕获的数据框架上

        df = self.get_new_df()
        df = operations(df)  # 直接在数据框架上应用操作

        self.assertTrue(df.equals(cdf))  # 断言捕获的数据框架与即时计算结果相等

    # 测试基本的数据捕获功能
    def test_basic_capture(self):
        # 定义在数据框架上的操作
        def operations(df):
            df["c"] = df.b + df["a"] * 7
            # 某种方式吞噬了 pandas 的 UserWarning，当 `df.c = df.b + df['a'] * 7` 时
            return df

        self.compare_capture_and_eager(operations)


class TestDataFramesPipes(TestCase):
    """
    如果没有安装 dill，大多数测试将失败。
    需要重写它们以避免多次跳过。
    """

    # 获取数字数据流
    def _get_datapipe(self, range=10, dataframe_size=7):
        return NumbersDataset(range).map(lambda i: (i, i % 3))

    # 获取数据框架数据流
    def _get_dataframes_pipe(self, range=10, dataframe_size=7):
        return (
            NumbersDataset(range)
            .map(lambda i: (i, i % 3))
            ._to_dataframes_pipe(columns=["i", "j"], dataframe_size=dataframe_size)
        )

    # 如果没有数据框架则跳过测试，如果没有 dill 则也跳过（由于 lambda 在 map 中使用）
    @skipIfNoDataFrames
    @skipIfNoDill
    def test_capture(self):
        # 创建包含 (i, j, i + 3*j) 的数据流 dp_numbers
        dp_numbers = self._get_datapipe().map(lambda x: (x[0], x[1], x[1] + 3 * x[0]))
        # 获取数据框架数据流 df_numbers
        df_numbers = self._get_dataframes_pipe()
        # 添加新的列 'k' 到 df_numbers，计算公式为 j + i * 3
        df_numbers["k"] = df_numbers["j"] + df_numbers.i * 3
        expected = list(dp_numbers)  # 期望结果是 dp_numbers 的列表形式
        actual = list(df_numbers)  # 实际结果是 df_numbers 的列表形式
        self.assertEqual(expected, actual)  # 断言期望结果和实际结果相等

    # 如果没有数据框架则跳过测试，如果没有 dill 则也跳过
    @skipIfNoDataFrames
    @skipIfNoDill
    def test_shuffle(self):
        # 将 df_numbers 进行洗牌操作
        df_numbers = self._get_dataframes_pipe(range=1000).shuffle()
        # 获取数字数据流 dp_numbers
        dp_numbers = self._get_datapipe(range=1000)
        # 将 df_numbers 转换为列表形式 df_result
        df_result = [tuple(item) for item in df_numbers]
        # 断言 dp_numbers 与 df_result 不相等
        self.assertNotEqual(list(dp_numbers), df_result)
        # 断言 dp_numbers 与排序后的 df_result 相等
        self.assertEqual(list(dp_numbers), sorted(df_result))

    # 如果没有数据框架则跳过测试，如果没有 dill 则也跳过
    @skipIfNoDataFrames
    @skipIfNoDill
    def test_batch(self):
        # 将 df_numbers 进行分批处理，每批大小为 8
        df_numbers = self._get_dataframes_pipe(range=100).batch(8)
        df_numbers_list = list(df_numbers)
        last_batch = df_numbers_list[-1]  # 获取最后一批数据
        self.assertEqual(4, len(last_batch))  # 断言最后一批数据长度为 4
        unpacked_batch = [tuple(row) for row in last_batch]  # 将最后一批数据解压为元组列表
        # 断言解压后的最后一批数据与预期结果相等
        self.assertEqual([(96, 0), (97, 1), (98, 2), (99, 0)], unpacked_batch)

    # 如果没有数据框架则跳过测试，如果没有 dill 则也跳过
    @skipIfNoDataFrames
    @skipIfNoDill
    def test_unbatch(self):
        # 将 df_numbers 进行分批处理，每批大小为 8，再将结果再次分批处理，每批大小为 3
        df_numbers = self._get_dataframes_pipe(range=100).batch(8).batch(3)
        dp_numbers = self._get_datapipe(range=100)
        # 断言数字数据流 dp_numbers 与数据框架 df_numbers 解批处理（分批处理逆操作）后结果相等
        self.assertEqual(list(dp_numbers), list(df_numbers.unbatch(2)))
    # 定义一个测试方法，用于测试数据筛选功能
    def test_filter(self):
        # 从数据管道获取数据框架，使用 lambda 函数筛选出 i 列大于 5 的行
        df_numbers = self._get_dataframes_pipe(range=10).filter(lambda x: x.i > 5)
        # 将筛选后的结果转换为列表
        actual = list(df_numbers)
        # 断言实际结果与预期结果是否一致
        self.assertEqual([(6, 0), (7, 1), (8, 2), (9, 0)], actual)

    # 在有数据框架和 Dill 库的情况下执行测试
    @skipIfNoDataFrames
    @skipIfNoDill
    # 定义一个测试方法，用于测试数据整合功能
    def test_collate(self):
        # 定义用于整合 i 列的函数
        def collate_i(column):
            return column.sum()

        # 定义用于整合 j 列的函数
        def collate_j(column):
            return column.prod()

        # 从数据管道获取数据框架，将数据分组为批次为 3 的数据框架对象
        df_numbers = self._get_dataframes_pipe(range(30)).batch(3)
        # 使用 collate 方法整合数据框架中的 j 和 i 列
        df_numbers = df_numbers.collate({"j": collate_j, "i": collate_i})

        # 预期的 i 列整合结果列表
        expected_i = [
            3,
            12,
            21,
            30,
            39,
            48,
            57,
            66,
            75,
            84,
        ]

        # 存储实际 i 列整合结果的列表
        actual_i = []
        # 遍历数据框架对象，将 i 列整合结果添加到实际结果列表中
        for i, j in df_numbers:
            actual_i.append(i)
        # 断言实际的 i 列整合结果与预期结果是否一致
        self.assertEqual(expected_i, actual_i)

        # 清空实际 i 列整合结果的列表
        actual_i = []
        # 遍历数据框架对象，将每个元素的 i 属性值添加到实际结果列表中
        for item in df_numbers:
            actual_i.append(item.i)
        # 断言实际的 i 列整合结果与预期结果是否一致
        self.assertEqual(expected_i, actual_i)
class IDP_NoLen(IterDataPipe):
    # 初始化方法，继承自IterDataPipe类，接收一个输入数据管道对象input_dp
    def __init__(self, input_dp):
        super().__init__()
        self.input_dp = input_dp

    # 迭代器方法，防止原地修改数据
    def __iter__(self):
        # 如果input_dp是IterDataPipe的实例，则直接使用；否则深拷贝一份input_dp
        input_dp = (
            self.input_dp
            if isinstance(self.input_dp, IterDataPipe)
            else copy.deepcopy(self.input_dp)
        )
        # 使用生成器从input_dp中产生数据
        yield from input_dp


# 一个伪造的函数，返回其输入数据
def _fake_fn(data):
    return data


# 一个伪造的函数，将一个常数和数据相加并返回结果
def _fake_add(constant, data):
    return constant + data


# 一个伪造的过滤函数，总是返回True
def _fake_filter_fn(data):
    return True


# 一个简单的过滤函数，返回数据是否大于等于5的布尔值
def _simple_filter_fn(data):
    return data >= 5


# 一个带常数参数的过滤函数，返回数据是否大于等于常数的布尔值
def _fake_filter_fn_constant(constant, data):
    return data >= constant


# 乘以10的函数
def _mul_10(x):
    return x * 10


# 模3余1的测试函数
def _mod_3_test(x):
    return x % 3 == 1


# 将输入数据转换为列表的函数
def _to_list(x):
    return [x]


# 一个匿名函数，返回其输入值本身
lambda_fn1 = lambda x: x  # noqa: E731

# 一个匿名函数，返回输入值除以2的余数
lambda_fn2 = lambda x: x % 2  # noqa: E731

# 一个匿名函数，返回输入值是否大于等于5的布尔值
lambda_fn3 = lambda x: x >= 5  # noqa: E731


# 添加1的PyTorch模块
class Add1Module(nn.Module):
    # 前向传播方法，将输入数据加1并返回结果
    def forward(self, x):
        return x + 1


# 添加1的可调用类
class Add1Callable:
    # 调用方法，将输入数据加1并返回结果
    def __call__(self, x):
        return x + 1


# 测试FunctionalIterDataPipe的测试用例类
class TestFunctionalIterDataPipe(TestCase):
    # 序列化测试辅助函数，测试序列化和反序列化后数据管道的一致性
    def _serialization_test_helper(self, datapipe, use_dill):
        # 根据use_dill选择使用dill或pickle进行序列化和反序列化
        if use_dill:
            serialized_dp = dill.dumps(datapipe)
            deserialized_dp = dill.loads(serialized_dp)
        else:
            serialized_dp = pickle.dumps(datapipe)
            deserialized_dp = pickle.loads(serialized_dp)
        try:
            # 比较原始数据管道和反序列化后数据管道的内容是否一致
            self.assertEqual(list(datapipe), list(deserialized_dp))
        except AssertionError as e:
            # 如果不一致，则打印失败信息并抛出异常
            print(f"{datapipe} is failing.")
            raise e

    # 对单个数据管道进行序列化测试的方法
    def _serialization_test_for_single_dp(self, dp, use_dill=False):
        # 1. 在任何迭代开始之前测试序列化
        self._serialization_test_helper(dp, use_dill)
        # 2. 部分数据被读取后测试序列化
        it = iter(dp)
        _ = next(it)
        self._serialization_test_helper(dp, use_dill)
        # 3. 完全读取数据后测试序列化
        it = iter(dp)
        _ = list(it)
        self._serialization_test_helper(dp, use_dill)
    # 1. 测试在任何迭代开始之前序列化情况
    self._serialization_test_helper(dp1, use_dill)
    self._serialization_test_helper(dp2, use_dill)

    # 2. 在部分读取数据管道后测试序列化
    it1, it2 = iter(dp1), iter(dp2)
    _, _ = next(it1), next(it2)
    # 捕获 `fork`、`demux` 的警告：“某些子数据管道未被完全消耗”
    with warnings.catch_warnings(record=True) as wa:
        self._serialization_test_helper(dp1, use_dill)
        self._serialization_test_helper(dp2, use_dill)

    # 2.5. 在一个子数据管道完全读取后测试序列化
    #      (仅适用于带有子数据管道的数据管道)
    it1 = iter(dp1)
    _ = list(it1)  # 完全读取一个子数据管道
    # 捕获 `fork`、`demux` 的警告：“某些子数据管道未被完全消耗”
    with warnings.catch_warnings(record=True) as wa:
        self._serialization_test_helper(dp1, use_dill)
        self._serialization_test_helper(dp2, use_dill)

    # 3. 在数据管道完全读取后测试序列化
    it2 = iter(dp2)
    _ = list(it2)  # 完全读取另一个子数据管道
    self._serialization_test_helper(dp1, use_dill)
    self._serialization_test_helper(dp2, use_dill)
    # 定义测试方法，用于测试序列化功能
    def test_serializable(self):
        # 定义可被序列化的数据管道列表
        picklable_datapipes: List = [
            (
                dp.iter.Batcher,  # 数据管道类型：批处理器
                None,  # 自定义输入：无
                (
                    3,  # 参数1: 批处理大小
                    True,  # 参数2: 是否随机打乱
                ),
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Collator,  # 数据管道类型：整合器
                None,  # 自定义输入：无
                (_fake_fn,),  # 参数1: 虚拟函数
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Concater,  # 数据管道类型：串联器
                None,  # 自定义输入：无
                (dp.iter.IterableWrapper(range(5)),),  # 参数1: 可迭代对象包装器
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Demultiplexer,  # 数据管道类型：解复用器
                None,  # 自定义输入：无
                (2, _simple_filter_fn),  # 参数1: 数量, 参数2: 简单过滤函数
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.FileLister,  # 数据管道类型：文件列表生成器
                ".",  # 自定义输入：当前目录
                (),  # 参数: 空元组
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.FileOpener,  # 数据管道类型：文件打开器
                None,  # 自定义输入：无
                (),  # 参数: 空元组
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Filter,  # 数据管道类型：过滤器
                None,  # 自定义输入：无
                (_fake_filter_fn,),  # 参数1: 虚拟过滤函数
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Filter,  # 数据管道类型：过滤器
                None,  # 自定义输入：无
                (partial(_fake_filter_fn_constant, 5),),  # 参数1: 带常量参数的虚拟过滤函数
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Forker,  # 数据管道类型：分叉器
                None,  # 自定义输入：无
                (2,),  # 参数1: 数量
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Forker,  # 数据管道类型：分叉器
                None,  # 自定义输入：无
                (2,),  # 参数1: 数量
                {"copy": "shallow"},  # 额外关键字参数: 浅拷贝
            ),
            (
                dp.iter.Grouper,  # 数据管道类型：分组器
                None,  # 自定义输入：无
                (_fake_filter_fn,),  # 参数1: 虚拟过滤函数
                {"group_size": 2},  # 额外关键字参数: 分组大小为2
            ),
            (
                dp.iter.IterableWrapper,  # 数据管道类型：可迭代对象包装器
                range(10),  # 自定义输入：范围为0到9的可迭代对象
                (),  # 参数: 空元组
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Mapper,  # 数据管道类型：映射器
                None,  # 自定义输入：无
                (_fake_fn,),  # 参数1: 虚拟函数
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Mapper,  # 数据管道类型：映射器
                None,  # 自定义输入：无
                (partial(_fake_add, 1),),  # 参数1: 带常量参数的虚拟加法函数
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Multiplexer,  # 数据管道类型：多路复用器
                None,  # 自定义输入：无
                (dp.iter.IterableWrapper(range(10)),),  # 参数1: 可迭代对象包装器
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Sampler,  # 数据管道类型：采样器
                None,  # 自定义输入：无
                (),  # 参数: 空元组
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Shuffler,  # 数据管道类型：随机重排器
                dp.iter.IterableWrapper([0] * 10),  # 自定义输入：长度为10的可迭代对象包装器
                (),  # 参数: 空元组
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.StreamReader,  # 数据管道类型：流读取器
                None,  # 自定义输入：无
                (),  # 参数: 空元组
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.UnBatcher,  # 数据管道类型：取消批处理器
                None,  # 自定义输入：无
                (0,),  # 参数1: 数量
                {},  # 额外关键字参数: 空字典
            ),
            (
                dp.iter.Zipper,  # 数据管道类型：压缩器
                None,  # 自定义输入：无
                (dp.iter.IterableWrapper(range(10)),),  # 参数1: 可迭代对象包装器
                {},  # 额外关键字参数: 空字典
            ),
        ]
        
        # 跳过这些数据管道的比较
        dp_skip_comparison = {dp.iter.FileOpener, dp.iter.StreamReader}
        
        # 需要比较这些数据管道的子管道
        dp_compare_children = {dp.iter.Demultiplexer, dp.iter.Forker}

        # 遍历所有可序列化的数据管道
        for dpipe, custom_input, dp_args, dp_kwargs in picklable_datapipes:
            # 如果自定义输入为None，则使用范围为0到9的可迭代对象包装器作为输入
            if custom_input is None:
                custom_input = dp.iter.IterableWrapper(range(10))
            
            # 如果数据管道属于跳过比较的类型
            if dpipe in dp_skip_comparison:
                # 仅确保它们可以被序列化和加载（不进行值比较）
                datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                serialized_dp = pickle.dumps(datapipe)
                _ = pickle.loads(serialized_dp)
            
            # 如果数据管道属于需要比较子管道的类型
            elif dpipe in dp_compare_children:
                # 创建两个数据管道实例，并进行子管道比较测试
                dp1, dp2 = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                self._serialization_test_for_dp_with_children(dp1, dp2)
            
            # 否则，对单个数据管道进行比较测试
            else:
                datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                self._serialization_test_for_single_dp(datapipe)

    # 在这里跳过 Torch Dynamo 相关的测试
    @skipIfTorchDynamo("Dict with function as keys")
    def test_serializable_with_dill(self):
        """定义一个测试方法，用于测试支持使用 dill 序列化的 DataPipes"""
        # 创建一个 IterableWrapper 对象，包装一个包含 0 到 9 的迭代器
        input_dp = dp.iter.IterableWrapper(range(10))

        # 定义一个包含 lambda 函数作为参数的 DataPipes 列表
        datapipes_with_lambda_fn: List[
            Tuple[Type[IterDataPipe], Tuple, Dict[str, Any]]
        ] = [
            # Collator 类型的 DataPipe，参数是 lambda_fn1
            (dp.iter.Collator, (lambda_fn1,), {}),
            # Demultiplexer 类型的 DataPipe，参数是 2 和 lambda_fn2
            (
                dp.iter.Demultiplexer,
                (
                    2,
                    lambda_fn2,
                ),
                {},
            ),
            # Filter 类型的 DataPipe，参数是 lambda_fn3
            (dp.iter.Filter, (lambda_fn3,), {}),
            # Grouper 类型的 DataPipe，参数是 lambda_fn3
            (dp.iter.Grouper, (lambda_fn3,), {}),
            # Mapper 类型的 DataPipe，参数是 lambda_fn1
            (dp.iter.Mapper, (lambda_fn1,), {}),
        ]

        # 定义本地函数并赋值给 fn1, fn2, fn3
        def _local_fns():
            def _fn1(x):
                return x

            def _fn2(x):
                return x % 2

            def _fn3(x):
                return x >= 5

            return _fn1, _fn2, _fn3

        # 调用 _local_fns 函数获取本地函数 fn1, fn2, fn3
        fn1, fn2, fn3 = _local_fns()

        # 定义一个包含本地函数作为参数的 DataPipes 列表
        datapipes_with_local_fn: List[
            Tuple[Type[IterDataPipe], Tuple, Dict[str, Any]]
        ] = [
            # Collator 类型的 DataPipe，参数是 fn1
            (dp.iter.Collator, (fn1,), {}),
            # Demultiplexer 类型的 DataPipe，参数是 2 和 fn2
            (
                dp.iter.Demultiplexer,
                (
                    2,
                    fn2,
                ),
                {},
            ),
            # Filter 类型的 DataPipe，参数是 fn3
            (dp.iter.Filter, (fn3,), {}),
            # Grouper 类型的 DataPipe，参数是 fn3
            (dp.iter.Grouper, (fn3,), {}),
            # Mapper 类型的 DataPipe，参数是 fn1
            (dp.iter.Mapper, (fn1,), {}),
        ]

        # 创建一个集合，包含 Demultiplexer 类型的 DataPipe
        dp_compare_children = {dp.iter.Demultiplexer}

        # 如果支持 dill 库
        if HAS_DILL:
            # 遍历 lambda 函数和本地函数 DataPipes 列表
            for dpipe, dp_args, dp_kwargs in (
                datapipes_with_lambda_fn + datapipes_with_local_fn
            ):
                # 如果当前 DataPipe 类型在 dp_compare_children 中
                if dpipe in dp_compare_children:
                    # 创建两个 DataPipe 对象 dp1 和 dp2，使用 dill 进行序列化测试
                    dp1, dp2 = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    self._serialization_test_for_dp_with_children(
                        dp1, dp2, use_dill=True
                    )
                else:
                    # 创建单个 DataPipe 对象 datapipe，使用 dill 进行序列化测试
                    datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    self._serialization_test_for_single_dp(datapipe, use_dill=True)
        else:
            # 如果不支持 dill 库，定义警告消息
            msgs = (
                r"^Lambda function is not supported by pickle",
                r"^Local function is not supported by pickle",
            )
            # 遍历 lambda 函数和本地函数 DataPipes 列表
            for dps, msg in zip(
                (datapipes_with_lambda_fn, datapipes_with_local_fn), msgs
            ):
                # 遍历每个 DataPipe 元组
                for dpipe, dp_args, dp_kwargs in dps:
                    # 断言警告消息
                    with self.assertWarnsRegex(UserWarning, msg):
                        datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    # 断言 PicklingError 或 AttributeError 异常
                    with self.assertRaises((pickle.PicklingError, AttributeError)):
                        pickle.dumps(datapipe)
    def test_docstring(self):
        """
        Ensure functional form of IterDataPipe has the correct docstring from
        the class form.

        Regression test for https://github.com/pytorch/data/issues/792.
        """
        # 创建一个包装了范围为10的可迭代对象的IterDataPipe实例
        input_dp = dp.iter.IterableWrapper(range(10))

        # 遍历测试函数名列表
        for dp_funcname in [
            "batch",        # 批处理操作
            "collate",      # 整理操作
            "concat",       # 连接操作
            "demux",        # 解复用操作
            "filter",       # 过滤操作
            "fork",         # 分叉操作
            "map",          # 映射操作
            "mux",          # 多路复用操作
            "read_from_stream",  # 从流中读取操作
            # "sampler",   # 采样器操作（已注释掉）
            "shuffle",      # 洗牌操作
            "unbatch",      # 解批处理操作
            "zip",          # 压缩操作
        ]:
            # 根据Python版本选择不同的方式获取文档字符串
            if sys.version_info >= (3, 9):
                docstring = pydoc.render_doc(
                    thing=getattr(input_dp, dp_funcname), forceload=True
                )
            elif sys.version_info < (3, 9):
                # 在Python 3.8上使用不同的pydoc方式
                # 参见 https://docs.python.org/3/whatsnew/3.9.html#pydoc
                docstring = getattr(input_dp, dp_funcname).__doc__

            # 断言确保文档字符串包含功能名称
            assert f"(functional name: ``{dp_funcname}``)" in docstring
            # 断言确保文档字符串包含"Args:"部分
            assert "Args:" in docstring
            # 断言确保文档字符串包含"Example:"或"Examples:"部分
            assert "Example:" in docstring or "Examples:" in docstring

    def test_iterable_wrapper_datapipe(self):
        # 创建一个包含0到9的列表
        input_ls = list(range(10))
        # 创建一个包装了输入列表的IterableWrapper对象
        input_dp = dp.iter.IterableWrapper(input_ls)

        # 功能测试: 确保值保持不变且顺序不变
        self.assertEqual(input_ls, list(input_dp))

        # 功能测试: 默认情况下在初始化迭代器时进行深拷贝（只有在读取第一个元素时才会发生深拷贝）
        it = iter(input_dp)
        self.assertEqual(
            0, next(it)
        )  # 只有在读取第一个元素时才会进行深拷贝
        input_ls.append(50)
        self.assertEqual(list(range(1, 10)), list(it))

        # 功能测试: 浅拷贝
        input_ls2 = [1, 2, 3]
        input_dp_shallow = dp.iter.IterableWrapper(input_ls2, deepcopy=False)
        input_ls2.append(10)
        self.assertEqual([1, 2, 3, 10], list(input_dp_shallow))

        # 重置测试: 重置DataPipe
        input_ls = list(range(10))
        input_dp = dp.iter.IterableWrapper(input_ls)
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            input_dp, n_elements_before_reset
        )
        self.assertEqual(input_ls[:n_elements_before_reset], res_before_reset)
        self.assertEqual(input_ls, res_after_reset)

        # __len__测试: 从序列继承长度
        self.assertEqual(len(input_ls), len(input_dp))
    # 定义一个测试方法，用于测试数据管道的连接功能
    def test_concat_iterdatapipe(self):
        # 创建两个包装了整数范围的迭代器数据管道对象
        input_dp1 = dp.iter.IterableWrapper(range(10))
        input_dp2 = dp.iter.IterableWrapper(range(5))

        # 功能测试：对空输入抛出异常
        with self.assertRaisesRegex(ValueError, r"Expected at least one DataPipe"):
            dp.iter.Concater()

        # 功能测试：对非IterDataPipe输入抛出异常
        with self.assertRaisesRegex(
            TypeError, r"Expected all inputs to be `IterDataPipe`"
        ):
            dp.iter.Concater(input_dp1, ())  # type: ignore[arg-type]

        # 功能测试：按预期连接数据管道
        concat_dp = input_dp1.concat(input_dp2)
        self.assertEqual(len(concat_dp), 15)  # 断言连接后数据管道的长度为15
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))  # 断言连接后数据管道的内容与预期一致

        # 重置测试：重置数据管道后进行测试
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            concat_dp, n_elements_before_reset
        )
        self.assertEqual(list(range(5)), res_before_reset)  # 断言重置前5个元素与预期一致
        self.assertEqual(list(range(10)) + list(range(5)), res_after_reset)  # 断言重置后数据管道的内容与预期一致

        # __len__测试：继承源数据管道的长度
        input_dp_nl = IDP_NoLen(range(5))
        concat_dp = input_dp1.concat(input_dp_nl)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(concat_dp)  # 断言对没有有效长度的数据管道调用len会抛出异常

        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))  # 断言连接后数据管道的内容与预期一致
    # 定义测试方法 test_mux_iterdatapipe，用于测试数据管道的多路复用功能
    def test_mux_iterdatapipe(self):
        # Functional Test: Elements are yielded one at a time from each DataPipe, until they are all exhausted
        # 创建三个可迭代包装器，模拟数据管道，分别包含不同的整数范围
        input_dp1 = dp.iter.IterableWrapper(range(4))
        input_dp2 = dp.iter.IterableWrapper(range(4, 8))
        input_dp3 = dp.iter.IterableWrapper(range(8, 12))
        # 将三个数据管道进行多路复用
        output_dp = input_dp1.mux(input_dp2, input_dp3)
        # 预期输出结果列表
        expected_output = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        # 断言：输出数据管道的长度与预期输出长度相等
        self.assertEqual(len(expected_output), len(output_dp))
        # 断言：输出数据管道的内容与预期输出内容列表相等
        self.assertEqual(expected_output, list(output_dp))

        # Functional Test: Uneven input Data Pipes
        # 创建不均匀的数据管道输入
        input_dp1 = dp.iter.IterableWrapper([1, 2, 3, 4])
        input_dp2 = dp.iter.IterableWrapper([10])
        input_dp3 = dp.iter.IterableWrapper([100, 200, 300])
        # 将不均匀的数据管道进行多路复用
        output_dp = input_dp1.mux(input_dp2, input_dp3)
        # 预期输出结果列表
        expected_output = [1, 10, 100]
        # 断言：输出数据管道的长度与预期输出长度相等
        self.assertEqual(len(expected_output), len(output_dp))
        # 断言：输出数据管道的内容与预期输出内容列表相等
        self.assertEqual(expected_output, list(output_dp))

        # Functional Test: Empty Data Pipe
        # 创建一个空数据管道作为输入
        input_dp1 = dp.iter.IterableWrapper([0, 1, 2, 3])
        input_dp2 = dp.iter.IterableWrapper([])
        # 将空数据管道进行多路复用
        output_dp = input_dp1.mux(input_dp2)
        # 断言：输出数据管道的长度与输入空数据管道的长度相等
        self.assertEqual(len(input_dp2), len(output_dp))
        # 断言：输出数据管道的内容与输入空数据管道的内容列表相等
        self.assertEqual(list(input_dp2), list(output_dp))

        # __len__ Test: raises TypeError when __len__ is called and an input doesn't have __len__
        # 测试在调用 __len__ 方法时，当输入数据管道没有 __len__ 方法时会引发 TypeError
        input_dp1 = dp.iter.IterableWrapper(range(10))
        input_dp_no_len = IDP_NoLen(range(10))  # 带有 __len__ 方法的数据管道
        output_dp = input_dp1.mux(input_dp_no_len)
        # 使用断言检查是否会引发 TypeError 异常
        with self.assertRaises(TypeError):
            len(output_dp)
    # 定义测试方法 test_map_iterdatapipe，用于测试数据管道的映射功能
    def test_map_iterdatapipe(self):
        # 设定目标长度为 10
        target_length = 10
        # 创建一个迭代器包装器，包含从 0 到目标长度减一的整数
        input_dp = dp.iter.IterableWrapper(range(target_length))

        # 定义一个函数 fn，用于将输入转换为 torch 张量，默认为浮点型，可选是否求和
        def fn(item, dtype=torch.float, *, sum=False):
            data = torch.tensor(item, dtype=dtype)
            return data if not sum else data.sum()

        # Functional Test: apply to each element correctly
        # 对 input_dp 中的每个元素应用 fn 函数，并检查结果是否正确
        map_dp = input_dp.map(fn)
        self.assertEqual(target_length, len(map_dp))
        for x, y in zip(map_dp, range(target_length)):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

        # Functional Test: works with partial function
        # 使用部分函数应用 fn，指定数据类型为整数，并求和
        map_dp = input_dp.map(partial(fn, dtype=torch.int, sum=True))
        for x, y in zip(map_dp, range(target_length)):
            self.assertEqual(x, torch.tensor(y, dtype=torch.int).sum())

        # __len__ Test: inherits length from source DataPipe
        # 检查 map_dp 的长度是否与源数据管道长度一致
        self.assertEqual(target_length, len(map_dp))

        # 创建一个没有定义 __len__ 方法的数据管道，并尝试映射操作
        input_dp_nl = IDP_NoLen(range(target_length))
        map_dp_nl = input_dp_nl.map(lambda x: x)
        for x, y in zip(map_dp_nl, range(target_length)):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

        # __len__ Test: inherits length from source DataPipe - raises error when invalid
        # 检查没有定义 __len__ 方法的数据管道，调用 len(map_dp_nl) 是否引发 TypeError 错误
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(map_dp_nl)

        # Reset Test: DataPipe resets properly
        # 测试数据管道的重置功能是否正常工作
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            map_dp, n_elements_before_reset
        )
        self.assertEqual(list(range(n_elements_before_reset)), res_before_reset)
        self.assertEqual(list(range(10)), res_after_reset)

    @suppress_warnings  # Suppress warning for lambda fn
    @suppress_warnings  # Suppress warning for lambda fn
    @skipIfTorchDynamo()
    # 定义一个测试函数，用于测试数据管道中的数据整合功能
    def test_collate_iterdatapipe(self):
        # 创建一个包含多个列表的数组作为测试数据
        arrs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # 使用IterableWrapper将数组包装成数据管道的输入对象
        input_dp = dp.iter.IterableWrapper(arrs)

        # 定义一个数据整合函数，将批次数据求和并转换为指定类型的张量
        def _collate_fn(batch, default_type=torch.float):
            return torch.tensor(sum(batch), dtype=default_type)

        # 功能测试：当未指定自定义整合函数时，默认使用默认整合函数
        collate_dp = input_dp.collate()
        # 遍历原始数组和整合后的数据管道，验证数据是否相等
        for x, y in zip(arrs, collate_dp):
            self.assertEqual(torch.tensor(x), y)

        # 功能测试：使用自定义的数据整合函数
        collate_dp = input_dp.collate(collate_fn=_collate_fn)
        # 遍历原始数组和整合后的数据管道，验证数据是否按预期转换
        for x, y in zip(arrs, collate_dp):
            self.assertEqual(torch.tensor(sum(x), dtype=torch.float), y)

        # 功能测试：使用自定义的部分参数化数据整合函数
        collate_dp = input_dp.collate(partial(_collate_fn, default_type=torch.int))
        # 遍历原始数组和整合后的数据管道，验证数据是否按预期转换
        for x, y in zip(arrs, collate_dp):
            self.assertEqual(torch.tensor(sum(x), dtype=torch.int), y)

        # 重置测试：重置数据管道后，验证结果是否仍然正确
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            collate_dp, n_elements_before_reset
        )
        # 验证重置前后的结果是否符合预期
        self.assertEqual([torch.tensor(6, dtype=torch.int)], res_before_reset)
        for x, y in zip(arrs, res_after_reset):
            self.assertEqual(torch.tensor(sum(x), dtype=torch.int), y)

        # __len__ 测试：验证整合后的数据管道是否继承了原始数据管道的长度
        self.assertEqual(len(input_dp), len(collate_dp))

        # __len__ 测试：验证当原始数据管道没有有效的长度时，整合后的数据管道是否抛出预期的异常
        input_dp_nl = IDP_NoLen(arrs)
        collate_dp_nl = input_dp_nl.collate()
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(collate_dp_nl)
        # 遍历原始数组和无长度信息的整合后数据管道，验证数据是否相等
        for x, y in zip(arrs, collate_dp_nl):
            self.assertEqual(torch.tensor(x), y)
    # 定义测试方法 test_batch_iterdatapipe，用于测试批处理数据管道的功能
    def test_batch_iterdatapipe(self):
        # 创建一个包含 0 到 9 的列表
        arrs = list(range(10))
        # 使用 IterableWrapper 将列表 arrs 包装成数据管道的输入
        input_dp = dp.iter.IterableWrapper(arrs)

        # 功能测试：当 batch_size = 0 时，应该抛出 AssertionError 错误
        with self.assertRaises(AssertionError):
            input_dp.batch(batch_size=0)

        # 功能测试：默认情况下，不丢弃最后一个批次
        bs = 3
        batch_dp = input_dp.batch(batch_size=bs)
        # 断言批处理后的数据管道长度为 4
        self.assertEqual(len(batch_dp), 4)
        # 遍历批处理后的数据管道，验证每个批次的长度和内容
        for i, batch in enumerate(batch_dp):
            self.assertEqual(len(batch), 1 if i == 3 else bs)
            self.assertEqual(batch, arrs[i * bs : i * bs + len(batch)])

        # 功能测试：当指定丢弃最后一个批次时
        bs = 4
        batch_dp = input_dp.batch(batch_size=bs, drop_last=True)
        # 遍历批处理后的数据管道，验证每个批次的内容
        for i, batch in enumerate(batch_dp):
            self.assertEqual(batch, arrs[i * bs : i * bs + len(batch)])

        # __len__ 测试：验证整体长度和每个批次的长度是否正确
        for i, batch in enumerate(batch_dp):
            self.assertEqual(len(batch), bs)

        # __len__ 测试：如果源数据管道没有长度信息，应该抛出 TypeError 错误
        self.assertEqual(len(batch_dp), 2)
        # 创建一个没有长度信息的数据管道
        input_dp_nl = IDP_NoLen(range(10))
        batch_dp_nl = input_dp_nl.batch(batch_size=2)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(batch_dp_nl)

        # 重置测试：确保数据管道可以正确重置
        n_elements_before_reset = 1
        # 使用 reset_after_n_next_calls 函数对数据管道进行重置测试
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            batch_dp, n_elements_before_reset
        )
        # 断言重置前后的结果是否符合预期
        self.assertEqual([[0, 1, 2, 3]], res_before_reset)
        self.assertEqual([[0, 1, 2, 3], [4, 5, 6, 7]], res_after_reset)
    # 定义一个测试函数 test_unbatch_iterdatapipe，用于测试 unbatch 方法
    def test_unbatch_iterdatapipe(self):
        # 目标长度设定为6
        target_length = 6
        # 创建一个包含目标长度范围的可迭代对象 prebatch_dp
        prebatch_dp = dp.iter.IterableWrapper(range(target_length))

        # 功能测试：对数据管道进行解批处理后应与预先批处理的数据管道相同
        input_dp = prebatch_dp.batch(3)
        # 执行 unbatch 操作
        unbatch_dp = input_dp.unbatch()
        # 断言解批处理后的数据管道长度与目标长度相同
        self.assertEqual(len(list(unbatch_dp)), target_length)  # __len__ is as expected
        # 遍历解批处理后的数据管道，验证每个元素与预期相符
        for i, res in zip(range(target_length), unbatch_dp):
            self.assertEqual(i, res)

        # 功能测试：对具有嵌套级别的输入进行 unbatch 操作
        input_dp = dp.iter.IterableWrapper([[0, 1, 2], [3, 4, 5]])
        unbatch_dp = input_dp.unbatch()
        # 断言解批处理后的数据管道长度与目标长度相同
        self.assertEqual(len(list(unbatch_dp)), target_length)
        # 遍历解批处理后的数据管道，验证每个元素与预期相符
        for i, res in zip(range(target_length), unbatch_dp):
            self.assertEqual(i, res)

        input_dp = dp.iter.IterableWrapper([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        # 功能测试：对具有多层嵌套级别的输入进行 unbatch 操作
        unbatch_dp = input_dp.unbatch()
        expected_dp = [[0, 1], [2, 3], [4, 5], [6, 7]]
        # 断言解批处理后的数据管道长度与预期相符
        self.assertEqual(len(list(unbatch_dp)), 4)
        # 遍历解批处理后的数据管道，验证每个元素与预期相符
        for j, res in zip(expected_dp, unbatch_dp):
            self.assertEqual(j, res)

        # 功能测试：同时解批处理多个级别的输入
        unbatch_dp = input_dp.unbatch(unbatch_level=2)
        expected_dp2 = [0, 1, 2, 3, 4, 5, 6, 7]
        # 断言解批处理后的数据管道长度与预期相符
        self.assertEqual(len(list(unbatch_dp)), 8)
        # 遍历解批处理后的数据管道，验证每个元素与预期相符
        for i, res in zip(expected_dp2, unbatch_dp):
            self.assertEqual(i, res)

        # 功能测试：同时解批处理所有级别的输入
        unbatch_dp = input_dp.unbatch(unbatch_level=-1)
        # 断言解批处理后的数据管道长度与预期相符
        self.assertEqual(len(list(unbatch_dp)), 8)
        # 遍历解批处理后的数据管道，验证每个元素与预期相符
        for i, res in zip(expected_dp2, unbatch_dp):
            self.assertEqual(i, res)

        # 功能测试：当输入的解批处理级别小于-1时抛出错误
        input_dp = dp.iter.IterableWrapper([[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            unbatch_dp = input_dp.unbatch(unbatch_level=-2)
            for i in unbatch_dp:
                print(i)

        # 功能测试：当输入的解批处理级别过高时抛出错误
        with self.assertRaises(IndexError):
            unbatch_dp = input_dp.unbatch(unbatch_level=5)
            for i in unbatch_dp:
                print(i)

        # 重置测试：验证 unbatch_dp 正确重置
        input_dp = dp.iter.IterableWrapper([[0, 1, 2], [3, 4, 5]])
        unbatch_dp = input_dp.unbatch(unbatch_level=-1)
        # 设置在重置前获取的元素数
        n_elements_before_reset = 3
        # 调用 reset_after_n_next_calls 函数进行重置测试
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            unbatch_dp, n_elements_before_reset
        )
        # 断言重置前后的结果与预期相符
        self.assertEqual([0, 1, 2], res_before_reset)
        self.assertEqual([0, 1, 2, 3, 4, 5], res_after_reset)
    def test_filter_datapipe(self):
        # 创建一个包含0到9的可迭代对象
        input_ds = dp.iter.IterableWrapper(range(10))

        def _filter_fn(data, val):
            # 过滤函数：返回数据是否大于等于给定值
            return data >= val

        # 功能测试：使用部分函数检查过滤器功能
        filter_dp = input_ds.filter(partial(_filter_fn, val=5))
        # 断言过滤后的结果与预期的列表相同
        self.assertEqual(list(filter_dp), list(range(5, 10)))

        def _non_bool_fn(data):
            # 非布尔值返回函数：始终返回整数1
            return 1

        # 功能测试：过滤函数必须返回布尔值
        filter_dp = input_ds.filter(filter_fn=_non_bool_fn)
        # 断言抛出值错误异常
        with self.assertRaises(ValueError):
            temp = list(filter_dp)

        # 功能测试：指定输入列
        tuple_input_ds = dp.iter.IterableWrapper([(d - 1, d, d + 1) for d in range(10)])

        # 单个输入列
        input_col_1_dp = tuple_input_ds.filter(partial(_filter_fn, val=5), input_col=1)
        # 断言过滤后的结果与预期的列表相同
        self.assertEqual(
            list(input_col_1_dp), [(d - 1, d, d + 1) for d in range(5, 10)]
        )

        # 多个输入列
        def _mul_filter_fn(a, b):
            # 多条件过滤函数：返回两个输入相加是否小于10
            return a + b < 10

        input_col_2_dp = tuple_input_ds.filter(_mul_filter_fn, input_col=[0, 2])
        # 断言过滤后的结果与预期的列表相同
        self.assertEqual(list(input_col_2_dp), [(d - 1, d, d + 1) for d in range(5)])

        # 无效的输入列
        with self.assertRaises(ValueError):
            tuple_input_ds.filter(_mul_filter_fn, input_col=0)

        p_mul_filter_fn = partial(_mul_filter_fn, b=1)
        out = tuple_input_ds.filter(p_mul_filter_fn, input_col=0)
        # 断言过滤后的结果与预期的列表相同
        self.assertEqual(list(out), [(d - 1, d, d + 1) for d in range(10)])

        def _mul_filter_fn_with_defaults(a, b=1):
            # 带默认参数的多条件过滤函数：返回两个输入相加是否小于10
            return a + b < 10

        out = tuple_input_ds.filter(_mul_filter_fn_with_defaults, input_col=0)
        # 断言过滤后的结果与预期的列表相同
        self.assertEqual(list(out), [(d - 1, d, d + 1) for d in range(10)])

        def _mul_filter_fn_with_kw_only(*, a, b):
            # 仅关键字参数的多条件过滤函数：返回两个输入相加是否小于10
            return a + b < 10

        with self.assertRaises(ValueError):
            tuple_input_ds.filter(_mul_filter_fn_with_kw_only, input_col=0)

        def _mul_filter_fn_with_kw_only_1_default(*, a, b=1):
            # 带有一个默认值的仅关键字参数的多条件过滤函数：返回两个输入相加是否小于10
            return a + b < 10

        with self.assertRaises(ValueError):
            tuple_input_ds.filter(_mul_filter_fn_with_kw_only_1_default, input_col=0)

        # __len__ 测试：DataPipe 没有有效的长度
        with self.assertRaisesRegex(TypeError, r"has no len"):
            len(filter_dp)

        # 重置测试：DataPipe 正确重置
        filter_dp = input_ds.filter(partial(_filter_fn, val=5))
        n_elements_before_reset = 3
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            filter_dp, n_elements_before_reset
        )
        # 断言重置前的部分结果与预期的列表相同
        self.assertEqual(list(range(5, 10))[:n_elements_before_reset], res_before_reset)
        # 断言重置后的结果与预期的列表相同
        self.assertEqual(list(range(5, 10)), res_after_reset)
    def test_sampler_iterdatapipe(self):
        input_dp = dp.iter.IterableWrapper(range(10))
        # 创建默认的顺序采样器 SequentialSampler
        sampled_dp = dp.iter.Sampler(input_dp)  # type: ignore[var-annotated]
        self.assertEqual(len(sampled_dp), 10)
        for i, x in enumerate(sampled_dp):
            self.assertEqual(x, i)

        # 创建随机采样器 RandomSampler
        random_sampled_dp = dp.iter.Sampler(
            input_dp, sampler=RandomSampler, sampler_kwargs={"replacement": True}
        )  # type: ignore[var-annotated] # noqa: B950

        # 需要确保 input_dp 具有 `__len__` 方法以构建 SamplerDataPipe
        input_dp_nolen = IDP_NoLen(range(10))
        with self.assertRaises(AssertionError):
            sampled_dp = dp.iter.Sampler(input_dp_nolen)

    def test_stream_reader_iterdatapipe(self):
        from io import StringIO

        input_dp = dp.iter.IterableWrapper(
            [("f1", StringIO("abcde")), ("f2", StringIO("bcdef"))]
        )
        expected_res = ["abcde", "bcdef"]

        # 功能测试：读取完整的数据块
        dp1 = input_dp.read_from_stream()
        self.assertEqual([d[1] for d in dp1], expected_res)

        # 功能测试：按块读取数据
        dp2 = input_dp.read_from_stream(chunk=1)
        self.assertEqual([d[1] for d in dp2], [c for s in expected_res for c in s])

        # `__len__` 测试
        with self.assertRaises(TypeError):
            len(dp1)
    # 定义测试方法：测试数据管道的随机打乱功能
    def test_shuffler_iterdatapipe(self):
        # 创建一个包装了列表 [0, 1, ..., 9] 的可迭代数据管道
        input_dp = dp.iter.IterableWrapper(list(range(10)))

        # 断言异常：缓冲区大小为0时应抛出断言错误
        with self.assertRaises(AssertionError):
            shuffle_dp = input_dp.shuffle(buffer_size=0)

        # 功能测试：无种子值的随机打乱
        shuffler_dp = input_dp.shuffle()
        # 断言：打乱后数据与原始数据集合相同
        self.assertEqual(set(range(10)), set(shuffler_dp))

        # 功能测试：使用全局种子值的随机打乱
        torch.manual_seed(123)
        shuffler_dp = input_dp.shuffle()
        res = list(shuffler_dp)
        torch.manual_seed(123)
        # 断言：重新设置相同的种子值后，结果应与之前相同
        self.assertEqual(list(shuffler_dp), res)

        # 功能测试：设置特定种子值的随机打乱
        shuffler_dp = input_dp.shuffle().set_seed(123)
        res = list(shuffler_dp)
        shuffler_dp.set_seed(123)
        # 断言：重新设置相同的种子值后，结果应与之前相同
        self.assertEqual(list(shuffler_dp), res)

        # 功能测试：通过 set_shuffle 方法停用随机打乱
        unshuffled_dp = input_dp.shuffle().set_shuffle(False)
        # 断言：停用随机打乱后，数据应与原始数据一致
        self.assertEqual(list(unshuffled_dp), list(input_dp))

        # 重置测试：
        shuffler_dp = input_dp.shuffle()
        n_elements_before_reset = 5
        # 调用 reset_after_n_next_calls 函数，获取重置前和重置后的结果
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            shuffler_dp, n_elements_before_reset
        )
        # 断言：重置前的结果长度应为5
        self.assertEqual(5, len(res_before_reset))
        # 断言：重置前的结果中的每个元素应位于原始数据集合中
        for x in res_before_reset:
            self.assertTrue(x in set(range(10)))
        # 断言：重置后的结果应与原始数据集合相同
        self.assertEqual(set(range(10)), set(res_after_reset))

        # __len__ 测试：返回输入数据管道的长度
        shuffler_dp = input_dp.shuffle()
        # 断言：数据管道的长度应为10
        self.assertEqual(10, len(shuffler_dp))
        exp = list(range(100))

        # 序列化测试
        from torch.utils.data.datapipes._hook_iterator import _SnapshotState

        def _serialization_helper(bs):
            # 创建具有缓冲区大小为 bs 的随机打乱数据管道
            shuffler_dp = input_dp.shuffle(buffer_size=bs)
            it = iter(shuffler_dp)
            # 迭代两次数据管道
            for _ in range(2):
                next(it)
            # 深度复制数据管道对象
            shuffler_dp_copy = pickle.loads(pickle.dumps(shuffler_dp))
            # 恢复简单图形的快照状态
            _simple_graph_snapshot_restoration(
                shuffler_dp_copy.datapipe,
                shuffler_dp.datapipe._number_of_samples_yielded,
            )

            # 获取迭代器剩余部分的预期值
            exp = list(it)
            # 标记状态为已恢复
            shuffler_dp_copy._snapshot_state = _SnapshotState.Restored
            # 断言：复制的数据管道的结果应与原始数据管道的结果相同
            self.assertEqual(exp, list(shuffler_dp_copy))

        buffer_sizes = [2, 5, 15]
        # 对不同的缓冲区大小进行序列化辅助函数的测试
        for bs in buffer_sizes:
            _serialization_helper(bs)
    # 定义测试方法 test_zip_iterdatapipe
    def test_zip_iterdatapipe(self):
        # 功能测试: 当输入不是 `IterDataPipe` 类型时，预期会引发 TypeError 异常
        with self.assertRaises(TypeError):
            # 创建 Zipper 对象时，将一个 IterableWrapper 和一个列表作为参数传递
            dp.iter.Zipper(dp.iter.IterableWrapper(range(10)), list(range(10)))  # type: ignore[arg-type]

        # 功能测试: 当输入对象没有有效的长度时，预期会引发 TypeError 异常
        zipped_dp = dp.iter.Zipper(
            dp.iter.IterableWrapper(range(10)), IDP_NoLen(range(5))
        )  # type: ignore[var-annotated]
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(zipped_dp)

        # 功能测试: 正确地对结果进行压缩（zip）
        exp = [(i, i) for i in range(5)]
        self.assertEqual(list(zipped_dp), exp)

        # 功能测试: 即使输入长度不同（以最短长度进行压缩）
        zipped_dp = dp.iter.Zipper(
            dp.iter.IterableWrapper(range(10)), dp.iter.IterableWrapper(range(5))
        )

        # __len__ 测试: 对象的长度应与最短输入的长度相匹配
        self.assertEqual(len(zipped_dp), 5)

        # 重置测试:
        n_elements_before_reset = 3
        # 调用 reset_after_n_next_calls 函数，测试重置前后的返回值
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            zipped_dp, n_elements_before_reset
        )
        expected_res = [(i, i) for i in range(5)]
        # 断言前 n_elements_before_reset 个元素与预期结果相同
        self.assertEqual(expected_res[:n_elements_before_reset], res_before_reset)
        # 断言全部元素与预期结果相同
        self.assertEqual(expected_res, res_after_reset)
# 定义一个测试类，用于测试功能映射数据管道的行为
class TestFunctionalMapDataPipe(TestCase):

    # 辅助方法：用于测试序列化功能
    def _serialization_test_helper(self, datapipe, use_dill):
        # 如果使用 dill 序列化
        if use_dill:
            serialized_dp = dill.dumps(datapipe)  # 序列化数据管道对象
            deserialized_dp = dill.loads(serialized_dp)  # 反序列化数据管道对象
        else:
            serialized_dp = pickle.dumps(datapipe)  # 序列化数据管道对象
            deserialized_dp = pickle.loads(serialized_dp)  # 反序列化数据管道对象
        try:
            self.assertEqual(list(datapipe), list(deserialized_dp))  # 断言序列化前后数据一致性
        except AssertionError as e:
            print(f"{datapipe} is failing.")  # 打印出现异常的数据管道
            raise e

    # 辅助方法：对单个数据管道对象进行序列化测试
    def _serialization_test_for_single_dp(self, dp, use_dill=False):
        # 1. 在迭代开始之前测试序列化
        self._serialization_test_helper(dp, use_dill)
        # 2. 在部分读取数据管道后测试序列化
        it = iter(dp)
        _ = next(it)
        self._serialization_test_helper(dp, use_dill)
        # 3. 在完全读取数据管道后测试序列化
        _ = list(dp)
        self._serialization_test_helper(dp, use_dill)

    # 测试方法：测试数据管道对象是否可序列化
    def test_serializable(self):
        picklable_datapipes: List = [
            (dp.map.Batcher, None, (2,), {}),  # Batcher 数据管道的序列化测试参数
            (dp.map.Concater, None, (dp.map.SequenceWrapper(range(10)),), {}),  # Concater 数据管道的序列化测试参数
            (dp.map.Mapper, None, (), {}),  # Mapper 数据管道的序列化测试参数
            (dp.map.Mapper, None, (_fake_fn,), {}),  # Mapper 数据管道的序列化测试参数
            (dp.map.Mapper, None, (partial(_fake_add, 1),), {}),  # Mapper 数据管道的序列化测试参数
            (dp.map.SequenceWrapper, range(10), (), {}),  # SequenceWrapper 数据管道的序列化测试参数
            (dp.map.Shuffler, dp.map.SequenceWrapper([0] * 5), (), {}),  # Shuffler 数据管道的序列化测试参数
            (dp.map.Zipper, None, (dp.map.SequenceWrapper(range(10)),), {}),  # Zipper 数据管道的序列化测试参数
        ]
        # 对每个可序列化的数据管道进行测试
        for dpipe, custom_input, dp_args, dp_kwargs in picklable_datapipes:
            if custom_input is None:
                custom_input = dp.map.SequenceWrapper(range(10))
            datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # 创建数据管道对象
            self._serialization_test_for_single_dp(datapipe)  # 测试该数据管道对象的序列化
    def test_serializable_with_dill(self):
        """对于接受函数作为参数的 DataPipes 测试序列化是否正常"""
        # 创建一个 SequenceWrapper 对象，传入一个包含 0 到 9 的序列
        input_dp = dp.map.SequenceWrapper(range(10))

        # 包含 lambda 函数的 DataPipe 列表，每个元素是一个三元组 (数据管道类型, 参数元组, 参数字典)
        datapipes_with_lambda_fn: List[
            Tuple[Type[MapDataPipe], Tuple, Dict[str, Any]]
        ] = [
            (dp.map.Mapper, (lambda_fn1,), {}),
        ]

        # 定义本地函数并将其赋值给 fn1
        def _local_fns():
            def _fn1(x):
                return x

            return _fn1

        fn1 = _local_fns()

        # 包含本地函数的 DataPipe 列表，每个元素是一个三元组 (数据管道类型, 参数元组, 参数字典)
        datapipes_with_local_fn: List[
            Tuple[Type[MapDataPipe], Tuple, Dict[str, Any]]
        ] = [
            (dp.map.Mapper, (fn1,), {}),
        ]

        # 如果系统支持 dill 库
        if HAS_DILL:
            # 对每个数据管道进行序列化测试
            for dpipe, dp_args, dp_kwargs in (
                datapipes_with_lambda_fn + datapipes_with_local_fn
            ):
                # 使用 dill 序列化数据管道对象，忽略调用参数类型的警告
                _ = dill.dumps(dpipe(input_dp, *dp_args, **dp_kwargs))  # type: ignore[call-arg]
        else:
            # 如果系统不支持 dill 库，则验证警告消息
            msgs = (
                r"^Lambda function is not supported by pickle",
                r"^Local function is not supported by pickle",
            )
            # 对每个数据管道列表及其对应的消息进行迭代验证
            for dps, msg in zip(
                (datapipes_with_lambda_fn, datapipes_with_local_fn), msgs
            ):
                for dpipe, dp_args, dp_kwargs in dps:
                    # 使用断言检查是否发出了 UserWarning 警告消息
                    with self.assertWarnsRegex(UserWarning, msg):
                        datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    # 使用断言检查是否抛出了 pickle 序列化错误或属性错误
                    with self.assertRaises((pickle.PicklingError, AttributeError)):
                        pickle.dumps(datapipe)

    def test_docstring(self):
        """
        确保 MapDataPipe 的函数形式具有来自类形式的正确文档字符串。

        https://github.com/pytorch/data/issues/792 的回归测试。
        """
        # 创建一个 SequenceWrapper 对象，传入一个包含 0 到 9 的序列
        input_dp = dp.map.SequenceWrapper(range(10))

        # 对于每个数据管道函数名称，验证其文档字符串的正确性
        for dp_funcname in [
            "batch",
            "concat",
            "map",
            "shuffle",
            "zip",
        ]:
            # 如果系统版本大于等于 Python 3.9
            if sys.version_info >= (3, 9):
                # 使用 pydoc 渲染文档，强制加载，获取函数的文档字符串
                docstring = pydoc.render_doc(
                    thing=getattr(input_dp, dp_funcname), forceload=True
                )
            # 如果系统版本小于 Python 3.9
            elif sys.version_info < (3, 9):
                # 在 Python 3.8 上，pydoc 的工作方式不同，直接获取函数的文档字符串
                docstring = getattr(input_dp, dp_funcname).__doc__
            # 断言文档字符串包含功能名称信息
            assert f"(functional name: ``{dp_funcname}``)" in docstring
            assert "Args:" in docstring
            assert "Example:" in docstring or "Examples:" in docstring
    def test_sequence_wrapper_datapipe(self):
        # 创建一个包含0到9的列表
        seq = list(range(10))
        # 使用SequenceWrapper将列表seq转换为数据管道对象input_dp
        input_dp = dp.map.SequenceWrapper(seq)

        # 功能测试：确认所有元素以相同顺序相等
        self.assertEqual(seq, list(input_dp))

        # 功能测试：确认默认情况下深拷贝有效
        seq.append(11)
        self.assertEqual(list(range(10)), list(input_dp))  # input_dp 不应包含11

        # 功能测试：确认非深拷贝版本有效
        seq2 = [1, 2, 3]
        input_dp_non_deep = dp.map.SequenceWrapper(seq2, deepcopy=False)
        seq2.append(4)
        self.assertEqual(list(seq2), list(input_dp_non_deep))  # 应包含4

        # 重置测试：重置数据管道
        seq = list(range(10))
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            input_dp, n_elements_before_reset
        )
        self.assertEqual(list(range(5)), res_before_reset)
        self.assertEqual(seq, res_after_reset)

        # __len__ 测试：继承长度自序列
        self.assertEqual(len(seq), len(input_dp))

    def test_concat_mapdatapipe(self):
        # 创建两个SequenceWrapper对象
        input_dp1 = dp.map.SequenceWrapper(range(10))
        input_dp2 = dp.map.SequenceWrapper(range(5))

        # 测试空输入抛出值错误异常
        with self.assertRaisesRegex(ValueError, r"Expected at least one DataPipe"):
            dp.map.Concater()

        # 测试输入不是MapDataPipe类型抛出类型错误异常
        with self.assertRaisesRegex(
            TypeError, r"Expected all inputs to be `MapDataPipe`"
        ):
            dp.map.Concater(input_dp1, ())  # type: ignore[arg-type]

        # 将两个数据管道对象拼接为一个新的数据管道对象concat_dp
        concat_dp = input_dp1.concat(input_dp2)
        # 确认concat_dp的长度为15
        self.assertEqual(len(concat_dp), 15)
        # 遍历确认concat_dp中每个元素与预期列表的值相等
        for index in range(15):
            self.assertEqual(
                concat_dp[index], (list(range(10)) + list(range(5)))[index]
            )
        # 确认list(concat_dp)与预期的完整列表相等
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))
    # 定义一个测试方法，用于测试 Zip 数据管道的功能
    def test_zip_mapdatapipe(self):
        # 创建三个序列数据包装器作为输入数据管道
        input_dp1 = dp.map.SequenceWrapper(range(10))
        input_dp2 = dp.map.SequenceWrapper(range(5))
        input_dp3 = dp.map.SequenceWrapper(range(15))

        # 功能测试：至少需要一个输入数据管道
        with self.assertRaisesRegex(ValueError, r"Expected at least one DataPipe"):
            # 初始化 Zipper 对象时会抛出 ValueError 异常
            dp.map.Zipper()

        # 功能测试：所有输入必须是 MapDataPipe 类型
        with self.assertRaisesRegex(
            TypeError, r"Expected all inputs to be `MapDataPipe`"
        ):
            # 初始化 Zipper 对象时会抛出 TypeError 异常
            dp.map.Zipper(input_dp1, ())  # type: ignore[arg-type]

        # 功能测试：将输入数据管道的元素打包成元组
        zip_dp = input_dp1.zip(input_dp2, input_dp3)
        # 断言：打包后的数据与预期的数据相等
        self.assertEqual([(i, i, i) for i in range(5)], [zip_dp[i] for i in range(5)])

        # 功能测试：当索引超过最短数据管道长度时，应引发 IndexError
        with self.assertRaisesRegex(IndexError, r"out of range"):
            # 尝试访问超出范围的索引位置时会抛出 IndexError 异常
            input_dp1.zip(input_dp2, input_dp3)[5]

        # 功能测试：确保 `zip` 可以与 `Batcher` 结合使用
        dp1 = dp.map.SequenceWrapper(range(10))
        shuffle_dp1 = dp1.batch(2)
        dp2 = dp.map.SequenceWrapper(range(10))
        shuffle_dp2 = dp2.batch(3)
        zip_dp1 = shuffle_dp1.zip(shuffle_dp2)
        # 断言：组合后的数据管道长度与预期值相等
        self.assertEqual(4, len(list(zip_dp1)))
        zip_dp2 = shuffle_dp1.zip(dp2)
        # 断言：组合后的数据管道长度与预期值相等
        self.assertEqual(5, len(list(zip_dp2)))

        # __len__ 测试：返回最短数据管道的长度
        zip_dp = input_dp1.zip(input_dp2, input_dp3)
        # 断言：返回的数据管道长度与预期值相等
        self.assertEqual(5, len(zip_dp))
    def test_shuffler_mapdatapipe(self):
        # 创建一个序列包装器，包含整数范围为 0 到 9
        input_dp1 = dp.map.SequenceWrapper(range(10))
        # 创建一个序列包装器，包含键值对为 {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5} 的字典

        input_dp2 = dp.map.SequenceWrapper({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})

        # Functional Test: Assumes 0-index when indices is not given
        # 对 input_dp1 进行随机打乱顺序
        shuffler_dp = input_dp1.shuffle()
        # 断言打乱后的结果与集合 {0, 1, 2, ..., 9} 相同
        self.assertEqual(set(range(10)), set(shuffler_dp))

        # Functional Test: Custom indices are working
        # 使用自定义的索引顺序对 input_dp2 进行打乱
        shuffler_dp = input_dp2.shuffle(indices=["a", "b", "c", "d", "e"])
        # 断言打乱后的结果与集合 {1, 2, 3, 4, 5} 相同
        self.assertEqual(set(range(1, 6)), set(shuffler_dp))

        # Functional Test: With global seed
        # 设置全局随机种子为 123
        torch.manual_seed(123)
        # 再次对 input_dp1 进行随机打乱
        shuffler_dp = input_dp1.shuffle()
        # 获取打乱后的结果列表
        res = list(shuffler_dp)
        # 重新设置随机种子为 123
        torch.manual_seed(123)
        # 再次断言打乱后的结果与之前相同
        self.assertEqual(list(shuffler_dp), res)

        # Functional Test: Set seed
        # 对 input_dp1 进行打乱，并设置随机种子为 123
        shuffler_dp = input_dp1.shuffle().set_seed(123)
        # 获取打乱后的结果列表
        res = list(shuffler_dp)
        # 再次设置随机种子为 123
        shuffler_dp.set_seed(123)
        # 再次断言打乱后的结果与之前相同
        self.assertEqual(list(shuffler_dp), res)

        # Functional Test: deactivate shuffling via set_shuffle
        # 使用 set_shuffle(False) 禁用打乱功能
        unshuffled_dp = input_dp1.shuffle().set_shuffle(False)
        # 断言不打乱的结果与 input_dp1 的列表相同
        self.assertEqual(list(unshuffled_dp), list(input_dp1))

        # Reset Test:
        # 对 input_dp1 进行打乱
        shuffler_dp = input_dp1.shuffle()
        # 在调用 reset_after_n_next_calls 函数之前已经获取的元素数目为 5
        n_elements_before_reset = 5
        # 调用 reset_after_n_next_calls 函数，获取重置前和重置后的结果
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            shuffler_dp, n_elements_before_reset
        )
        # 断言重置前获取的元素数目为 5
        self.assertEqual(5, len(res_before_reset))
        # 断言重置前获取的元素都在集合 {0, 1, 2, ..., 9} 中
        for x in res_before_reset:
            self.assertTrue(x in set(range(10)))
        # 断言重置后获取的元素集合与 {0, 1, 2, ..., 9} 相同
        self.assertEqual(set(range(10)), set(res_after_reset))

        # __len__ Test: returns the length of the input DataPipe
        # 获取 input_dp1 经过打乱后的长度
        shuffler_dp = input_dp1.shuffle()
        # 断言打乱后的长度为 10
        self.assertEqual(10, len(shuffler_dp))

        # Serialization Test
        # 导入 _SnapshotState 类
        from torch.utils.data.datapipes._hook_iterator import _SnapshotState

        # 对 input_dp1 进行打乱
        shuffler_dp = input_dp1.shuffle()
        # 创建迭代器
        it = iter(shuffler_dp)
        # 遍历前两个元素
        for _ in range(2):
            next(it)
        # 使用 pickle 序列化和反序列化 shuffler_dp
        shuffler_dp_copy = pickle.loads(pickle.dumps(shuffler_dp))

        # 获取剩余未遍历的元素列表
        exp = list(it)
        # 将 shuffler_dp_copy 的状态设置为 _SnapshotState.Restored
        shuffler_dp_copy._snapshot_state = _SnapshotState.Restored
        # 断言 exp 与 shuffler_dp_copy 的元素列表相同
        self.assertEqual(exp, list(shuffler_dp_copy))

    def test_map_mapdatapipe(self):
        # 创建一个序列包装器，包含整数范围为 0 到 9
        arr = range(10)
        input_dp = dp.map.SequenceWrapper(arr)

        # 定义一个操作函数 fn，将 item 转换为 torch.tensor 类型为 dtype 的数据，如果 sum=True，则返回数据的和
        def fn(item, dtype=torch.float, *, sum=False):
            data = torch.tensor(item, dtype=dtype)
            return data if not sum else data.sum()

        # 对 input_dp 应用 fn 函数进行映射操作
        map_dp = input_dp.map(fn)
        # 断言映射后的数据管道 map_dp 的长度与 input_dp 相同
        self.assertEqual(len(input_dp), len(map_dp))
        # 遍历数组 arr
        for index in arr:
            # 断言 map_dp 的索引 index 的值与 torch.tensor(input_dp[index], dtype=torch.float) 相同
            self.assertEqual(
                map_dp[index], torch.tensor(input_dp[index], dtype=torch.float)
            )

        # 使用 partial 函数对 input_dp 应用 fn 函数进行映射操作，设置 dtype=torch.int 且 sum=True
        map_dp = input_dp.map(partial(fn, dtype=torch.int, sum=True))
        # 断言映射后的数据管道 map_dp 的长度与 input_dp 相同
        self.assertEqual(len(input_dp), len(map_dp))
        # 遍历数组 arr
        for index in arr:
            # 断言 map_dp 的索引 index 的值与 torch.tensor(input_dp[index], dtype=torch.int).sum() 相同
            self.assertEqual(
                map_dp[index], torch.tensor(input_dp[index], dtype=torch.int).sum()
            )
    # 定义测试方法：批处理数据管道的功能测试
    def test_batch_mapdatapipe(self):
        # 创建一个包含数字0到12的列表
        arr = list(range(13))
        # 使用SequenceWrapper将列表转换为数据管道的输入对象
        input_dp = dp.map.SequenceWrapper(arr)

        # Functional Test: batches top level by default
        # 创建批处理器对象，每批次大小为2，默认顶层批处理
        batch_dp = dp.map.Batcher(input_dp, batch_size=2)
        # 断言批处理结果是否符合预期
        self.assertEqual(
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12]], list(batch_dp)
        )

        # Functional Test: drop_last on command
        # 创建批处理器对象，每批次大小为2，设置丢弃最后一批次
        batch_dp = dp.map.Batcher(input_dp, batch_size=2, drop_last=True)
        # 断言批处理结果是否符合预期
        self.assertEqual(
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], list(batch_dp)
        )

        # Functional Test: nested batching
        # 对上一个批处理结果再进行一次批处理，每批次大小为3，实现嵌套批处理
        batch_dp_2 = batch_dp.batch(batch_size=3)
        # 断言嵌套批处理结果是否符合预期
        self.assertEqual(
            [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], list(batch_dp_2)
        )

        # Reset Test:
        # 测试重置功能：在调用指定次数后重置批处理器对象
        n_elements_before_reset = 3
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            batch_dp, n_elements_before_reset
        )
        # 断言重置前的部分结果是否符合预期
        self.assertEqual([[0, 1], [2, 3], [4, 5]], res_before_reset)
        # 断言重置后的完整结果是否符合预期
        self.assertEqual(
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], res_after_reset
        )

        # __len__ Test:
        # 测试数据管道的长度计算是否正确
        self.assertEqual(6, len(batch_dp))
        # 测试嵌套批处理器的长度计算是否正确
        self.assertEqual(2, len(batch_dp_2))
# 检查是否允许使用泛型命名元组。Python 3.7及以上版本且不包括Python 3.9支持泛型命名元组。
_generic_namedtuple_allowed = sys.version_info >= (3, 7) and sys.version_info < (3, 9)
if _generic_namedtuple_allowed:

    # 定义一个名为InvalidData的类，它继承自NamedTuple和泛型类型T_co。
    # 该类包含两个属性：name和data。
    class InvalidData(NamedTuple, Generic[T_co]):
        name: str
        data: T_co


class TestTyping(TestCase):
    # 测试isinstance函数的行为
    def test_isinstance(self):
        # 定义类A，它继承自IterDataPipe。
        class A(IterDataPipe):
            pass

        # 定义类B，它继承自IterDataPipe。
        class B(IterDataPipe):
            pass

        # 创建类A的实例a。
        a = A()
        # 断言a是类A的实例。
        self.assertTrue(isinstance(a, A))
        # 断言a不是类B的实例。
        self.assertFalse(isinstance(a, B))

    # 测试定义Protocol类型
    def test_protocol(self):
        try:
            from typing import Protocol  # type: ignore[attr-defined]
        except ImportError:
            from typing import _Protocol  # type: ignore[attr-defined]

            Protocol = _Protocol

        # 定义一个空的Protocol子类P。
        class P(Protocol):
            pass

        # 定义类A，它继承自IterDataPipe，并指定P为其泛型参数。
        class A(IterDataPipe[P]):
            pass

    # 标记为跳过类型检查的测试函数
    @skipTyping
    def test_subtype(self):
        # 导入torch.utils.data.datapipes._typing模块中的issubtype函数。
        from torch.utils.data.datapipes._typing import issubtype

        # 定义基本类型列表basic_type，包括int、str、bool、float、complex、list、tuple、dict、set和T_co。
        basic_type = (int, str, bool, float, complex, list, tuple, dict, set, T_co)
        # 遍历基本类型列表，对每个类型t进行测试。
        for t in basic_type:
            # 断言每个类型t都是其自身的子类型。
            self.assertTrue(issubtype(t, t))
            # 断言每个类型t都是Any类型的子类型。
            self.assertTrue(issubtype(t, Any))
            # 如果t为T_co类型，则断言Any类型是t的子类型。
            if t == T_co:
                self.assertTrue(issubtype(Any, t))
            else:
                self.assertFalse(issubtype(Any, t))
        # 对于基本类型列表中的每对类型t1和t2，检查它们的子类型关系。
        for t1, t2 in itertools.product(basic_type, basic_type):
            if t1 == t2 or t2 == T_co:
                self.assertTrue(issubtype(t1, t2))
            else:
                self.assertFalse(issubtype(t1, t2))

        # 定义类型变量T和S，并指定它们的约束。
        T = TypeVar("T", int, str)
        S = TypeVar("S", bool, Union[str, int], Tuple[int, T])  # type: ignore[valid-type]
        # 定义类型元组列表types，每个元组包含两个类型，分别作为issubtype的参数。
        types = (
            (int, Optional[int]),
            (List, Union[int, list]),
            (Tuple[int, str], S),
            (Tuple[int, str], tuple),
            (T, S),
            (S, T_co),
            (T, Union[S, Set]),
        )
        # 对于types列表中的每对子类型sub和父类型par，检查它们的子类型关系。
        for sub, par in types:
            self.assertTrue(issubtype(sub, par))
            self.assertFalse(issubtype(par, sub))

        # 定义可索引类型字典subscriptable_types，包括List、Tuple、Set和Dict。
        subscriptable_types = {
            List: 1,
            Tuple: 2,  # 使用两个参数
            Set: 1,
            Dict: 2,
        }
        # 对于subscriptable_types中的每种可索引类型subscript_type，以及每种类型的组合ts，检查它们的子类型关系。
        for subscript_type, n in subscriptable_types.items():
            for ts in itertools.combinations(types, n):
                subs, pars = zip(*ts)
                # 通过索引操作获取子类型和父类型的具体值。
                sub = subscript_type[subs]  # type: ignore[index]
                par = subscript_type[pars]  # type: ignore[index]
                # 断言子类型sub是父类型par的子类型。
                self.assertTrue(issubtype(sub, par))
                # 断言父类型par不是子类型sub的子类型（反向）。
                self.assertFalse(issubtype(par, sub))
                # 使用非递归方式检查子类型和父类型的关系。
                self.assertTrue(issubtype(par, sub, recursive=False))

    # 标记为跳过类型检查的测试函数
    @skipTyping
    # 测试 issubinstance 函数的功能
    def test_issubinstance(self):
        # 从 torch.utils.data.datapipes._typing 模块导入 issubinstance 函数
        from torch.utils.data.datapipes._typing import issubinstance

        # 定义基本数据集合
        basic_data = (1, "1", True, 1.0, complex(1.0, 0.0))
        # 定义基本类型集合
        basic_type = (int, str, bool, float, complex)
        # 定义类型变量 S，可以是 bool 或者 Union[str, int] 的子类型
        S = TypeVar("S", bool, Union[str, int])

        # 遍历基本数据集合
        for d in basic_data:
            # 断言 d 是 Any 类型的子类型
            self.assertTrue(issubinstance(d, Any))
            # 断言 d 是 T_co 类型的子类型
            self.assertTrue(issubinstance(d, T_co))
            # 如果 d 的类型是 bool、int 或者 str 中的一种
            if type(d) in (bool, int, str):
                # 断言 d 是类型变量 S 的子类型
                self.assertTrue(issubinstance(d, S))
            else:
                # 否则断言 d 不是类型变量 S 的子类型
                self.assertFalse(issubinstance(d, S))
            # 遍历基本类型集合
            for t in basic_type:
                # 如果 d 的类型和 t 相同
                if type(d) == t:
                    # 断言 d 是类型 t 的子类型
                    self.assertTrue(issubinstance(d, t))
                else:
                    # 否则断言 d 不是类型 t 的子类型
                    self.assertFalse(issubinstance(d, t))

        # 针对 list/set 类型的测试数据
        dt = (([1, "1", 2], List), (set({1, "1", 2}), Set))
        for d, t in dt:
            # 断言 d 是类型 t 的子类型
            self.assertTrue(issubinstance(d, t))
            # 断言 d 是类型 t[T_co] 的子类型
            self.assertTrue(issubinstance(d, t[T_co]))  # type: ignore[index]
            # 断言 d 不是类型 t[int] 的子类型
            self.assertFalse(issubinstance(d, t[int]))  # type: ignore[index]

        # 针对 dict 类型的测试数据
        d = {"1": 1, "2": 2.0}
        # 断言 d 是类型 Dict 的子类型
        self.assertTrue(issubinstance(d, Dict))
        # 断言 d 是类型 Dict[str, T_co] 的子类型
        self.assertTrue(issubinstance(d, Dict[str, T_co]))
        # 断言 d 不是类型 Dict[str, int] 的子类型
        self.assertFalse(issubinstance(d, Dict[str, int]))

        # 针对 tuple 类型的测试数据
        d = (1, "1", 2)
        # 断言 d 是类型 Tuple 的子类型
        self.assertTrue(issubinstance(d, Tuple))
        # 断言 d 是类型 Tuple[int, str, T_co] 的子类型
        self.assertTrue(issubinstance(d, Tuple[int, str, T_co]))
        # 断言 d 不是类型 Tuple[int, Any] 的子类型
        self.assertFalse(issubinstance(d, Tuple[int, Any]))
        # 断言 d 不是类型 Tuple[int, int, int] 的子类型
        self.assertFalse(issubinstance(d, Tuple[int, int, int]))

    # 静态类型检查的注解
    @skipTyping
    @skipTyping
    def test_construct_time(self):
        # 定义一个数据管道 DP0，元素类型为 Tuple
        class DP0(IterDataPipe[Tuple]):
            # 参数验证的装饰器
            @argument_validation
            # 初始化方法，接受一个 IterDataPipe 类型的参数 dp
            def __init__(self, dp: IterDataPipe):
                self.dp = dp

            # 迭代器方法，返回一个迭代器，元素类型为 Tuple
            def __iter__(self) -> Iterator[Tuple]:
                for d in self.dp:
                    yield d, str(d)

        # 定义一个数据管道 DP1，元素类型为 int
        class DP1(IterDataPipe[int]):
            # 参数验证的装饰器
            @argument_validation
            # 初始化方法，接受一个 IterDataPipe[Tuple[int, str]] 类型的参数 dp
            def __init__(self, dp: IterDataPipe[Tuple[int, str]]):
                self.dp = dp

            # 迭代器方法，返回一个迭代器，元素类型为 int
            def __iter__(self) -> Iterator[int]:
                for a, b in self.dp:
                    yield a

        # 非数据管道类型的输入，但使用了数据管道的提示
        datasource = [(1, "1"), (2, "2"), (3, "3")]
        # 断言创建 DP0 对象时会抛出 TypeError 异常，错误信息中包含字符串 "Expected argument 'dp' as a IterDataPipe"
        with self.assertRaisesRegex(
            TypeError, r"Expected argument 'dp' as a IterDataPipe"
        ):
            dp0 = DP0(datasource)

        # 创建一个 IterableWrapper 对象作为 DP0 的参数
        dp0 = DP0(dp.iter.IterableWrapper(range(10)))
        # 断言创建 DP1 对象时会抛出 TypeError 异常，错误信息中包含字符串 "Expected type of argument 'dp' as a subtype"
        with self.assertRaisesRegex(
            TypeError, r"Expected type of argument 'dp' as a subtype"
        ):
            dp1 = DP1(dp0)

    # 静态类型检查的注解
    @skipTyping
    def test_runtime(self):
        # 定义一个数据管道类 DP，继承自 IterDataPipe，并泛型化为 Tuple[int, T_co]
        class DP(IterDataPipe[Tuple[int, T_co]]):
            def __init__(self, datasource):
                self.ds = datasource

            @runtime_validation
            # 迭代器方法，返回一个迭代器，迭代元素类型为 Tuple[int, T_co]
            def __iter__(self) -> Iterator[Tuple[int, T_co]]:
                yield from self.ds

        # 第一组测试数据集合 dss
        dss = ([(1, "1"), (2, "2")], [(1, 1), (2, "2")])
        # 遍历数据集合 dss
        for ds in dss:
            # 创建数据管道对象 dp0，传入数据集 ds
            dp0 = DP(ds)  # type: ignore[var-annotated]
            # 断言数据管道返回的列表与原始数据集相同
            self.assertEqual(list(dp0), ds)
            # 重置 __iter__ 方法
            self.assertEqual(list(dp0), ds)

        # 第二组测试数据集合 dss，包含多种类型不匹配的情况
        dss = (
            [(1, 1), ("2", 2)],  # type: ignore[assignment, list-item]
            [[1, "1"], [2, "2"]],  # type: ignore[list-item]
            [1, "1", 2, "2"],
        )
        # 遍历数据集合 dss
        for ds in dss:
            # 创建数据管道对象 dp0，传入数据集 ds
            dp0 = DP(ds)
            # 使用断言捕获 RuntimeError 异常，检查是否为预期的子类型实例
            with self.assertRaisesRegex(
                RuntimeError, r"Expected an instance as subtype"
            ):
                list(dp0)

            # 使用 runtime_validation_disabled 上下文管理器，禁用运行时验证
            with runtime_validation_disabled():
                # 断言数据管道返回的列表与原始数据集相同
                self.assertEqual(list(dp0), ds)
                # 嵌套禁用运行时验证
                with runtime_validation_disabled():
                    self.assertEqual(list(dp0), ds)

            # 再次使用断言捕获 RuntimeError 异常，检查是否为预期的子类型实例
            with self.assertRaisesRegex(
                RuntimeError, r"Expected an instance as subtype"
            ):
                list(dp0)

    @skipTyping
    def test_reinforce(self):
        # 定义一个泛型类型 T，限制为 int 或 str
        T = TypeVar("T", int, str)

        # 定义一个数据管道类 DP，继承自 IterDataPipe，并泛型化为 T
        class DP(IterDataPipe[T]):
            def __init__(self, ds):
                self.ds = ds

            @runtime_validation
            # 迭代器方法，返回一个迭代器，迭代元素类型为 T
            def __iter__(self) -> Iterator[T]:
                yield from self.ds

        # 创建一个整数列表 ds
        ds = list(range(10))
        # 对数据管道进行类型强化，预期类型为 int
        dp0 = DP(ds).reinforce_type(int)
        # 断言数据管道对象 dp0 的类型为 int
        self.assertTrue(dp0.type, int)
        # 断言数据管道返回的列表与原始数据集 ds 相同
        self.assertEqual(list(dp0), ds)

        # 尝试使用不正确的类型进行类型强化，捕获 TypeError 异常
        with self.assertRaisesRegex(TypeError, r"'expected_type' must be a type"):
            dp1 = DP(ds).reinforce_type(1)

        # 尝试使用不是预期子类型的类型进行类型强化，捕获 TypeError 异常
        with self.assertRaisesRegex(
            TypeError, r"Expected 'expected_type' as subtype of"
        ):
            dp2 = DP(ds).reinforce_type(float)

        # 使用不匹配类型数据运行时，捕获 RuntimeError 异常
        dp3 = DP(ds).reinforce_type(str)
        with self.assertRaisesRegex(RuntimeError, r"Expected an instance as subtype"):
            list(dp3)

        # 使用 runtime_validation_disabled 上下文管理器，禁用运行时验证
        with runtime_validation_disabled():
            # 断言数据管道返回的列表与原始数据集相同
            self.assertEqual(list(dp3), ds)
class NumbersDataset(IterDataPipe):
    # NumbersDataset 类，继承自 IterDataPipe，表示一个生成数字范围的数据集
    def __init__(self, size=10):
        # 初始化方法，设定数据集的大小
        self.size = size

    def __iter__(self):
        # 迭代器方法，生成范围内的数字
        yield from range(self.size)

    def __len__(self):
        # 返回数据集的大小
        return self.size


class TestGraph(TestCase):
    # 测试类 TestGraph，继承自 TestCase
    class CustomIterDataPipe(IterDataPipe):
        # 内部类 CustomIterDataPipe，继承自 IterDataPipe
        def add_v(self, x):
            # 方法 add_v，接受一个参数 x，返回 x 加上对象的 v 属性值
            return x + self.v

        def __init__(self, source_dp, v=1):
            # 初始化方法，接受一个数据管道对象和一个可选的参数 v
            self._dp = source_dp.map(self.add_v)  # 对源数据管道中的每个元素应用 add_v 方法
            self.v = 1  # 设置对象的 v 属性值为 1

        def __iter__(self):
            # 迭代器方法，生成处理后数据管道的元素
            yield from self._dp

        def __hash__(self):
            # hash 方法，抛出未实现错误
            raise NotImplementedError

    def test_simple_traverse(self):
        # 测试简单遍历方法
        numbers_dp = NumbersDataset(size=50)  # 创建一个包含 50 个数字的数据集
        shuffled_dp = numbers_dp.shuffle()  # 对数据集进行洗牌操作
        sharded_dp = shuffled_dp.sharding_filter()  # 对洗牌后的数据集进行分片过滤操作
        mapped_dp = sharded_dp.map(lambda x: x * 10)  # 对分片后的数据集中的每个元素应用 lambda 函数进行映射
        graph = traverse_dps(mapped_dp)  # 遍历处理后的数据管道并生成图形表示
        expected: Dict[Any, Any] = {
            id(mapped_dp): (
                mapped_dp,
                {
                    id(sharded_dp): (
                        sharded_dp,
                        {
                            id(shuffled_dp): (
                                shuffled_dp,
                                {id(numbers_dp): (numbers_dp, {})},
                            )
                        },
                    )
                },
            )
        }
        self.assertEqual(expected, graph)  # 断言生成的图形与预期的图形相等

        dps = torch.utils.data.graph_settings.get_all_graph_pipes(graph)  # 获取图形中的所有数据管道对象
        self.assertEqual(len(dps), 4)  # 断言图中包含的数据管道数量为 4
        for datapipe in (numbers_dp, shuffled_dp, sharded_dp, mapped_dp):
            self.assertTrue(datapipe in dps)  # 断言每个数据管道对象存在于图形中

    def test_traverse_mapdatapipe(self):
        # 测试遍历 map 数据管道的方法
        source_dp = dp.map.SequenceWrapper(range(10))  # 创建一个序列包装器数据管道对象
        map_dp = source_dp.map(partial(_fake_add, 1))  # 对源数据管道中的每个元素应用偏函数 _fake_add(1)
        graph = traverse_dps(map_dp)  # 遍历处理后的数据管道并生成图形表示
        expected: Dict[Any, Any] = {
            id(map_dp): (map_dp, {id(source_dp): (source_dp, {})})
        }
        self.assertEqual(expected, graph)  # 断言生成的图形与预期的图形相等

    def test_traverse_mixdatapipe(self):
        # 测试遍历混合数据管道的方法
        source_map_dp = dp.map.SequenceWrapper(range(10))  # 创建一个序列包装器数据管道对象
        iter_dp = dp.iter.IterableWrapper(source_map_dp)  # 创建一个可迭代包装器数据管道对象
        graph = traverse_dps(iter_dp)  # 遍历处理后的数据管道并生成图形表示
        expected: Dict[Any, Any] = {
            id(iter_dp): (iter_dp, {id(source_map_dp): (source_map_dp, {})})
        }
        self.assertEqual(expected, graph)  # 断言生成的图形与预期的图形相等
    # 定义测试方法，用于测试环形数据管道的遍历功能
    def test_traverse_circular_datapipe(self):
        # 创建一个包装了列表范围的可迭代数据管道
        source_iter_dp = dp.iter.IterableWrapper(list(range(10)))
        # 创建一个自定义的数据管道，作为环形数据管道的包装器
        circular_dp = TestGraph.CustomIterDataPipe(source_iter_dp)
        # 对环形数据管道进行遍历，生成数据管道的图形表示
        graph = traverse_dps(circular_dp)
        # 设置预期结果，表示环形数据管道及其子数据管道的结构
        expected: Dict[Any, Any] = {
            id(circular_dp): (
                circular_dp,
                {
                    id(circular_dp._dp): (
                        circular_dp._dp,
                        {id(source_iter_dp): (source_iter_dp, {})},
                    )
                },
            )
        }
        # 断言预期结果与实际生成的数据管道图形相等
        self.assertEqual(expected, graph)

        # 获取图形中的所有数据管道对象
        dps = torch.utils.data.graph_settings.get_all_graph_pipes(graph)
        # 断言图中数据管道对象的数量为3
        self.assertEqual(len(dps), 3)
        # 验证环形数据管道、其子数据管道和源数据管道是否都在获取到的数据管道列表中
        for _dp in [circular_dp, circular_dp._dp, source_iter_dp]:
            self.assertTrue(_dp in dps)

    # 定义测试方法，用于测试不可哈希数据管道的遍历功能
    def test_traverse_unhashable_datapipe(self):
        # 创建一个包装了列表范围的可迭代数据管道
        source_iter_dp = dp.iter.IterableWrapper(list(range(10)))
        # 创建一个自定义的数据管道，作为不可哈希数据管道的包装器
        unhashable_dp = TestGraph.CustomIterDataPipe(source_iter_dp)
        # 对不可哈希数据管道进行遍历，生成数据管道的图形表示
        graph = traverse_dps(unhashable_dp)
        # 断言对不可哈希数据管道调用 hash() 方法会抛出 NotImplementedError 异常
        with self.assertRaises(NotImplementedError):
            hash(unhashable_dp)
        # 设置预期结果，表示不可哈希数据管道及其子数据管道的结构
        expected: Dict[Any, Any] = {
            id(unhashable_dp): (
                unhashable_dp,
                {
                    id(unhashable_dp._dp): (
                        unhashable_dp._dp,
                        {id(source_iter_dp): (source_iter_dp, {})},
                    )
                },
            )
        }
        # 断言预期结果与实际生成的数据管道图形相等
        self.assertEqual(expected, graph)
# 定义一个函数unbatch，用于从输入列表x中获取第一个元素并返回
def unbatch(x):
    return x[0]


# 定义一个测试类TestSerialization，继承自TestCase
class TestSerialization(TestCase):

    # 装饰器，用于标记测试函数在没有Dill支持时跳过执行
    @skipIfNoDill
    # 测试函数，验证使用spawn模式创建DataLoader加载数据时的行为
    def test_spawn_lambdas_iter(self):
        # 创建一个IterableWrapper对象idp，包装一个范围为0到2的迭代器，并进行映射、混洗操作
        idp = dp.iter.IterableWrapper(range(3)).map(lambda x: x + 1).shuffle()
        # 创建一个DataLoader对象dl，加载idp作为数据源，设置参数如下：
        # num_workers表示工作进程数为2
        # shuffle表示进行数据加载时是否打乱顺序
        # multiprocessing_context表示使用spawn方式创建进程
        # collate_fn表示用于数据合并的函数unbatch
        # batch_size表示每个batch的大小为1
        dl = DataLoader(
            idp,
            num_workers=2,
            shuffle=True,
            multiprocessing_context="spawn",
            collate_fn=unbatch,
            batch_size=1,
        )
        # 执行DataLoader加载数据，将结果转换为列表
        result = list(dl)
        # 断言加载结果是否符合预期：应为[1, 1, 2, 2, 3, 3]，且是有序的
        self.assertEqual([1, 1, 2, 2, 3, 3], sorted(result))

    # 同上一个测试函数类似，验证使用spawn模式创建DataLoader加载数据时的行为
    @skipIfNoDill
    def test_spawn_lambdas_map(self):
        # 创建一个SequenceWrapper对象mdp，包装一个范围为0到2的序列，并进行映射、混洗操作
        mdp = dp.map.SequenceWrapper(range(3)).map(lambda x: x + 1).shuffle()
        # 创建一个DataLoader对象dl，加载mdp作为数据源，设置参数与上一个测试函数相同
        dl = DataLoader(
            mdp,
            num_workers=2,
            shuffle=True,
            multiprocessing_context="spawn",
            collate_fn=unbatch,
            batch_size=1,
        )
        # 执行DataLoader加载数据，将结果转换为列表
        result = list(dl)
        # 断言加载结果是否符合预期：应为[1, 1, 2, 2, 3, 3]，且是有序的
        self.assertEqual([1, 1, 2, 2, 3, 3], sorted(result))


# 定义一个测试类TestCircularSerialization，继承自TestCase
class TestCircularSerialization(TestCase):

    # 定义一个内部类CustomIterDataPipe，继承自IterDataPipe
    class CustomIterDataPipe(IterDataPipe):

        # 静态方法，用于将输入x加1并返回
        @staticmethod
        def add_one(x):
            return x + 1

        # 类方法，用于对输入x进行分类，始终返回0
        @classmethod
        def classify(cls, x):
            return 0

        # 实例方法，对输入x加上实例属性v的值并返回
        def add_v(self, x):
            return x + self.v

        # 初始化方法，接收fn和source_dp两个参数
        def __init__(self, fn, source_dp=None):
            # 初始化fn属性和source_dp属性，如果source_dp为空，则默认为一个IterableWrapper包装的[1, 2, 4]
            self.fn = fn
            self.source_dp = (
                source_dp if source_dp else dp.iter.IterableWrapper([1, 2, 4])
            )
            # 使用source_dp进行映射操作，依次调用add_one、add_v方法，然后进行demux操作，分解为两部分，取第一部分作为数据源_dp
            self._dp = (
                self.source_dp.map(self.add_one)
                .map(self.add_v)
                .demux(2, self.classify)[0]
            )
            # 初始化实例属性v为1
            self.v = 1

        # 迭代器方法，返回_dp的迭代器
        def __iter__(self):
            yield from self._dp

    # 定义一个内部类LambdaIterDataPipe，继承自CustomIterDataPipe
    class LambdaIterDataPipe(CustomIterDataPipe):

        # 初始化方法，接收fn和source_dp两个参数，并调用父类的初始化方法
        def __init__(self, fn, source_dp=None):
            super().__init__(fn, source_dp)
            # 创建一个lambda表达式列表container，该表达式对输入x执行加1操作
            self.container = [
                lambda x: x + 1,
            ]
            # 创建一个lambda表达式lambda_fn，对输入x执行加1操作
            self.lambda_fn = lambda x: x + 1
            # 使用source_dp进行映射操作，依次调用add_one、lambda_fn、add_v方法，然后进行demux操作，分解为两部分，取第一部分作为数据源_dp
            self._dp = (
                self.source_dp.map(self.add_one)
                .map(self.lambda_fn)
                .map(self.add_v)
                .demux(2, self.classify)[0]
            )

    # 装饰器，用于标记测试函数跳过执行，并附带跳过的原因"Dill Tests"
    @skipIfNoDill
    @skipIf(True, "Dill Tests")
    # 定义一个内部类CustomShardingIterDataPipe，继承自IterDataPipe
class CustomShardingIterDataPipe(IterDataPipe):

    # 初始化方法，接收dp作为参数
    def __init__(self, dp):
        # 初始化实例属性dp为输入参数dp
        self.dp = dp
        # 初始化实例属性num_of_instances为1，instance_id为0
        self.num_of_instances = 1
        self.instance_id = 0

    # 方法，用于应用分片，接收num_of_instances和instance_id两个参数
    def apply_sharding(self, num_of_instances, instance_id):
        # 更新实例属性num_of_instances和instance_id为输入参数值
        self.num_of_instances = num_of_instances
        self.instance_id = instance_id

    # 迭代器方法，迭代处理self.dp中的数据
    def __iter__(self):
        for i, d in enumerate(self.dp):
            # 如果i除以num_of_instances的余数等于instance_id，则yield返回当前数据d
            if i % self.num_of_instances == self.instance_id:
                yield d


# 定义一个测试类TestSharding，继承自TestCase
class TestSharding(TestCase):

    # 方法，返回一个数据处理管道combined_dp
    def _get_pipeline(self):
        # 创建一个NumbersDataset对象numbers_dp，大小为10
        numbers_dp = NumbersDataset(size=10)
        # 对numbers_dp进行fork操作，分成两个数据管道dp0和dp1
        dp0, dp1 = numbers_dp.fork(num_instances=2)
        # 对dp0进行映射操作_mul_10
        dp0_upd = dp0.map(_mul_10)
        # 对dp1进行过滤操作_mod_3_test
        dp1_upd = dp1.filter(_mod_3_test)
        # 合并dp0_upd和dp1_upd两个数据管道为combined_dp
        combined_dp = dp0_upd.mux(dp1_upd)
        # 返回合并后的数据管道combined_dp
        return combined_dp
    # 定义一个方法，用于获取一个带有特定配置的数据管道
    def _get_dill_pipeline(self):
        # 创建一个包含10个元素的NumbersDataset对象
        numbers_dp = NumbersDataset(size=10)
        # 对数据集进行分叉，得到两个数据管道dp0和dp1
        dp0, dp1 = numbers_dp.fork(num_instances=2)
        # 对dp0中的每个元素应用一个乘以10的映射操作，得到更新后的dp0_upd
        dp0_upd = dp0.map(lambda x: x * 10)
        # 对dp1中的元素进行过滤，保留能被3整除余1的元素，得到更新后的dp1_upd
        dp1_upd = dp1.filter(lambda x: x % 3 == 1)
        # 将更新后的dp0_upd和dp1_upd数据管道进行复用（mux），得到组合的数据管道combined_dp
        combined_dp = dp0_upd.mux(dp1_upd)
        # 返回组合后的数据管道
        return combined_dp

    # 测试简单的数据分片操作
    def test_simple_sharding(self):
        # 获取一个包含分片过滤器的数据管道
        sharded_dp = self._get_pipeline().sharding_filter()
        # 对分片数据管道应用分片设置，设置分片大小为3，当前分片索引为1
        torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, 1)
        # 将分片数据管道中的元素转换为列表
        items = list(sharded_dp)
        # 断言列表中的元素与预期相符
        self.assertEqual([1, 20], items)

        # 定义一个包含所有元素的列表
        all_items = [0, 1, 10, 4, 20, 7]
        items = []
        # 循环三次，分别处理三个不同的分片
        for i in range(3):
            # 获取一个包含分片过滤器的数据管道
            sharded_dp = self._get_pipeline().sharding_filter()
            # 对分片数据管道应用分片设置，设置分片大小为3，当前分片索引为i
            torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, i)
            # 将分片数据管道中的元素添加到items列表中
            items += list(sharded_dp)
        # 断言排序后的items与预期的所有元素列表排序后相符
        self.assertEqual(sorted(all_items), sorted(items))

    # 测试分片分组功能
    def test_sharding_groups(self):
        # 定义一个构建分片管道的内部方法
        def construct_sharded_pipe():
            # 创建一个包含90个元素的NumbersDataset对象
            dp = NumbersDataset(size=90)
            # 对数据集应用分片过滤器，使用分布式分片组优先级进行过滤
            dp = dp.sharding_filter(
                sharding_group_filter=SHARDING_PRIORITIES.DISTRIBUTED
            )
            # 将过滤后的数据管道添加到sharding_pipes列表中
            sharding_pipes.append(dp)
            # 再次对数据集应用分片过滤器，使用多处理分片组优先级进行过滤
            dp = dp.sharding_filter(
                sharding_group_filter=SHARDING_PRIORITIES.MULTIPROCESSING
            )
            # 将过滤后的数据管道添加到sharding_pipes列表中
            sharding_pipes.append(dp)
            # 再次对数据集应用分片过滤器，使用分片组优先级为300进行过滤
            dp = dp.sharding_filter(sharding_group_filter=300)
            # 将过滤后的数据管道返回
            return dp, sharding_pipes

        # 调用内部方法构建分片管道dp和sharding_pipes列表
        dp, sharding_pipes = construct_sharded_pipe()

        # 遍历sharding_pipes列表中的每个管道
        for pipe in sharding_pipes:
            # 对管道应用分片设置，设置分片大小为2，当前分片索引为1，使用分布式分片组优先级
            pipe.apply_sharding(2, 1, sharding_group=SHARDING_PRIORITIES.DISTRIBUTED)
            # 对管道应用分片设置，设置分片大小为5，当前分片索引为3，使用多处理分片组优先级
            pipe.apply_sharding(
                5, 3, sharding_group=SHARDING_PRIORITIES.MULTIPROCESSING
            )
            # 对管道应用分片设置，设置分片大小为3，当前分片索引为1，使用分片组优先级为300
            pipe.apply_sharding(3, 1, sharding_group=300)

        # 将分片管道dp中的元素转换为列表
        actual = list(dp)
        # 定义预期的元素列表
        expected = [17, 47, 77]
        # 断言实际的元素列表与预期的元素列表相符
        self.assertEqual(expected, actual)
        # 断言分片管道dp中的元素个数为3
        self.assertEqual(3, len(dp))

        # 再次调用内部方法构建分片管道dp和丢弃sharding_pipes列表
        dp, _ = construct_sharded_pipe()
        # 对管道应用分片设置，设置分片大小为2，当前分片索引为1，使用默认分片组优先级
        dp.apply_sharding(2, 1, sharding_group=SHARDING_PRIORITIES.DEFAULT)
        # 断言抛出异常，因为不支持对同一管道多次应用不同优先级的分片设置
        with self.assertRaises(Exception):
            dp.apply_sharding(5, 3, sharding_group=SHARDING_PRIORITIES.MULTIPROCESSING)

        # 再次调用内部方法构建分片管道dp和丢弃sharding_pipes列表
        dp, _ = construct_sharded_pipe()
        # 对管道应用分片设置，设置分片大小为5，当前分片索引为3，使用多处理分片组优先级
        dp.apply_sharding(5, 3, sharding_group=SHARDING_PRIORITIES.MULTIPROCESSING)
        # 断言抛出异常，因为不支持对同一管道多次应用不同优先级的分片设置
        with self.assertRaises(Exception):
            dp.apply_sharding(2, 1, sharding_group=SHARDING_PRIORITIES.DEFAULT)
    # 测试旧版分组包中的分片组功能
    def test_sharding_groups_in_legacy_grouping_package(self):
        # 断言会发出未来警告，提醒使用新版的数据管道分片优先级常量
        with self.assertWarnsRegex(
            FutureWarning,
            r"Please use `SHARDING_PRIORITIES` "
            "from the `torch.utils.data.datapipes.iter.sharding`",
        ):
            # 导入旧版分组包中的分片优先级常量并重命名为LEGACY_SHARDING_PRIORITIES
            from torch.utils.data.datapipes.iter.grouping import (
                SHARDING_PRIORITIES as LEGACY_SHARDING_PRIORITIES,
            )

        # 定义创建分片管道的函数
        def construct_sharded_pipe():
            sharding_pipes = []
            # 创建一个包含90个数值的数据集对象dp
            dp = NumbersDataset(size=90)
            # 应用分片过滤器，使用分布式分片组过滤器
            dp = dp.sharding_filter(
                sharding_group_filter=LEGACY_SHARDING_PRIORITIES.DISTRIBUTED
            )
            sharding_pipes.append(dp)
            # 应用分片过滤器，使用多进程分片组过滤器
            dp = dp.sharding_filter(
                sharding_group_filter=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING
            )
            sharding_pipes.append(dp)
            # 应用分片过滤器，使用数值300作为分片组过滤器
            dp = dp.sharding_filter(sharding_group_filter=300)
            sharding_pipes.append(dp)
            return dp, sharding_pipes

        # 调用构建分片管道的函数，获取dp和分片管道列表sharding_pipes
        dp, sharding_pipes = construct_sharded_pipe()

        # 遍历分片管道列表，为每个管道应用分片
        for pipe in sharding_pipes:
            # 应用分片：2个任务，1个副本，使用分布式分片组
            pipe.apply_sharding(
                2, 1, sharding_group=LEGACY_SHARDING_PRIORITIES.DISTRIBUTED
            )
            # 应用分片：5个任务，3个副本，使用多进程分片组
            pipe.apply_sharding(
                5, 3, sharding_group=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING
            )
            # 应用分片：3个任务，1个副本，使用数值300作为分片组
            pipe.apply_sharding(3, 1, sharding_group=300)

        # 获取dp的迭代结果actual
        actual = list(dp)
        # 预期的迭代结果
        expected = [17, 47, 77]
        # 断言实际结果与预期结果相等
        self.assertEqual(expected, actual)
        # 断言dp的长度为3
        self.assertEqual(3, len(dp))

        # 重新调用构建分片管道的函数，获取dp和忽略的分片管道列表
        dp, _ = construct_sharded_pipe()
        # 应用分片：2个任务，1个副本，使用默认分片组
        dp.apply_sharding(2, 1, sharding_group=LEGACY_SHARDING_PRIORITIES.DEFAULT)
        # 断言会抛出异常
        with self.assertRaises(Exception):
            # 应用分片：5个任务，3个副本，使用多进程分片组
            dp.apply_sharding(
                5, 3, sharding_group=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING
            )

        # 重新调用构建分片管道的函数，获取dp和忽略的分片管道列表
        dp, _ = construct_sharded_pipe()
        # 应用分片：5个任务，3个副本，使用多进程分片组
        dp.apply_sharding(
            5, 3, sharding_group=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING
        )
        # 断言会抛出异常
        with self.assertRaises(Exception):
            # 应用分片：2个任务，1个副本，使用默认分片组
            dp.apply_sharding(2, 1, sharding_group=LEGACY_SHARDING_PRIORITIES.DEFAULT)

    # 测试旧版自定义分片
    def test_legacy_custom_sharding(self):
        # 获取数据管道对象dp
        dp = self._get_pipeline()
        # 创建自定义分片迭代数据管道对象sharded_dp
        sharded_dp = CustomShardingIterDataPipe(dp)
        # 应用分片设置：3个任务，1个副本
        torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, 1)
        # 获取分片后的数据列表
        items = list(sharded_dp)
        # 断言分片后的数据列表与预期结果相等
        self.assertEqual([1, 20], items)
    # 测试数据分片的长度
    def test_sharding_length(self):
        # 创建一个包含数字 0 到 12 的可迭代对象
        numbers_dp = dp.iter.IterableWrapper(range(13))
        # 对数字进行分片过滤，生成第一个分片数据管道
        sharded_dp0 = numbers_dp.sharding_filter()
        # 对第一个分片数据应用分片设置，分为3部分，选择第0部分
        torch.utils.data.graph_settings.apply_sharding(sharded_dp0, 3, 0)
        # 生成第二个分片数据管道
        sharded_dp1 = numbers_dp.sharding_filter()
        # 对第二个分片数据应用分片设置，分为3部分，选择第1部分
        torch.utils.data.graph_settings.apply_sharding(sharded_dp1, 3, 1)
        # 生成第三个分片数据管道
        sharded_dp2 = numbers_dp.sharding_filter()
        # 对第三个分片数据应用分片设置，分为3部分，选择第2部分
        torch.utils.data.graph_settings.apply_sharding(sharded_dp2, 3, 2)
        # 断言：原始数据长度应为13
        self.assertEqual(13, len(numbers_dp))
        # 断言：第一个分片数据长度应为5
        self.assertEqual(5, len(sharded_dp0))
        # 断言：第二个分片数据长度应为4
        self.assertEqual(4, len(sharded_dp1))
        # 断言：第三个分片数据长度应为4
        self.assertEqual(4, len(sharded_dp2))

        # 创建一个仅包含数字1的可迭代对象
        numbers_dp = dp.iter.IterableWrapper(range(1))
        # 对数字进行分片过滤，生成第一个分片数据管道
        sharded_dp0 = numbers_dp.sharding_filter()
        # 对第一个分片数据应用分片设置，分为2部分，选择第0部分
        torch.utils.data.graph_settings.apply_sharding(sharded_dp0, 2, 0)
        # 生成第二个分片数据管道
        sharded_dp1 = numbers_dp.sharding_filter()
        # 对第二个分片数据应用分片设置，分为2部分，选择第1部分
        torch.utils.data.graph_settings.apply_sharding(sharded_dp1, 2, 1)
        # 断言：第一个分片数据长度应为1
        self.assertEqual(1, len(sharded_dp0))
        # 断言：第二个分片数据长度应为0
        self.assertEqual(0, len(sharded_dp1))

    # 测试旧数据加载器
    def test_old_dataloader(self):
        # 获取数据管道的第一个处理步骤
        dp0 = self._get_pipeline()
        # 期望的输出列表
        expected = list(dp0)

        # 对数据管道应用分片过滤
        dp0 = self._get_pipeline().sharding_filter()
        # 使用DataLoader加载数据管道，设置批大小为1，不打乱数据，使用2个工作线程
        dl = DataLoader(dp0, batch_size=1, shuffle=False, num_workers=2)
        # 从数据加载器中获取所有项目
        items = list(dl)

        # 断言：期望的排序后的输出应与实际输出一致
        self.assertEqual(sorted(expected), sorted(items))

    # 测试旧数据加载器与旧的自定义分片方法
    def test_legacy_custom_sharding_with_old_dataloader(self):
        # 获取数据管道的第一个处理步骤
        dp0 = self._get_pipeline()
        # 期望的输出列表
        expected = list(dp0)

        # 使用自定义分片方法包装数据管道
        dp0 = CustomShardingIterDataPipe(dp0)
        # 使用DataLoader加载自定义分片后的数据管道，设置批大小为1，不打乱数据，使用2个工作线程
        dl = DataLoader(dp0, batch_size=1, shuffle=False, num_workers=2)
        # 从数据加载器中获取所有项目
        items = list(dl)

        # 断言：期望的排序后的输出应与实际输出一致
        self.assertEqual(sorted(expected), sorted(items))
    # 定义一个测试方法，用于测试多重分片功能
    def test_multi_sharding(self):
        # 当在单一分支上进行多次分片时抛出错误
        numbers_dp = dp.iter.IterableWrapper(range(13))
        # 对数据源进行第一次分片操作
        sharded_dp = numbers_dp.sharding_filter()
        # 对第一次分片结果再次进行分片操作
        sharded_dp = sharded_dp.sharding_filter()
        # 验证是否抛出指定的运行时错误
        with self.assertRaisesRegex(
            RuntimeError, "Sharding twice on a single pipeline"
        ):
            # 尝试在同一管道上应用第二次分片
            torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, 0)

        # 当在数据源和分支上同时进行分片操作时抛出错误
        numbers_dp = dp.iter.IterableWrapper(range(13)).sharding_filter()
        # 将数据源进行分叉操作，得到两个数据流 dp1 和 dp2
        dp1, dp2 = numbers_dp.fork(2)
        # 对 dp1 分片
        sharded_dp = dp1.sharding_filter()
        # 将 dp2 和 sharded_dp 进行压缩操作
        zip_dp = dp2.zip(sharded_dp)
        # 验证是否抛出指定的运行时错误
        with self.assertRaisesRegex(
            RuntimeError, "Sharding twice on a single pipeline"
        ):
            # 尝试在同一管道上应用第二次分片
            torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)

        # 当在分支末端和分支上同时进行分片操作时抛出错误
        numbers_dp = dp.iter.IterableWrapper(range(13))
        # 将数据源进行分叉操作，得到两个数据流 dp1 和 dp2
        dp1, dp2 = numbers_dp.fork(2)
        # 对 dp1 进行分片
        sharded_dp = dp1.sharding_filter()
        # 将 dp2 和 sharded_dp 进行压缩操作，并对结果进行分片
        zip_dp = dp2.zip(sharded_dp).sharding_filter()
        # 验证是否抛出指定的运行时错误
        with self.assertRaisesRegex(
            RuntimeError, "Sharding twice on a single pipeline"
        ):
            # 尝试在同一管道上应用第二次分片
            torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)

        # 对数据源进行单一分片操作
        numbers_dp = dp.iter.IterableWrapper(range(13)).sharding_filter()
        # 将数据源进行分叉操作，得到两个数据流 dp1 和 dp2
        dp1, dp2 = numbers_dp.fork(2)
        # 将 dp1 和 dp2 进行压缩操作
        zip_dp = dp1.zip(dp2)
        # 对压缩后的数据流应用分片操作
        torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)
        # 验证分片后的结果是否符合预期
        self.assertEqual(list(zip_dp), [(i * 3, i * 3) for i in range(13 // 3 + 1)])

        # 对每个分支进行单一分片操作
        numbers_dp = dp.iter.IterableWrapper(range(13))
        # 将数据源进行分叉操作，得到两个数据流 dp1 和 dp2
        dp1, dp2 = numbers_dp.fork(2)
        # 对 dp1 进行分片
        sharded_dp1 = dp1.sharding_filter()
        # 对 dp2 进行分片
        sharded_dp2 = dp2.sharding_filter()
        # 将分片后的 dp1 和 dp2 进行压缩操作
        zip_dp = sharded_dp1.zip(sharded_dp2)
        # 对压缩后的数据流应用分片操作
        torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)
        # 验证分片后的结果是否符合预期
        self.assertEqual(list(zip_dp), [(i * 3, i * 3) for i in range(13 // 3 + 1)])
    class TestIterDataPipeSingletonConstraint(TestCase):
        r"""
        Each `IterDataPipe` can only have one active iterator. Whenever a new iterator is created, older
        iterators are invalidated. These tests aim to ensure `IterDataPipe` follows this behavior.
        """

        def _check_single_iterator_invalidation_logic(self, source_dp: IterDataPipe):
            r"""
            Given a IterDataPipe, verifies that the iterator can be read, reset, and the creation of
            a second iterator invalidates the first one.
            """
            # 创建第一个迭代器
            it1 = iter(source_dp)
            # 断言第一个迭代器返回的内容与预期一致
            self.assertEqual(list(range(10)), list(it1))
            # 重新创建第一个迭代器，确认可以再次完整读取
            it1 = iter(source_dp)
            self.assertEqual(
                list(range(10)), list(it1)
            )  # A fresh iterator can be read in full again
            # 重新创建第一个迭代器，读取第一个元素并断言
            it1 = iter(source_dp)
            self.assertEqual(0, next(it1))
            # 创建第二个迭代器，预期第一个迭代器被失效
            it2 = iter(source_dp)  # This should invalidate `it1`
            # 断言第二个迭代器能够从头开始读取
            self.assertEqual(0, next(it2))  # Should read from the beginning again
            # 使用断言确保尝试使用失效的迭代器会触发 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
                next(it1)

        def test_iterdatapipe_singleton_generator(self):
            r"""
            Testing for the case where IterDataPipe's `__iter__` is a generator function.
            """

            # Functional Test: Check if invalidation logic is correct
            # 创建一个基于 range 的 IterableWrapper 作为测试数据源
            source_dp: IterDataPipe = dp.iter.IterableWrapper(range(10))
            # 调用 _check_single_iterator_invalidation_logic 方法进行验证
            self._check_single_iterator_invalidation_logic(source_dp)

            # Functional Test: extend the test to a pipeline
            # 对数据源进行 map 和 filter 操作，创建一个处理数据的 pipeline
            dps = source_dp.map(_fake_fn).filter(_fake_filter_fn)
            # 再次调用验证方法，确保管道操作的正确性
            self._check_single_iterator_invalidation_logic(dps)

            # Functional Test: multiple simultaneous references to the same DataPipe fails
            # 对于同一数据源进行并行迭代，预期会触发 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
                for _ in zip(source_dp, source_dp):
                    pass

            # Function Test: sequential references work
            # 对同一数据源进行连续迭代，确认能正常工作
            for _ in zip(list(source_dp), list(source_dp)):
                pass
    # 定义一个测试方法，用于测试 IterDataPipe 的 `__iter__` 返回 `self` 并且有 `__next__` 方法的情况
    r"""
    Testing for the case where IterDataPipe's `__iter__` returns `self` and there is a `__next__` method
    Note that the following DataPipe by is singleton by default (because `__iter__` returns `self`).
    """

    # 定义一个自定义的 IterDataPipe 子类 _CustomIterDP_Self
    class _CustomIterDP_Self(IterDataPipe):
        
        # 初始化方法，接受一个可迭代对象作为参数
        def __init__(self, iterable):
            self.source = iterable  # 将传入的可迭代对象保存为实例变量
            self.iterable = iter(iterable)  # 使用传入的可迭代对象创建一个迭代器

        # 实现 __iter__ 方法，重置迭代器并返回自身
        def __iter__(self):
            self.reset()  # 调用 reset 方法重置迭代器状态
            return self  # 返回自身作为迭代器

        # 实现 __next__ 方法，通过内置的迭代器获取下一个元素
        def __next__(self):
            return next(self.iterable)  # 返回迭代器的下一个元素

        # 自定义方法 reset，用于重置迭代器为初始状态
        def reset(self):
            self.iterable = iter(self.source)  # 将迭代器重置为传入的可迭代对象的新迭代器

    # 功能测试：检查每次调用 `__iter__` 方法返回的对象是否相同
    source_dp = _CustomIterDP_Self(range(10))
    res = list(source_dp)
    it = iter(source_dp)
    self.assertEqual(res, list(it))

    # 功能测试：检查失效逻辑是否正确
    source_dp = _CustomIterDP_Self(range(10))
    self._check_single_iterator_invalidation_logic(source_dp)
    self.assertEqual(
        1, next(source_dp)
    )  # `source_dp` 仍然有效并且可以读取

    # 功能测试：将测试扩展到一个数据处理管道
    source_dp = _CustomIterDP_Self(
        dp.iter.IterableWrapper(range(10)).map(_fake_fn).filter(_fake_filter_fn)
    )
    self._check_single_iterator_invalidation_logic(source_dp)
    self.assertEqual(
        1, next(source_dp)
    )  # `source_dp` 仍然有效并且可以读取

    # 功能测试：多个对同一 DataPipe 的同时引用将失败
    with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
        for _ in zip(source_dp, source_dp):
            pass
    def test_iterdatapipe_singleton_new_object(self):
        r"""
        Testing for the case where IterDataPipe's `__iter__` isn't a generator nor returns `self`,
        and there isn't a `__next__` method.
        """

        class _CustomIterDP(IterDataPipe):
            def __init__(self, iterable):
                self.iterable = iter(iterable)

            def __iter__(self):  # Note that this doesn't reset
                return self.iterable  # Intentionally not returning `self`

        # Functional Test: Check if invalidation logic is correct
        source_dp = _CustomIterDP(range(10))  # 创建一个自定义的迭代数据管道对象，使用一个范围为 0 到 9 的迭代器
        it1 = iter(source_dp)  # 获取迭代器 it1
        self.assertEqual(0, next(it1))  # 确认 it1 的下一个元素是 0
        it2 = iter(source_dp)  # 获取新的迭代器 it2，与 source_dp 共享相同的数据源
        self.assertEqual(1, next(it2))  # 确认 it2 的下一个元素是 1
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it1)  # 确认试图使用已失效的迭代器 it1 会触发 RuntimeError 异常

        # Functional Test: extend the test to a pipeline
        source_dp = _CustomIterDP(
            dp.iter.IterableWrapper(range(10)).map(_fake_fn).filter(_fake_filter_fn)
        )  # 创建一个包含映射和过滤操作的自定义迭代数据管道对象
        it1 = iter(source_dp)  # 获取迭代器 it1
        self.assertEqual(0, next(it1))  # 确认 it1 的下一个元素是 0
        it2 = iter(source_dp)  # 获取新的迭代器 it2，与 source_dp 共享相同的数据源
        self.assertEqual(1, next(it2))  # 确认 it2 的下一个元素是 1
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it1)  # 确认试图使用已失效的迭代器 it1 会触发 RuntimeError 异常

        # Functional Test: multiple simultaneous references to the same DataPipe fails
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            for _ in zip(source_dp, source_dp):
                pass  # 确认尝试同时迭代相同的数据管道对象 source_dp 会触发 RuntimeError 异常
    def test_iterdatapipe_singleton_buggy(self):
        r"""
        Buggy test case case where IterDataPipe's `__iter__` returns a new object, but also has
        a `__next__` method.
        """

        class _CustomIterDP(IterDataPipe):
            def __init__(self, iterable):
                self.source = iterable
                self.iterable = iter(iterable)

            def __iter__(self):
                return iter(self.source)  # 返回了来源数据的新迭代器，而非 `self`

            def __next__(self):
                return next(self.iterable)  # 返回下一个迭代器中的元素

        # Functional Test: Check if invalidation logic is correct
        source_dp = _CustomIterDP(range(10))
        self._check_single_iterator_invalidation_logic(source_dp)
        self.assertEqual(0, next(source_dp))  # `__next__` 与 `__iter__` 无关

        # Functional Test: Special case to show `__next__` is unrelated with `__iter__`
        source_dp = _CustomIterDP(range(10))
        self.assertEqual(0, next(source_dp))
        it1 = iter(source_dp)
        self.assertEqual(0, next(it1))
        self.assertEqual(1, next(source_dp))
        it2 = iter(source_dp)  # 使 `it1` 失效
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it1)
        self.assertEqual(2, next(source_dp))  # 不受 `it2` 创建的影响
        self.assertEqual(
            list(range(10)), list(it2)
        )  # `it2` 仍然有效，因为它是一个新的对象
class TestIterDataPipeCountSampleYielded(TestCase):
    def _yield_count_test_helper(self, datapipe, n_expected_samples):
        # Functional Test: Check if number of samples yielded is as expected
        res = list(datapipe)
        self.assertEqual(len(res), datapipe._number_of_samples_yielded)

        # Functional Test: Check if the count is correct when DataPipe is partially read
        it = iter(datapipe)
        res = []
        for i, value in enumerate(it):
            res.append(value)
            if i == n_expected_samples - 1:
                break
        self.assertEqual(n_expected_samples, datapipe._number_of_samples_yielded)

        # Functional Test: Check for reset behavior and if iterator also works
        it = iter(datapipe)  # reset the DataPipe
        res = list(it)
        self.assertEqual(len(res), datapipe._number_of_samples_yielded)

    def test_iterdatapipe_sample_yielded_generator_function(self):
        # Functional Test: `__iter__` is a generator function
        datapipe: IterDataPipe = dp.iter.IterableWrapper(range(10))
        self._yield_count_test_helper(datapipe, n_expected_samples=5)

    def test_iterdatapipe_sample_yielded_generator_function_exception(self):
        # Functional Test: `__iter__` is a custom generator function with exception
        class _CustomGeneratorFnDataPipe(IterDataPipe):
            # This class's `__iter__` has a Runtime Error
            def __iter__(self):
                yield 0
                yield 1
                yield 2
                raise RuntimeError("Custom test error after yielding 3 elements")
                yield 3

        # Functional Test: Ensure the count is correct even when exception is raised
        datapipe: IterDataPipe = _CustomGeneratorFnDataPipe()
        with self.assertRaisesRegex(
            RuntimeError, "Custom test error after yielding 3 elements"
        ):
            list(datapipe)
        self.assertEqual(3, datapipe._number_of_samples_yielded)

        # Functional Test: Check for reset behavior and if iterator also works
        it = iter(datapipe)  # reset the DataPipe
        with self.assertRaisesRegex(
            RuntimeError, "Custom test error after yielding 3 elements"
        ):
            list(it)
        self.assertEqual(3, datapipe._number_of_samples_yielded)

    def test_iterdatapipe_sample_yielded_return_self(self):
        class _CustomGeneratorDataPipe(IterDataPipe):
            # This class's `__iter__` is not a generator function
            def __init__(self):
                self.source = iter(range(10))

            def __iter__(self):
                return self.source

            def reset(self):
                self.source = iter(range(10))

        datapipe: IterDataPipe = _CustomGeneratorDataPipe()
        self._yield_count_test_helper(datapipe, n_expected_samples=5)
    def test_iterdatapipe_sample_yielded_next(self):
        class _CustomNextDataPipe(IterDataPipe):
            # 定义一个自定义的数据管道类，继承自IterDataPipe

            # 构造函数，初始化数据源为一个迭代器生成的范围内的整数
            def __init__(self):
                self.source = iter(range(10))

            # 返回自身，使得该类的实例可以迭代
            def __iter__(self):
                return self

            # 实现迭代器的__next__方法，从数据源中获取下一个元素
            def __next__(self):
                return next(self.source)

            # 重置数据源，重新生成一个迭代器
            def reset(self):
                self.source = iter(range(10))

        # 创建_CustomNextDataPipe类的实例作为数据管道
        datapipe: IterDataPipe = _CustomNextDataPipe()
        # 调用辅助函数验证数据管道产生的样本数量是否符合预期
        self._yield_count_test_helper(datapipe, n_expected_samples=5)

    def test_iterdatapipe_sample_yielded_next_exception(self):
        class _CustomNextDataPipe(IterDataPipe):
            # 定义一个自定义的数据管道类，继承自IterDataPipe

            # 构造函数，初始化数据源为一个迭代器生成的范围内的整数，并初始化计数器
            def __init__(self):
                self.source = iter(range(10))
                self.count = 0

            # 返回自身，使得该类的实例可以迭代
            def __iter__(self):
                return self

            # 实现迭代器的__next__方法，当计数到3时抛出异常
            def __next__(self):
                if self.count == 3:
                    raise RuntimeError("Custom test error after yielding 3 elements")
                self.count += 1
                return next(self.source)

            # 重置计数器和数据源
            def reset(self):
                self.count = 0
                self.source = iter(range(10))

        # Functional Test: 确保在抛出异常后，计数器的值正确
        datapipe: IterDataPipe = _CustomNextDataPipe()
        with self.assertRaisesRegex(
            RuntimeError, "Custom test error after yielding 3 elements"
        ):
            list(datapipe)
        # 验证抛出异常前已经产生了3个样本
        self.assertEqual(3, datapipe._number_of_samples_yielded)

        # Functional Test: 检查重置行为和迭代器的正常工作
        it = iter(datapipe)  # 重置数据管道
        with self.assertRaisesRegex(
            RuntimeError, "Custom test error after yielding 3 elements"
        ):
            list(it)
        # 验证重置后再次抛出异常前已经产生了3个样本
        self.assertEqual(3, datapipe._number_of_samples_yielded)
class _CustomNonGeneratorTestDataPipe(IterDataPipe):
    def __init__(self):
        # 初始化数据管道，设定数据个数为 10
        self.n = 10
        # 创建数据源，为一个包含 0 到 9 的列表
        self.source = list(range(self.n))

    # This class's `__iter__` is not a generator function
    # 此类的 `__iter__` 不是生成器函数
    def __iter__(self):
        # 返回数据源的迭代器
        return iter(self.source)

    def __len__(self):
        # 返回数据管道中元素的个数
        return self.n


class _CustomSelfNextTestDataPipe(IterDataPipe):
    def __init__(self):
        # 初始化数据管道，设定数据个数为 10
        self.n = 10
        # 创建数据源的迭代器，为一个产生 0 到 9 的迭代器
        self.iter = iter(range(self.n))

    def __iter__(self):
        # 返回自身作为迭代器
        return self

    def __next__(self):
        # 返回迭代器的下一个元素
        return next(self.iter)

    def reset(self):
        # 重置迭代器，重新创建一个产生 0 到 9 的迭代器
        self.iter = iter(range(self.n))

    def __len__(self):
        # 返回数据管道中元素的个数
        return self.n


class TestIterDataPipeGraphFastForward(TestCase):
    def _fast_forward_graph_test_helper(
        self, datapipe, fast_forward_fn, expected_res, n_iterations=3, rng=None
    ):
        if rng is None:
            # 如果 RNG 为 None，则创建一个 Torch 随机数生成器
            rng = torch.Generator()
        # 设定随机数种子为 0
        rng = rng.manual_seed(0)
        # 应用随机数种子到数据管道
        torch.utils.data.graph_settings.apply_random_seed(datapipe, rng)

        # Test Case: fast forward works with list
        # 测试用例：使用列表进行快速前进
        rng.manual_seed(0)
        # 使用快速前进函数对数据管道进行操作
        fast_forward_fn(datapipe, n_iterations, rng)
        # 将快速前进后的数据管道转换为列表
        actual_res = list(datapipe)
        # 断言：快速前进后的数据个数应该减少 n_iterations
        self.assertEqual(len(datapipe) - n_iterations, len(actual_res))
        # 断言：快速前进后的数据应该与预期结果匹配
        self.assertEqual(expected_res[n_iterations:], actual_res)

        # Test Case: fast forward works with iterator
        # 测试用例：使用迭代器进行快速前进
        rng.manual_seed(0)
        # 使用快速前进函数对数据管道进行操作
        fast_forward_fn(datapipe, n_iterations, rng)
        # 获取数据管道的迭代器
        it = iter(datapipe)
        # 将迭代器转换为列表
        actual_res = list(it)
        # 断言：快速前进后的数据个数应该减少 n_iterations
        self.assertEqual(len(datapipe) - n_iterations, len(actual_res))
        # 断言：快速前进后的数据应该与预期结果匹配
        self.assertEqual(expected_res[n_iterations:], actual_res)
        # 断言：迭代器应该触发 StopIteration 异常
        with self.assertRaises(StopIteration):
            next(it)
    # 定义测试函数，用于测试简单快照图形的行为
    def test_simple_snapshot_graph(self):
        # 创建一个包装了 0 到 9 的可迭代对象 graph1
        graph1 = dp.iter.IterableWrapper(range(10))
        # 预期的结果是包含 0 到 9 的列表
        res1 = list(range(10))
        # 调用辅助函数，测试图形快照的恢复行为，期望结果是 res1
        self._fast_forward_graph_test_helper(
            graph1, _simple_graph_snapshot_restoration, expected_res=res1
        )

        # 对 graph1 中的每个元素乘以 10，创建 graph2
        graph2 = graph1.map(_mul_10)
        # 预期结果是将 res1 中的每个元素乘以 10 得到的列表
        res2 = [10 * x for x in res1]
        # 再次调用测试辅助函数，期望结果是 res2
        self._fast_forward_graph_test_helper(
            graph2, _simple_graph_snapshot_restoration, expected_res=res2
        )

        # 使用 torch.Generator 创建一个随机数生成器 rng
        rng = torch.Generator()
        # 对 graph2 进行随机洗牌，得到 graph3
        graph3 = graph2.shuffle()
        # 手动设置随机种子为 0
        rng.manual_seed(0)
        # 将随机种子应用到 graph3 上
        torch.utils.data.graph_settings.apply_random_seed(graph3, rng)
        # 预期结果是 graph3 中的所有元素组成的列表
        res3 = list(graph3)
        # 再次调用测试辅助函数，期望结果是 res3
        self._fast_forward_graph_test_helper(
            graph3, _simple_graph_snapshot_restoration, expected_res=res3
        )

        # 对 graph3 中的每个元素乘以 10，创建 graph4
        graph4 = graph3.map(_mul_10)
        # 预期结果是将 res3 中的每个元素乘以 10 得到的列表
        res4 = [10 * x for x in res3]
        # 再次调用测试辅助函数，期望结果是 res4
        self._fast_forward_graph_test_helper(
            graph4, _simple_graph_snapshot_restoration, expected_res=res4
        )

        # 设定批处理大小为 2，将 graph4 分成多个大小为 batch_size 的图形，创建 graph5
        batch_size = 2
        graph5 = graph4.batch(batch_size)
        # 预期结果是将 res4 划分为每个子列表大小为 batch_size 的列表
        res5 = [
            res4[i : i + batch_size] for i in range(0, len(res4), batch_size)
        ]  # .batch(2)
        # 再次调用测试辅助函数，期望结果是 res5
        self._fast_forward_graph_test_helper(
            graph5, _simple_graph_snapshot_restoration, expected_res=res5
        )

        # 使用 fork 方法将 graph5 分成两部分 cdp1 和 cdp2
        cdp1, cdp2 = graph5.fork(2)
        # 将 cdp1 和 cdp2 进行 zip 操作，创建 graph6
        graph6 = cdp1.zip(cdp2)
        # 重新设置随机种子为 100
        rng = rng.manual_seed(100)
        # 将新的随机种子应用到 graph6 上
        torch.utils.data.graph_settings.apply_random_seed(graph6, rng)
        # 预期结果是将 res5 中的每个元素与自身组成的元组列表
        res6 = [(x, x) for x in res5]
        # 再次调用测试辅助函数，期望结果是 res6
        self._fast_forward_graph_test_helper(
            graph6, _simple_graph_snapshot_restoration, expected_res=res6
        )

        # 使用 fork 方法将 graph5 分成两部分 cdp1 和 cdp2
        cdp1, cdp2 = graph5.fork(2)
        # 将 cdp1 和 cdp2 进行 concat 操作，创建 graph7
        graph7 = cdp1.concat(cdp2)
        # 预期结果是将 res5 重复两次组成的列表
        res7 = res5 * 2
        # 再次调用测试辅助函数，期望结果是 res7
        self._fast_forward_graph_test_helper(
            graph7, _simple_graph_snapshot_restoration, expected_res=res7
        )

        # 如果图形已经恢复，使用 `fork` 和 `zip` 将会引发异常
        with self.assertRaisesRegex(
            RuntimeError, "Snapshot restoration cannot be applied."
        ):
            # 两次尝试使用 _simple_graph_snapshot_restoration 恢复 graph7
            _simple_graph_snapshot_restoration(graph7, 1)
            _simple_graph_snapshot_restoration(graph7, 1)

    # 测试使用自定义非生成器的简单快照
    def test_simple_snapshot_custom_non_generator(self):
        # 创建一个自定义非生成器的测试数据管道 graph
        graph = _CustomNonGeneratorTestDataPipe()
        # 调用辅助函数，测试图形快照的恢复行为，期望结果是包含 0 到 9 的范围对象
        self._fast_forward_graph_test_helper(
            graph, _simple_graph_snapshot_restoration, expected_res=range(10)
        )

    # 测试使用自定义具有自身 next 方法的简单快照
    def test_simple_snapshot_custom_self_next(self):
        # 创建一个具有自身 next 方法的自定义测试数据管道 graph
        graph = _CustomSelfNextTestDataPipe()
        # 调用辅助函数，测试图形快照的恢复行为，期望结果是包含 0 到 9 的范围对象
        self._fast_forward_graph_test_helper(
            graph, _simple_graph_snapshot_restoration, expected_res=range(10)
        )
    # 定义测试辅助函数，用于测试数据管道的快照、序列化和反序列化
    def _snapshot_test_helper(self, datapipe, expected_res, n_iter=3, rng=None):
        """
        Extend the previous test with serialization and deserialization test.
        扩展之前的测试，增加序列化和反序列化测试。
        """
        # 如果随机数生成器未提供，则使用默认生成器，并设置种子为0
        if rng is None:
            rng = torch.Generator()
        rng.manual_seed(0)
        # 将随机种子应用于数据管道
        torch.utils.data.graph_settings.apply_random_seed(datapipe, rng)
        # 创建数据管道的迭代器
        it = iter(datapipe)
        # 迭代执行数据管道中的内容指定次数
        for _ in range(n_iter):
            next(it)
        # 将数据管道序列化为字节流
        serialized_graph = pickle.dumps(datapipe)
        # 从序列化的字节流中反序列化回数据管道对象
        deserialized_graph = pickle.loads(serialized_graph)
        # 断言：检查原始数据管道已生成样本数量是否符合预期
        self.assertEqual(n_iter, datapipe._number_of_samples_yielded)
        # 断言：检查反序列化后的数据管道生成样本数量是否符合预期
        self.assertEqual(n_iter, deserialized_graph._number_of_samples_yielded)

        # 为反序列化后的数据管道创建新的随机数生成器
        rng_for_deserialized = torch.Generator()
        rng_for_deserialized.manual_seed(0)
        # 使用新生成的随机数生成器恢复简单图的快照
        _simple_graph_snapshot_restoration(
            deserialized_graph, n_iter, rng=rng_for_deserialized
        )
        # 断言：检查反序列化后的数据管道生成的结果是否与预期一致
        self.assertEqual(expected_res[n_iter:], list(it))
        # 断言：检查反序列化后的数据管道生成的结果是否与预期一致
        self.assertEqual(expected_res[n_iter:], list(deserialized_graph))

    # 测试函数：测试带有序列化功能的简单快照图
    def test_simple_snapshot_graph_with_serialization(self):
        # 创建一个迭代器封装的数据管道，范围为0到9
        graph1 = dp.iter.IterableWrapper(range(10))
        # 生成预期结果的列表
        res1 = list(range(10))
        # 调用测试辅助函数，测试数据管道快照和序列化，比较结果是否符合预期
        self._snapshot_test_helper(graph1, expected_res=res1)

        # 对数据管道进行映射操作，每个元素乘以10
        graph2 = graph1.map(_mul_10)
        # 生成预期结果的列表
        res2 = [10 * x for x in res1]
        # 调用测试辅助函数，测试映射后的数据管道快照和序列化，比较结果是否符合预期
        self._snapshot_test_helper(graph2, expected_res=res2)

        # 创建一个新的随机数生成器
        rng = torch.Generator()
        # 对数据管道进行洗牌操作
        graph3 = graph2.shuffle()
        # 设置随机数生成器的种子为0
        rng.manual_seed(0)
        # 将随机种子应用于洗牌后的数据管道
        torch.utils.data.graph_settings.apply_random_seed(graph3, rng)
        # 生成预期结果的列表
        res3 = list(graph3)
        # 调用测试辅助函数，测试洗牌后的数据管道快照和序列化，比较结果是否符合预期
        self._snapshot_test_helper(graph3, expected_res=res3)

        # 对数据管道进行映射操作，每个元素乘以10
        graph4 = graph3.map(_mul_10)
        # 生成预期结果的列表
        res4 = [10 * x for x in res3]
        # 调用测试辅助函数，测试映射后的数据管道快照和序列化，比较结果是否符合预期
        self._snapshot_test_helper(graph4, expected_res=res4)

        # 设置批处理大小为2
        batch_size = 2
        # 将数据管道分批处理
        graph5 = graph4.batch(batch_size)
        # 生成预期结果的列表
        res5 = [
            res4[i : i + batch_size] for i in range(0, len(res4), batch_size)
        ]  # .batch(2)
        # 调用测试辅助函数，测试分批处理后的数据管道快照和序列化，比较结果是否符合预期
        self._snapshot_test_helper(graph5, expected_res=res5)

        # 使用 fork 和 zip 操作创建两个数据管道
        cdp1, cdp2 = graph5.fork(2)
        # 将两个数据管道进行配对操作
        graph6 = cdp1.zip(cdp2)
        # 生成预期结果的列表
        res6 = [(x, x) for x in res5]
        # 调用测试辅助函数，测试配对后的数据管道快照和序列化，比较结果是否符合预期
        self._snapshot_test_helper(graph6, expected_res=res6)

        # 使用 fork 和 concat 操作创建一个数据管道
        graph7 = cdp1.concat(cdp2)
        # 生成预期结果的列表
        res7 = res5 * 2
        # 调用测试辅助函数，测试连接后的数据管道快照和序列化，比较结果是否符合预期
        self._snapshot_test_helper(graph7, expected_res=res7)
    # 定义一个测试函数，用于测试简单快照图的重复性
    def test_simple_snapshot_graph_repeated(self):
        # 使用 IterableWrapper 将范围为 0 到 9 的整数转换为可迭代对象
        cdp1, cdp2 = (
            dp.iter.IterableWrapper(range(10))
            # 对每个元素乘以 10
            .map(_mul_10)
            # 随机打乱元素顺序
            .shuffle()
            # 再次对每个元素乘以 10
            .map(_mul_10)
            .map(_mul_10)
            # 将流分成两部分
            .fork(2)
        )
        # 创建一个图对象，将两个部分合并成一个图
        graph = cdp1.zip(cdp2)

        # 创建一个随机数生成器，并设置种子为 0
        rng = torch.Generator()
        rng.manual_seed(0)
        # 应用随机种子到图对象上
        torch.utils.data.graph_settings.apply_random_seed(graph, rng)

        # 获取预期的结果列表
        expected_res = list(graph)

        # 重新设置种子为 0，并再次应用到图对象上
        rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(graph, rng)
        # 创建图对象的迭代器，并迭代若干次
        it = iter(graph)
        n_iter = 3
        for _ in range(n_iter):
            next(it)

        # 进行首次序列化和反序列化操作
        serialized_graph = pickle.dumps(graph)
        deserialized_graph = pickle.loads(serialized_graph)

        # 为反序列化后的图对象创建新的随机数生成器，并设置种子为 0
        rng_for_deserialized = torch.Generator()
        rng_for_deserialized.manual_seed(0)
        # 使用简单图快照恢复函数，恢复图对象状态
        _simple_graph_snapshot_restoration(
            deserialized_graph,
            deserialized_graph._number_of_samples_yielded,
            rng=rng_for_deserialized,
        )

        # 创建反序列化后图对象的新迭代器，并获取下一个元素，确保其与预期值相符
        it = iter(deserialized_graph)
        self.assertEqual(expected_res[3], next(it))

        # 再次进行序列化/反序列化，并快进以确保其正常工作
        serialized_graph2 = pickle.dumps(deserialized_graph)
        deserialized_graph2 = pickle.loads(serialized_graph2)

        # 为第二个反序列化后的图对象创建新的随机数生成器，并设置种子为 0
        rng_for_deserialized = torch.Generator()
        rng_for_deserialized.manual_seed(0)
        # 使用简单图快照恢复函数，再次恢复图对象状态
        _simple_graph_snapshot_restoration(
            deserialized_graph2,
            deserialized_graph._number_of_samples_yielded,
            rng=rng_for_deserialized,
        )

        # 获取下一个元素，并确保其与预期值相符
        self.assertEqual(expected_res[4:], list(deserialized_graph2))
# 如果当前脚本作为主程序运行（而不是作为模块导入到其他程序中），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```