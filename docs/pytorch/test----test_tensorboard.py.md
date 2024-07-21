# `.\pytorch\test\test_tensorboard.py`

```py
# Owner(s): ["module: unknown"]

# 导入必要的模块和库
import io               # 提供了对文件和字符串的 I/O 支持
import os               # 提供了与操作系统交互的功能
import shutil           # 提供了高级文件操作功能，如复制、移动和删除
import sys              # 提供了对 Python 解释器的访问和控制
import tempfile         # 提供了临时文件和目录的支持
import unittest         # 提供了单元测试框架
from pathlib import Path  # 提供了处理文件路径的类和函数

import expecttest       # 引入了一个自定义的 expecttest 模块
import numpy as np      # 引入了 NumPy 库，用于科学计算

TEST_TENSORBOARD = True  # 是否测试 TensorBoard 的标志，默认为 True
try:
    import tensorboard.summary.writer.event_file_writer  # 尝试导入 TensorBoard 的事件文件写入器模块
    from tensorboard.compat.proto.summary_pb2 import Summary  # 导入 TensorBoard 的 Summary 类
except ImportError:
    TEST_TENSORBOARD = False  # 如果导入失败，则将 TEST_TENSORBOARD 设为 False

HAS_TORCHVISION = True  # 是否安装了 torchvision 的标志，默认为 True
try:
    import torchvision  # 尝试导入 torchvision 模块
except ImportError:
    HAS_TORCHVISION = False  # 如果导入失败，则将 HAS_TORCHVISION 设为 False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")  # 如果没有安装 torchvision，则跳过测试

TEST_MATPLOTLIB = True  # 是否测试 Matplotlib 的标志，默认为 True
try:
    import matplotlib  # 尝试导入 Matplotlib 库
    if os.environ.get('DISPLAY', '') == '':
        matplotlib.use('Agg')  # 如果没有显示环境，则使用 Agg 后端
    import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块
except ImportError:
    TEST_MATPLOTLIB = False  # 如果导入失败，则将 TEST_MATPLOTLIB 设为 False
skipIfNoMatplotlib = unittest.skipIf(not TEST_MATPLOTLIB, "no matplotlib")  # 如果没有安装 Matplotlib，则跳过测试

import torch  # 导入 PyTorch 深度学习库
from torch.testing._internal.common_utils import (  # 导入 PyTorch 内部测试工具模块的多个函数和类
    instantiate_parametrized_tests,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TEST_WITH_CROSSREF,
    TestCase,
)


def tensor_N(shape, dtype=float):
    # 创建一个形状为 shape 的 N 维张量，并初始化为从 0 到 numel-1 的值
    numel = np.prod(shape)
    x = (np.arange(numel, dtype=dtype)).reshape(shape)
    return x

class BaseTestCase(TestCase):
    """ Base class used for all TensorBoard tests """
    def setUp(self):
        super().setUp()
        if not TEST_TENSORBOARD:
            return self.skipTest("Skip the test since TensorBoard is not installed")  # 如果没有安装 TensorBoard，则跳过测试
        if TEST_WITH_CROSSREF:
            return self.skipTest("Don't run TensorBoard tests with crossref")  # 如果设置了跨引用测试标志，则跳过 TensorBoard 测试
        self.temp_dirs = []  # 用于存储临时目录的列表

    def createSummaryWriter(self):
        # 创建一个用于写入摘要的临时目录，tearDown() 负责清理
        temp_dir = tempfile.TemporaryDirectory(prefix="test_tensorboard").name
        self.temp_dirs.append(temp_dir)
        return SummaryWriter(temp_dir)  # 返回一个用于写入摘要的 SummaryWriter 对象

    def tearDown(self):
        super().tearDown()
        # 清理 SummaryWriter 创建的临时目录
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

if TEST_TENSORBOARD:
    from google.protobuf import text_format  # 导入 Google Protobuf 的文本格式模块
    from PIL import Image  # 导入 PIL 库的 Image 模块
    from tensorboard.compat.proto.graph_pb2 import GraphDef  # 导入 TensorBoard 的 GraphDef 类
    from tensorboard.compat.proto.types_pb2 import DataType  # 导入 TensorBoard 的 DataType 类

    from torch.utils.tensorboard import summary, SummaryWriter  # 导入 PyTorch 的 TensorBoard 摘要和摘要写入器
    from torch.utils.tensorboard._convert_np import make_np  # 导入 PyTorch 的 _convert_np 模块
    from torch.utils.tensorboard._pytorch_graph import graph  # 导入 PyTorch 的 _pytorch_graph 模块
    from torch.utils.tensorboard._utils import _prepare_video, convert_to_HWC  # 导入 PyTorch 的 _utils 模块中的函数
    from torch.utils.tensorboard.summary import int_to_half, tensor_proto  # 导入 PyTorch 的 summary 模块中的函数和类

class TestTensorBoardPyTorchNumpy(BaseTestCase):
    # 测试函数，验证将 PyTorch 张量转换为 NumPy 数组的功能
    def test_pytorch_np(self):
        # 创建几个不同类型的 PyTorch 张量列表
        tensors = [torch.rand(3, 10, 10), torch.rand(1), torch.rand(1, 2, 3, 4, 5)]
        for tensor in tensors:
            # 检查 make_np 函数是否正确将普通张量转换为 NumPy 数组
            self.assertIsInstance(make_np(tensor), np.ndarray)

            # 如果 CUDA 可用，检查 make_np 函数是否正确处理 CUDA 张量转换为 NumPy 数组
            if torch.cuda.is_available():
                self.assertIsInstance(make_np(tensor.cuda()), np.ndarray)

            # 检查 make_np 函数是否正确将普通变量（Variable）转换为 NumPy 数组
            self.assertIsInstance(make_np(torch.autograd.Variable(tensor)), np.ndarray)

            # 如果 CUDA 可用，检查 make_np 函数是否正确处理 CUDA 变量（Variable）转换为 NumPy 数组
            if torch.cuda.is_available():
                self.assertIsInstance(make_np(torch.autograd.Variable(tensor).cuda()), np.ndarray)

        # 检查 make_np 函数是否正确处理 Python 原始类型转换为 NumPy 数组
        self.assertIsInstance(make_np(0), np.ndarray)
        self.assertIsInstance(make_np(0.1), np.ndarray)

    # 测试函数，验证将 PyTorch 自动求导变量转换为 NumPy 数组的功能
    def test_pytorch_autograd_np(self):
        # 创建一个 PyTorch 自动求导变量
        x = torch.autograd.Variable(torch.empty(1))
        # 检查 make_np 函数是否正确将自动求导变量转换为 NumPy 数组
        self.assertIsInstance(make_np(x), np.ndarray)

    # 测试函数，验证在 PyTorch 中写入操作的功能
    def test_pytorch_write(self):
        # 使用创建的 SummaryWriter 对象进行写入操作，添加标量值
        with self.createSummaryWriter() as w:
            w.add_scalar('scalar', torch.autograd.Variable(torch.rand(1)), 0)

    # 测试函数，验证在 PyTorch 中添加直方图操作的功能
    def test_pytorch_histogram(self):
        # 使用创建的 SummaryWriter 对象进行直方图写入操作，添加浮点数直方图
        with self.createSummaryWriter() as w:
            w.add_histogram('float histogram', torch.rand((50,)))
            # 添加整数直方图
            w.add_histogram('int histogram', torch.randint(0, 100, (50,)))
            # 添加 bfloat16 类型的直方图
            w.add_histogram('bfloat16 histogram', torch.rand(50, dtype=torch.bfloat16))
    # 定义一个测试方法，用于测试直接使用 PyTorch 的直方图原始数据功能
    def test_pytorch_histogram_raw(self):
        # 创建一个用于记录数据的 SummaryWriter 对象，并使用上下文管理器确保资源正确释放
        with self.createSummaryWriter() as w:
            # 定义生成随机浮点数的数量
            num = 50
            # 使用 make_np 函数将 PyTorch 随机生成的浮点数转换为 NumPy 数组
            floats = make_np(torch.rand((num,)))
            # 定义直方图的分桶边界
            bins = [0.0, 0.25, 0.5, 0.75, 1.0]
            # 使用 NumPy 的直方图函数计算浮点数的分桶计数和边界
            counts, limits = np.histogram(floats, bins)
            # 计算浮点数的平方和
            sum_sq = floats.dot(floats).item()
            # 将直方图原始数据添加到 SummaryWriter 中
            w.add_histogram_raw('float histogram raw',
                                min=floats.min().item(),
                                max=floats.max().item(),
                                num=num,
                                sum=floats.sum().item(),
                                sum_squares=sum_sq,
                                bucket_limits=limits[1:].tolist(),
                                bucket_counts=counts.tolist())

            # 使用 make_np 函数将 PyTorch 随机生成的整数转换为 NumPy 数组
            ints = make_np(torch.randint(0, 100, (num,)))
            # 重新定义整数直方图的分桶边界
            bins = [0, 25, 50, 75, 100]
            # 使用 NumPy 的直方图函数计算整数的分桶计数和边界
            counts, limits = np.histogram(ints, bins)
            # 计算整数的平方和
            sum_sq = ints.dot(ints).item()
            # 将整数直方图原始数据添加到 SummaryWriter 中
            w.add_histogram_raw('int histogram raw',
                                min=ints.min().item(),
                                max=ints.max().item(),
                                num=num,
                                sum=ints.sum().item(),
                                sum_squares=sum_sq,
                                bucket_limits=limits[1:].tolist(),
                                bucket_counts=counts.tolist())

            # 生成从 0 到 99 的浮点数张量
            ints = torch.tensor(range(0, 100)).float()
            # 定义直方图的分桶数量
            nbins = 100
            # 使用 PyTorch 的 histc 函数计算浮点数张量的直方图
            counts = torch.histc(ints, bins=nbins, min=0, max=99)
            # 定义直方图的分桶边界
            limits = torch.tensor(range(nbins))
            # 计算浮点数张量的平方和
            sum_sq = ints.dot(ints).item()
            # 将浮点数张量的直方图原始数据添加到 SummaryWriter 中
            w.add_histogram_raw('int histogram raw',
                                min=ints.min().item(),
                                max=ints.max().item(),
                                num=num,
                                sum=ints.sum().item(),
                                sum_squares=sum_sq,
                                bucket_limits=limits.tolist(),
                                bucket_counts=counts.tolist())
class TestTensorBoardUtils(BaseTestCase):
    def test_to_HWC(self):
        # 创建一个随机的三维 uint8 类型的 numpy 数组作为测试图像
        test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        # 调用函数 convert_to_HWC 将图像从 'chw' 格式转换为 'HWC' 格式
        converted = convert_to_HWC(test_image, 'chw')
        # 断言转换后的图像形状应为 (32, 32, 3)
        self.assertEqual(converted.shape, (32, 32, 3))
        
        # 创建一个随机的四维 uint8 类型的 numpy 数组作为测试图像
        test_image = np.random.randint(0, 256, size=(16, 3, 32, 32), dtype=np.uint8)
        # 调用函数 convert_to_HWC 将图像从 'nchw' 格式转换为 'HWC' 格式
        converted = convert_to_HWC(test_image, 'nchw')
        # 断言转换后的图像形状应为 (64, 256, 3)
        self.assertEqual(converted.shape, (64, 256, 3))
        
        # 创建一个随机的二维 uint8 类型的 numpy 数组作为测试图像
        test_image = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
        # 调用函数 convert_to_HWC 将图像从 'hw' 格式转换为 'HWC' 格式
        converted = convert_to_HWC(test_image, 'hw')
        # 断言转换后的图像形状应为 (32, 32, 3)
        self.assertEqual(converted.shape, (32, 32, 3))

    def test_convert_to_HWC_dtype_remains_same(self):
        # 测试确保 convert_to_HWC 函数能够恢复输入 np 数组的 dtype
        # 因此，图像的缩放因子应为 1
        test_image = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]], dtype=torch.uint8)
        # 调用 make_np 函数将 torch 张量转换为 numpy 数组
        tensor = make_np(test_image)
        # 调用 convert_to_HWC 函数将图像从 'NCHW' 格式转换为 'HWC' 格式
        tensor = convert_to_HWC(tensor, 'NCHW')
        # 计算图像的缩放因子
        scale_factor = summary._calc_scale_factor(tensor)
        # 断言图像的缩放因子应为 1
        self.assertEqual(scale_factor, 1, msg='Values are already in [0, 255], scale factor should be 1')

    def test_prepare_video(self):
        # 在每个时间帧上，视频所有其他维度的总和应该相同
        shapes = [
            (16, 30, 3, 28, 28),
            (36, 30, 3, 28, 28),
            (19, 29, 3, 23, 19),
            (3, 3, 3, 3, 3)
        ]
        # 遍历不同形状的视频输入
        for s in shapes:
            # 创建一个随机的 s 形状的 numpy 数组作为视频输入
            V_input = np.random.random(s)
            # 调用 _prepare_video 函数处理视频输入，返回处理后的视频数据
            V_after = _prepare_video(np.copy(V_input))
            # 获取视频的总帧数
            total_frame = s[1]
            # 交换视频输入的前两个轴
            V_input = np.swapaxes(V_input, 0, 1)
            # 遍历每一帧
            for f in range(total_frame):
                # 将视频输入的每一帧重塑为一维数组 x
                x = np.reshape(V_input[f], newshape=(-1))
                # 将处理后的视频的每一帧重塑为一维数组 y
                y = np.reshape(V_after[f], newshape=(-1))
                # 断言 x 和 y 的元素总和应该几乎相等
                np.testing.assert_array_almost_equal(np.sum(x), np.sum(y))

    def test_numpy_vid_uint8(self):
        # 创建一个随机的四维 uint8 类型的 numpy 数组作为视频输入
        V_input = np.random.randint(0, 256, (16, 30, 3, 28, 28)).astype(np.uint8)
        # 调用 _prepare_video 函数处理视频输入，并将处理后的数据乘以 255
        V_after = _prepare_video(np.copy(V_input)) * 255
        # 获取视频的总帧数
        total_frame = V_input.shape[1]
        # 交换视频输入的前两个轴
        V_input = np.swapaxes(V_input, 0, 1)
        # 遍历每一帧
        for f in range(total_frame):
            # 将视频输入的每一帧重塑为一维数组 x
            x = np.reshape(V_input[f], newshape=(-1))
            # 将处理后的视频的每一帧重塑为一维数组 y
            y = np.reshape(V_after[f], newshape=(-1))
            # 断言 x 和 y 的元素总和应该几乎相等
            np.testing.assert_array_almost_equal(np.sum(x), np.sum(y))

freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

true_positive_counts = [75, 64, 21, 5, 0]
false_positive_counts = [150, 105, 18, 0, 0]
true_negative_counts = [0, 45, 132, 150, 150]
false_negative_counts = [0, 11, 54, 70, 75]
precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]

class TestTensorBoardWriter(BaseTestCase):
    # 定义一个测试方法，用于测试写入功能
    def test_writer(self):
        # 创建一个摘要写入器
        with self.createSummaryWriter() as writer:
            # 设置采样率为44100
            sample_rate = 44100

            # 初始化迭代次数为0
            n_iter = 0

            # 添加超参数记录
            writer.add_hparams(
                {'lr': 0.1, 'bsize': 1},  # 超参数字典
                {'hparam/accuracy': 10, 'hparam/loss': 10}  # 指标字典
            )

            # 添加标量数据记录，记录系统时间的标量数据
            writer.add_scalar('data/scalar_systemtime', 0.1, n_iter)

            # 添加带有自定义时间的标量数据记录
            writer.add_scalar('data/scalar_customtime', 0.2, n_iter, walltime=n_iter)

            # 添加新风格的标量数据记录，指定使用新风格
            writer.add_scalar('data/new_style', 0.2, n_iter, new_style=True)

            # 添加一组标量数据记录，包括 xsinx、xcosx 和 arctanx
            writer.add_scalars('data/scalar_group', {
                "xsinx": n_iter * np.sin(n_iter),
                "xcosx": n_iter * np.cos(n_iter),
                "arctanx": np.arctan(n_iter)
            }, n_iter)

            # 创建一个32x3x64x64的零数组 x，代表网络输出
            x = np.zeros((32, 3, 64, 64))

            # 添加图像数据记录，用于显示输出图像
            writer.add_images('Image', x, n_iter)

            # 添加带框的图像数据记录，用于显示带有框的图像
            writer.add_image_with_boxes('imagebox',
                                        np.zeros((3, 64, 64)),  # 图像数据
                                        np.array([[10, 10, 40, 40], [40, 40, 60, 60]]),  # 框坐标
                                        n_iter)  # 迭代次数

            # 创建一个长度为 sample_rate*2 的零数组 x，代表音频数据
            x = np.zeros(sample_rate * 2)

            # 添加音频数据记录，记录音频数据
            writer.add_audio('myAudio', x, n_iter)

            # 添加视频数据记录，记录视频数据
            writer.add_video('myVideo', np.random.rand(16, 48, 1, 28, 28).astype(np.float32), n_iter)

            # 添加文本数据记录，记录文本数据
            writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)

            # 添加 Markdown 格式的文本数据记录，记录 Markdown 格式的文本
            writer.add_text('markdown Text', '''a|b\n-|-\nc|d''', n_iter)

            # 添加直方图数据记录，记录直方图数据
            writer.add_histogram('hist', np.random.rand(100, 100), n_iter)

            # 添加 PR 曲线数据记录，记录 PR 曲线数据
            writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)

            # 添加原始数据的 PR 曲线数据记录，记录原始数据的 PR 曲线数据
            writer.add_pr_curve_raw('prcurve with raw data', true_positive_counts,
                                    false_positive_counts,
                                    true_negative_counts,
                                    false_negative_counts,
                                    precision,
                                    recall, n_iter)

            # 创建一个1x4x3的顶点数组 v，一个1x4x3的颜色数组 c，一个1x4x3的面数组 f，代表网格数据
            v = np.array([[[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]]], dtype=float)
            c = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]], dtype=int)
            f = np.array([[[0, 2, 3], [0, 3, 1], [0, 1, 2], [1, 3, 2]]], dtype=int)

            # 添加网格数据记录，记录网格数据
            writer.add_mesh('my_mesh', vertices=v, colors=c, faces=f)
class TestTensorBoardSummaryWriter(BaseTestCase):
    # 定义测试类 TestTensorBoardSummaryWriter，继承自 BaseTestCase

    def test_summary_writer_ctx(self):
        # 测试函数：测试 SummaryWriter 作为上下文管理器使用后应该被关闭

        with self.createSummaryWriter() as writer:
            # 使用 createSummaryWriter 方法创建 SummaryWriter 实例，并作为上下文管理器使用

            writer.add_scalar('test', 1)
            # 向 SummaryWriter 中添加一个名为 'test'，值为 1 的标量数据

        self.assertIs(writer.file_writer, None)
        # 断言 writer 的 file_writer 属性为 None，验证 SummaryWriter 已被关闭

    def test_summary_writer_close(self):
        # 测试函数：频繁打开和关闭 SummaryWriter 不应该导致 "OSError: [Errno 24] Too many open files"

        passed = True
        try:
            writer = self.createSummaryWriter()
            # 创建 SummaryWriter 实例

            writer.close()
            # 关闭 SummaryWriter 实例

        except OSError:
            passed = False
            # 捕获 OSError 异常，将 passed 设置为 False

        self.assertTrue(passed)
        # 断言 passed 为 True，验证没有遇到 "Too many open files" 错误

    def test_pathlib(self):
        # 测试函数：测试使用 pathlib 的 SummaryWriter 功能

        with tempfile.TemporaryDirectory(prefix="test_tensorboard_pathlib") as d:
            # 创建临时目录，并将其路径存储在变量 d 中

            p = Path(d)
            # 将路径 d 转换为 Path 对象 p

            with SummaryWriter(p) as writer:
                # 使用路径 p 创建 SummaryWriter 实例，并作为上下文管理器使用

                writer.add_scalar('test', 1)
                # 向 SummaryWriter 中添加一个名为 'test'，值为 1 的标量数据



class TestTensorBoardEmbedding(BaseTestCase):
    # 定义测试类 TestTensorBoardEmbedding，继承自 BaseTestCase

    def test_embedding(self):
        # 测试函数：测试向 SummaryWriter 添加嵌入数据

        w = self.createSummaryWriter()
        # 创建 SummaryWriter 实例 w

        all_features = torch.tensor([[1., 2., 3.], [5., 4., 1.], [3., 7., 7.]])
        # 创建包含特征数据的 Tensor 对象 all_features

        all_labels = torch.tensor([33., 44., 55.])
        # 创建包含标签数据的 Tensor 对象 all_labels

        all_images = torch.zeros(3, 3, 5, 5)
        # 创建形状为 (3, 3, 5, 5) 的全零 Tensor 对象 all_images

        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        global_step=2)
        # 向 SummaryWriter 中添加嵌入数据，包括特征、标签、图像，并设置全局步数为 2

        dataset_label = ['test'] * 2 + ['train'] * 2
        # 创建包含数据集标签的列表 dataset_label

        all_labels = list(zip(all_labels, dataset_label))
        # 将 all_labels 和 dataset_label 打包成元组列表，并赋值给 all_labels

        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        metadata_header=['digit', 'dataset'],
                        global_step=2)
        # 再次向 SummaryWriter 中添加嵌入数据，包括特征、元数据标签、图像，设置元数据头为 ['digit', 'dataset']，全局步数为 2

    def test_embedding_64(self):
        # 测试函数：测试向 SummaryWriter 添加使用 float64 数据类型的嵌入数据

        w = self.createSummaryWriter()
        # 创建 SummaryWriter 实例 w

        all_features = torch.tensor([[1., 2., 3.], [5., 4., 1.], [3., 7., 7.]])
        # 创建包含特征数据的 Tensor 对象 all_features

        all_labels = torch.tensor([33., 44., 55.])
        # 创建包含标签数据的 Tensor 对象 all_labels

        all_images = torch.zeros((3, 3, 5, 5), dtype=torch.float64)
        # 创建使用 float64 数据类型的全零 Tensor 对象 all_images

        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        global_step=2)
        # 向 SummaryWriter 中添加嵌入数据，包括特征、标签、图像，并设置全局步数为 2

        dataset_label = ['test'] * 2 + ['train'] * 2
        # 创建包含数据集标签的列表 dataset_label

        all_labels = list(zip(all_labels, dataset_label))
        # 将 all_labels 和 dataset_label 打包成元组列表，并赋值给 all_labels

        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        metadata_header=['digit', 'dataset'],
                        global_step=2)
        # 再次向 SummaryWriter 中添加嵌入数据，包括特征、元数据标签、图像，设置元数据头为 ['digit', 'dataset']，全局步数为 2

class TestTensorBoardSummary(BaseTestCase):
    # 定义测试类 TestTensorBoardSummary，继承自 BaseTestCase

    def test_uint8_image(self):
        # 测试函数：测试 uint8 图像（像素值在 [0, 255]）不被修改

        '''
        Tests that uint8 image (pixel values in [0, 255]) is not changed
        '''

        test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        # 创建形状为 (3, 32, 32)，类型为 uint8 的随机图像数据 test_image

        scale_factor = summary._calc_scale_factor(test_image)
        # 调用 _calc_scale_factor 方法计算 test_image 的缩放因子，并将结果赋值给 scale_factor

        self.assertEqual(scale_factor, 1, msg='Values are already in [0, 255], scale factor should be 1')
        # 断言 scale_factor 等于 1，验证图像数据已在 [0, 255] 范围内，缩放因子应为 1
    def test_float32_image(self):
        '''
        Tests that float32 image (pixel values in [0, 1]) are scaled correctly
        to [0, 255]
        '''
        # 创建一个随机的 float32 图像，大小为 3x32x32，并转换为 np.float32 类型
        test_image = np.random.rand(3, 32, 32).astype(np.float32)
        # 计算将 [0, 1] 范围内的像素值缩放到 [0, 255] 的比例因子
        scale_factor = summary._calc_scale_factor(test_image)
        # 断言缩放因子应该是 255，因为像素值被缩放到了 [0, 255] 范围内
        self.assertEqual(scale_factor, 255, msg='Values are in [0, 1], scale factor should be 255')

    def test_list_input(self):
        # 断言直方图函数对列表输入会抛出异常
        with self.assertRaises(Exception) as e_info:
            summary.histogram('dummy', [1, 3, 4, 5, 6], 'tensorflow')

    def test_empty_input(self):
        # 断言直方图函数对空的 ndarray 输入会抛出异常
        with self.assertRaises(Exception) as e_info:
            summary.histogram('dummy', np.ndarray(0), 'tensorflow')

    def test_image_with_boxes(self):
        # 断言图像带有边界框的输出与预期的图像 proto 数据相匹配
        self.assertTrue(compare_image_proto(summary.image_boxes('dummy',
                                            tensor_N(shape=(3, 32, 32)),
                                            np.array([[10, 10, 40, 40]])),
                                            self))

    def test_image_with_one_channel(self):
        # 断言带有单通道的图像输出与预期的图像 proto 数据相匹配
        self.assertTrue(compare_image_proto(
            summary.image('dummy',
                          tensor_N(shape=(1, 8, 8)),
                          dataformats='CHW'),
                          self))  # noqa: E131

    def test_image_with_one_channel_batched(self):
        # 断言带有单通道并批处理的图像输出与预期的图像 proto 数据相匹配
        self.assertTrue(compare_image_proto(
            summary.image('dummy',
                          tensor_N(shape=(2, 1, 8, 8)),
                          dataformats='NCHW'),
                          self))  # noqa: E131

    def test_image_with_3_channel_batched(self):
        # 断言带有三通道并批处理的图像输出与预期的图像 proto 数据相匹配
        self.assertTrue(compare_image_proto(
            summary.image('dummy',
                          tensor_N(shape=(2, 3, 8, 8)),
                          dataformats='NCHW'),
                          self))  # noqa: E131

    def test_image_without_channel(self):
        # 断言没有通道信息的图像输出与预期的图像 proto 数据相匹配
        self.assertTrue(compare_image_proto(
            summary.image('dummy',
                          tensor_N(shape=(8, 8)),
                          dataformats='HW'),
                          self))  # noqa: E131

    def test_video(self):
        try:
            import moviepy  # noqa: F401
        except ImportError:
            return
        # 断言视频数据的 proto 数据与预期的数据相匹配
        self.assertTrue(compare_proto(summary.video('dummy', tensor_N(shape=(4, 3, 1, 8, 8))), self))
        # 生成随机数据的视频 proto 数据
        summary.video('dummy', np.random.rand(16, 48, 1, 28, 28))
        # 生成另一组随机数据的视频 proto 数据
        summary.video('dummy', np.random.rand(20, 7, 1, 8, 8))

    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    def test_audio(self):
        # 断言音频数据的 proto 数据与预期的数据相匹配
        self.assertTrue(compare_proto(summary.audio('dummy', tensor_N(shape=(42,))), self))

    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    def test_text(self):
        # 断言文本数据的 proto 数据与预期的数据相匹配
        self.assertTrue(compare_proto(summary.text('dummy', 'text 123'), self))

    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    # 定义单元测试方法，测试自动生成直方图功能
    def test_histogram_auto(self):
        # 使用 summary.histogram 生成 'dummy' 张量的直方图，自动确定箱数，最大为 5
        self.assertTrue(compare_proto(summary.histogram('dummy', tensor_N(shape=(1024,)), bins='auto', max_bins=5), self))

    # 在 macOS 上跳过该测试，详细信息见链接
    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    # 定义单元测试方法，测试使用 Freedman-Diaconis 准则生成直方图
    def test_histogram_fd(self):
        # 使用 summary.histogram 生成 'dummy' 张量的直方图，使用 Freedman-Diaconis 准则确定箱数，最大为 5
        self.assertTrue(compare_proto(summary.histogram('dummy', tensor_N(shape=(1024,)), bins='fd', max_bins=5), self))

    # 在 macOS 上跳过该测试，详细信息见链接
    # 定义单元测试方法，测试使用 Doane's 准则生成直方图
    def test_histogram_doane(self):
        # 使用 summary.histogram 生成 'dummy' 张量的直方图，使用 Doane's 准则确定箱数，最大为 5
        self.assertTrue(compare_proto(summary.histogram('dummy', tensor_N(shape=(1024,)), bins='doane', max_bins=5), self))

    # 定义单元测试方法，测试自定义标量图
    def test_custom_scalars(self):
        # 定义自定义标量图的布局
        layout = {
            'Taiwan': {
                'twse': ['Multiline', ['twse/0050', 'twse/2330']]
            },
            'USA': {
                'dow': ['Margin', ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                'nasdaq': ['Margin', ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]
            }
        }
        # 调用 summary.custom_scalars 方法展示自定义标量图（仅作为烟雾测试，因为 Python2/3 的 protobuf 序列化方式不同）
        summary.custom_scalars(layout)

    # 在 macOS 上跳过该测试，详细信息见链接
    # 定义单元测试方法，测试网格数据的可视化
    def test_mesh(self):
        # 定义网格的顶点、颜色和面
        v = np.array([[[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]]], dtype=float)
        c = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]], dtype=int)
        f = np.array([[[0, 2, 3], [0, 3, 1], [0, 1, 2], [1, 3, 2]]], dtype=int)
        # 调用 summary.mesh 方法展示名为 'my_mesh' 的网格数据可视化
        mesh = summary.mesh('my_mesh', vertices=v, colors=c, faces=f, config_dict=None)
        self.assertTrue(compare_proto(mesh, self))

    # 在 macOS 上跳过该测试，详细信息见链接
    # 定义单元测试方法，测试新样式的标量数据记录
    def test_scalar_new_style(self):
        # 调用 summary.scalar 方法记录名为 'test_scalar' 的新样式标量数据为 1.0
        scalar = summary.scalar('test_scalar', 1.0, new_style=True)
        self.assertTrue(compare_proto(scalar, self))
        # 使用断言检查无法使用新样式记录包含张量数据的标量数据
        with self.assertRaises(AssertionError):
            summary.scalar('test_scalar2', torch.Tensor([1, 2, 3]), new_style=True)
# 定义函数：移除字符串中的空白字符（空格、制表符、换行符）
def remove_whitespace(string):
    return string.replace(' ', '').replace('\t', '').replace('\n', '')

# 获取预期文件路径函数，根据给定的函数指针确定所在的模块，并找到对应的测试文件路径
def get_expected_file(function_ptr):
    # 获取函数所在模块的名称
    module_id = function_ptr.__class__.__module__
    # 获取模块的文件路径，考虑到 __file__ 可能是编译后的 .pyc 文件，将其改为 .py 文件
    test_file = sys.modules[module_id].__file__
    test_file = ".".join(test_file.split('.')[:-1]) + '.py'

    # 使用 realpath 来跟踪符号链接的真实路径
    test_dir = os.path.dirname(os.path.realpath(test_file))
    # 获取函数名称，并构造预期文件的路径
    functionName = function_ptr.id().split('.')[-1]
    return os.path.join(test_dir, "expect", 'TestTensorBoard.' + functionName + ".expect")

# 读取预期内容函数，根据给定的函数指针获取预期文件的路径，并读取其内容
def read_expected_content(function_ptr):
    expected_file = get_expected_file(function_ptr)
    # 断言预期文件存在
    assert os.path.exists(expected_file), expected_file
    # 打开文件并读取其内容，返回内容作为字符串
    with open(expected_file) as f:
        return f.read()

# 比较图像 proto 的函数，根据实际 proto 和函数指针，与预期值比较
def compare_image_proto(actual_proto, function_ptr):
    if expecttest.ACCEPT:
        # 如果处于接受模式，则将实际 proto 写入预期文件
        expected_file = get_expected_file(function_ptr)
        with open(expected_file, 'w') as f:
            f.write(text_format.MessageToString(actual_proto))
        return True
    
    # 否则，读取预期内容，并将其解析为 proto 对象
    expected_str = read_expected_content(function_ptr)
    expected_proto = Summary()
    text_format.Parse(expected_str, expected_proto)

    # 获取实际和预期的 proto 对象
    [actual, expected] = [actual_proto.value[0], expected_proto.value[0]]
    # 使用 PIL 库打开实际和预期的图像，并比较它们的属性
    actual_img = Image.open(io.BytesIO(actual.image.encoded_image_string))
    expected_img = Image.open(io.BytesIO(expected.image.encoded_image_string))

    # 返回比较结果，包括标签和图像属性的比较
    return (
        actual.tag == expected.tag and
        actual.image.height == expected.image.height and
        actual.image.width == expected.image.width and
        actual.image.colorspace == expected.image.colorspace and
        actual_img == expected_img
    )

# 比较 proto 字符串的函数，根据实际字符串和函数指针，与预期值比较
def compare_proto(str_to_compare, function_ptr):
    if expecttest.ACCEPT:
        # 如果处于接受模式，则将实际 proto 字符串写入预期文件
        write_proto(str_to_compare, function_ptr)
        return True
    
    # 否则，读取预期内容并移除空白字符后，与实际字符串进行比较
    expected = read_expected_content(function_ptr)
    str_to_compare = str(str_to_compare)
    return remove_whitespace(str_to_compare) == remove_whitespace(expected)

# 写入 proto 字符串到文件的函数，根据给定的 proto 字符串和函数指针，将其写入对应的预期文件
def write_proto(str_to_compare, function_ptr):
    expected_file = get_expected_file(function_ptr)
    with open(expected_file, 'w') as f:
        f.write(str(str_to_compare))

# 测试类 TestTensorBoardPytorchGraph，继承自 BaseTestCase
class TestTensorBoardPytorchGraph(BaseTestCase):
    # 定义一个测试函数，用于测试 PyTorch 图的生成
    def test_pytorch_graph(self):
        # 创建一个用于测试的虚拟输入数据
        dummy_input = (torch.zeros(1, 3),)

        # 定义一个自定义的 PyTorch 模块，包含一个线性层
        class myLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.l(x)

        # 创建一个用于记录摘要信息的 SummaryWriter 对象
        with self.createSummaryWriter() as w:
            # 向摘要中添加 myLinear 模块的计算图
            w.add_graph(myLinear(), dummy_input)

        # 调用 graph 函数，生成实际的计算图
        actual_proto, _ = graph(myLinear(), dummy_input)

        # 读取预期的计算图描述信息
        expected_str = read_expected_content(self)
        expected_proto = GraphDef()
        # 解析预期的计算图描述信息
        text_format.Parse(expected_str, expected_proto)

        # 断言预期计算图的节点数与实际计算图的节点数相同
        self.assertEqual(len(expected_proto.node), len(actual_proto.node))
        # 逐个比较每个节点的属性
        for i in range(len(expected_proto.node)):
            expected_node = expected_proto.node[i]
            actual_node = actual_proto.node[i]
            # 断言节点的名称相同
            self.assertEqual(expected_node.name, actual_node.name)
            # 断言节点的操作类型相同
            self.assertEqual(expected_node.op, actual_node.op)
            # 断言节点的输入相同
            self.assertEqual(expected_node.input, actual_node.input)
            # 断言节点的设备信息相同
            self.assertEqual(expected_node.device, actual_node.device)
            # 断言节点的属性键列表排序后相同
            self.assertEqual(
                sorted(expected_node.attr.keys()), sorted(actual_node.attr.keys()))
    def test_nested_nn_squential(self):
        # 创建一个随机张量作为测试输入数据
        dummy_input = torch.randn(2, 3)

        # 定义内部神经网络模块 InnerNNSquential
        class InnerNNSquential(torch.nn.Module):
            def __init__(self, dim1, dim2):
                super().__init__()
                # 内部网络模块由两个线性层组成
                self.inner_nn_squential = torch.nn.Sequential(
                    torch.nn.Linear(dim1, dim2),
                    torch.nn.Linear(dim2, dim1),
                )

            def forward(self, x):
                # 将输入数据传递给内部网络模块并返回结果
                x = self.inner_nn_squential(x)
                return x

        # 定义外部神经网络模块 OuterNNSquential
        class OuterNNSquential(torch.nn.Module):
            def __init__(self, dim1=3, dim2=4, depth=2):
                super().__init__()
                layers = []
                # 根据深度参数循环创建多个 InnerNNSquential 实例
                for _ in range(depth):
                    layers.append(InnerNNSquential(dim1, dim2))
                # 外部网络模块由多个内部网络模块组成
                self.outer_nn_squential = torch.nn.Sequential(*layers)

            def forward(self, x):
                # 将输入数据传递给外部网络模块并返回结果
                x = self.outer_nn_squential(x)
                return x

        # 使用自定义方法创建 SummaryWriter 对象 w
        with self.createSummaryWriter() as w:
            # 将 OuterNNSquential 模型的图形结构添加到 SummaryWriter 中
            w.add_graph(OuterNNSquential(), dummy_input)

        # 获取预期的图形结构
        actual_proto, _ = graph(OuterNNSquential(), dummy_input)

        # 读取预期内容并解析为 GraphDef 对象
        expected_str = read_expected_content(self)
        expected_proto = GraphDef()
        text_format.Parse(expected_str, expected_proto)

        # 断言预期图结构中节点数量与实际图结构中节点数量相等
        self.assertEqual(len(expected_proto.node), len(actual_proto.node))
        for i in range(len(expected_proto.node)):
            # 逐个比较每个节点的名称、操作、输入、设备和属性
            expected_node = expected_proto.node[i]
            actual_node = actual_proto.node[i]
            self.assertEqual(expected_node.name, actual_node.name)
            self.assertEqual(expected_node.op, actual_node.op)
            self.assertEqual(expected_node.input, actual_node.input)
            self.assertEqual(expected_node.device, actual_node.device)
            self.assertEqual(
                sorted(expected_node.attr.keys()), sorted(actual_node.attr.keys()))

    def test_pytorch_graph_dict_input(self):
        # 定义一个简单的 PyTorch 模型 Model，包含一个线性层
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.l(x)

        # 定义一个返回字典输出的 PyTorch 模型 ModelDict，包含一个线性层
        class ModelDict(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(3, 5)

            def forward(self, x):
                return {"out": self.l(x)}

        # 创建一个全零张量作为测试输入数据
        dummy_input = torch.zeros(1, 3)

        # 使用自定义方法创建 SummaryWriter 对象 w，并添加 Model 模型的图形结构
        with self.createSummaryWriter() as w:
            w.add_graph(Model(), dummy_input)

        # 使用自定义方法创建 SummaryWriter 对象 w，并以严格追踪模式添加 Model 模型的图形结构
        with self.createSummaryWriter() as w:
            w.add_graph(Model(), dummy_input, use_strict_trace=True)

        # 预期出现错误：在追踪器输出中遇到字典类型的输出...
        with self.assertRaises(RuntimeError):
            with self.createSummaryWriter() as w:
                w.add_graph(ModelDict(), dummy_input, use_strict_trace=True)

        # 使用自定义方法创建 SummaryWriter 对象 w，并以非严格追踪模式添加 ModelDict 模型的图形结构
        with self.createSummaryWriter() as w:
            w.add_graph(ModelDict(), dummy_input, use_strict_trace=False)
    def test_mlp_graph(self):
        dummy_input = (torch.zeros(2, 1, 28, 28),)

        # 定义一个简单的多层感知机模型类
        class myMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_len = 1 * 28 * 28  # 输入长度为图像的像素数
                self.fc1 = torch.nn.Linear(self.input_len, 1200)  # 第一层全连接层
                self.fc2 = torch.nn.Linear(1200, 1200)  # 第二层全连接层
                self.fc3 = torch.nn.Linear(1200, 10)  # 输出层

            def forward(self, x, update_batch_stats=True):
                # 前向传播函数，依次经过激活函数和线性层
                h = torch.nn.functional.relu(
                    self.fc1(x.view(-1, self.input_len)))
                h = self.fc2(h)
                h = torch.nn.functional.relu(h)
                h = self.fc3(h)
                return h

        # 使用自定义的模型类和虚拟输入创建 SummaryWriter 对象，并添加图结构
        with self.createSummaryWriter() as w:
            w.add_graph(myMLP(), dummy_input)

    def test_wrong_input_size(self):
        # 测试输入尺寸不匹配的情况
        with self.assertRaises(RuntimeError) as e_info:
            dummy_input = torch.rand(1, 9)  # 错误的输入尺寸
            model = torch.nn.Linear(3, 5)  # 创建线性模型
            with self.createSummaryWriter() as w:
                w.add_graph(model, dummy_input)  # 尝试添加图结构，会出错

    @skipIfNoTorchVision
    def test_torchvision_smoke(self):
        # 对 TorchVision 中的几个预训练模型进行简单的功能验证
        model_input_shapes = {
            'alexnet': (2, 3, 224, 224),
            'resnet34': (2, 3, 224, 224),
            'resnet152': (2, 3, 224, 224),
            'densenet121': (2, 3, 224, 224),
            'vgg16': (2, 3, 224, 224),
            'vgg19': (2, 3, 224, 224),
            'vgg16_bn': (2, 3, 224, 224),
            'vgg19_bn': (2, 3, 224, 224),
            'mobilenet_v2': (2, 3, 224, 224),
        }
        # 遍历模型及其对应的输入形状
        for model_name, input_shape in model_input_shapes.items():
            with self.createSummaryWriter() as w:
                # 根据模型名动态获取 TorchVision 中的模型，并创建实例
                model = getattr(torchvision.models, model_name)()
                # 使用 SummaryWriter 添加模型的图结构
                w.add_graph(model, torch.zeros(input_shape))
class TestTensorBoardFigure(BaseTestCase):
    # 测试类：TensorBoard 图形测试，继承自 BaseTestCase

    @skipIfNoMatplotlib
    # 如果没有 Matplotlib，跳过该测试
    def test_figure(self):
        # 测试方法：测试单个图形添加到 TensorBoard

        writer = self.createSummaryWriter()
        # 创建 TensorBoard SummaryWriter 对象

        figure, axes = plt.figure(), plt.gca()
        # 创建 Matplotlib 图形和当前轴对象

        circle1 = plt.Circle((0.2, 0.5), 0.2, color='r')
        circle2 = plt.Circle((0.8, 0.5), 0.2, color='g')
        # 创建两个圆形对象，分别指定位置和颜色

        axes.add_patch(circle1)
        axes.add_patch(circle2)
        # 将圆形对象添加到当前轴上

        plt.axis('scaled')
        plt.tight_layout()
        # 设置图形的轴比例和紧凑布局

        writer.add_figure("add_figure/figure", figure, 0, close=False)
        # 将图形添加到 TensorBoard，标签为 "add_figure/figure"，索引为 0，不关闭图形

        self.assertTrue(plt.fignum_exists(figure.number))
        # 断言图形是否存在于 Matplotlib 的图形编号列表中

        writer.add_figure("add_figure/figure", figure, 1)
        # 再次将图形添加到 TensorBoard，索引为 1

        if matplotlib.__version__ != '3.3.0':
            self.assertFalse(plt.fignum_exists(figure.number))
            # 如果 Matplotlib 版本不是 3.3.0，断言图形不再存在于 Matplotlib 的图形编号列表中
        else:
            print("Skipping fignum_exists, see https://github.com/matplotlib/matplotlib/issues/18163")
            # 否则，输出跳过消息，参考链接详细信息

        writer.close()
        # 关闭 TensorBoard SummaryWriter 对象

    @skipIfNoMatplotlib
    # 如果没有 Matplotlib，跳过该测试
    def test_figure_list(self):
        # 测试方法：测试多个图形添加到 TensorBoard

        writer = self.createSummaryWriter()
        # 创建 TensorBoard SummaryWriter 对象

        figures = []
        for i in range(5):
            figure = plt.figure()
            # 循环创建五个 Matplotlib 图形对象

            plt.plot([i * 1, i * 2, i * 3], label="Plot " + str(i))
            plt.xlabel("X")
            plt.xlabel("Y")
            plt.legend()
            plt.tight_layout()
            # 在每个图形上绘制不同的线条，添加标签和轴标签，添加图例，调整布局

            figures.append(figure)
            # 将图形对象添加到列表中

        writer.add_figure("add_figure/figure_list", figures, 0, close=False)
        # 将图形列表添加到 TensorBoard，标签为 "add_figure/figure_list"，索引为 0，不关闭图形

        self.assertTrue(all(plt.fignum_exists(figure.number) is True for figure in figures))
        # 断言所有图形存在于 Matplotlib 的图形编号列表中

        writer.add_figure("add_figure/figure_list", figures, 1)
        # 再次将图形列表添加到 TensorBoard，索引为 1

        if matplotlib.__version__ != '3.3.0':
            self.assertTrue(all(plt.fignum_exists(figure.number) is False for figure in figures))
            # 如果 Matplotlib 版本不是 3.3.0，断言所有图形不再存在于 Matplotlib 的图形编号列表中
        else:
            print("Skipping fignum_exists, see https://github.com/matplotlib/matplotlib/issues/18163")
            # 否则，输出跳过消息，参考链接详细信息

        writer.close()
        # 关闭 TensorBoard SummaryWriter 对象
    @parametrize(
        "tensor_type,proto_type",
        [  # 参数化测试用例，定义了两组参数：tensor_type 和 proto_type
            (torch.float16, DataType.DT_HALF),   # 第一组参数：torch.float16 对应 DataType.DT_HALF
            (torch.bfloat16, DataType.DT_BFLOAT16),  # 第二组参数：torch.bfloat16 对应 DataType.DT_BFLOAT16
        ],
    )
    def test_half_tensor_proto(self, tensor_type, proto_type):
        float_values = [1.0, 2.0, 3.0]
        actual_proto = tensor_proto(  # 调用 tensor_proto 函数创建 Protobuf 对象 actual_proto
            "dummy",
            torch.tensor(float_values, dtype=tensor_type),  # 使用给定的 tensor_type 创建 Torch 张量
        ).value[0].tensor
        self.assertSequenceEqual(
            [int_to_half(x) for x in actual_proto.half_val],  # 断言：将 actual_proto.half_val 中的值转换为半精度浮点数
            float_values,  # 与 float_values 进行比较
        )
        self.assertTrue(actual_proto.dtype == proto_type)  # 断言：actual_proto 的数据类型与 proto_type 相符

    def test_float_tensor_proto(self):
        float_values = [1.0, 2.0, 3.0]
        actual_proto = (
            tensor_proto("dummy", torch.tensor(float_values)).value[0].tensor  # 创建 Protobuf 对象 actual_proto
        )
        self.assertEqual(actual_proto.float_val, float_values)  # 断言：actual_proto.float_val 与 float_values 相等
        self.assertTrue(actual_proto.dtype == DataType.DT_FLOAT)  # 断言：actual_proto 的数据类型为 DataType.DT_FLOAT

    def test_int_tensor_proto(self):
        int_values = [1, 2, 3]
        actual_proto = (
            tensor_proto("dummy", torch.tensor(int_values, dtype=torch.int32))  # 创建 Protobuf 对象 actual_proto
            .value[0]
            .tensor
        )
        self.assertEqual(actual_proto.int_val, int_values)  # 断言：actual_proto.int_val 与 int_values 相等
        self.assertTrue(actual_proto.dtype == DataType.DT_INT32)  # 断言：actual_proto 的数据类型为 DataType.DT_INT32

    def test_scalar_tensor_proto(self):
        scalar_value = 0.1
        actual_proto = (
            tensor_proto("dummy", torch.tensor(scalar_value)).value[0].tensor  # 创建 Protobuf 对象 actual_proto
        )
        self.assertAlmostEqual(actual_proto.float_val[0], scalar_value)  # 断言：actual_proto.float_val 的第一个值约等于 scalar_value

    def test_complex_tensor_proto(self):
        real = torch.tensor([1.0, 2.0])
        imag = torch.tensor([3.0, 4.0])
        actual_proto = (
            tensor_proto("dummy", torch.complex(real, imag)).value[0].tensor  # 创建 Protobuf 对象 actual_proto
        )
        self.assertEqual(actual_proto.scomplex_val, [1.0, 3.0, 2.0, 4.0])  # 断言：actual_proto.scomplex_val 与 [1.0, 3.0, 2.0, 4.0] 相等

    def test_empty_tensor_proto(self):
        actual_proto = tensor_proto("dummy", torch.empty(0)).value[0].tensor  # 创建 Protobuf 对象 actual_proto
        self.assertEqual(actual_proto.float_val, [])  # 断言：actual_proto.float_val 为空列表
#`
# 实例化参数化测试，使用 TestTensorProtoSummary 类进行参数化测试
instantiate_parametrized_tests(TestTensorProtoSummary)

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == '__main__':
    # 运行测试套件中的所有测试用ts()
```