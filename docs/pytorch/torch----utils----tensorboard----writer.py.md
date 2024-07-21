# `.\pytorch\torch\utils\tensorboard\writer.py`

```
"""
Provide an API for writing protocol buffers to event files to be consumed by TensorBoard for visualization.
"""

import os                               # 导入操作系统相关功能
import time                             # 导入时间相关功能
from typing import List, Optional, TYPE_CHECKING, Union  # 引入类型提示相关功能

import torch                            # 导入PyTorch库

if TYPE_CHECKING:
    from matplotlib.figure import Figure  # 条件引入类型检查，若是类型检查环境，引入matplotlib的Figure类型
from tensorboard.compat import tf       # 导入兼容层中的TensorFlow
from tensorboard.compat.proto import event_pb2  # 导入兼容层中的protocol buffer定义
from tensorboard.compat.proto.event_pb2 import Event, SessionLog  # 导入事件和会话日志定义
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig  # 导入投影仪配置定义
from tensorboard.summary.writer.event_file_writer import EventFileWriter  # 导入事件文件写入类

from ._convert_np import make_np         # 从内部模块导入函数make_np
from ._embedding import get_embedding_info, make_mat, make_sprite, make_tsv, write_pbtxt  # 从内部模块导入函数和工具
from ._onnx_graph import load_onnx_graph  # 从内部模块导入加载ONNX图函数
from ._pytorch_graph import graph        # 从内部模块导入PyTorch图函数
from ._utils import figure_to_image      # 从内部模块导入将图形转换为图像的工具函数
from .summary import (                   # 从summary模块导入多个函数
    audio, custom_scalars, histogram, histogram_raw, hparams, image,
    image_boxes, mesh, pr_curve, pr_curve_raw, scalar, tensor_proto, text,
    video,
)

__all__ = ["FileWriter", "SummaryWriter"]  # 模块的公开接口包括FileWriter和SummaryWriter类


class FileWriter:
    """
    Writes protocol buffers to event files to be consumed by TensorBoard.

    The `FileWriter` class provides a mechanism to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(self, log_dir, max_queue=10, flush_secs=120, filename_suffix=""):
        """
        Create a `FileWriter` and an event file.

        On construction the writer creates a new event file in `log_dir`.
        The other arguments to the constructor control the asynchronous writes to
        the event file.

        Args:
          log_dir: A string. Directory where event file will be written.
          max_queue: Integer. Size of the queue for pending events and
            summaries before one of the 'add' calls forces a flush to disk.
            Default is ten items.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk. Default is every two minutes.
          filename_suffix: A string. Suffix added to all event filenames
            in the log_dir directory. More details on filename construction in
            tensorboard.summary.writer.event_file_writer.EventFileWriter.
        """
        log_dir = str(log_dir)  # 强制将log_dir转换为字符串，以确保一致性
        self.event_writer = EventFileWriter(  # 创建一个EventFileWriter实例
            log_dir, max_queue, flush_secs, filename_suffix
        )
    def get_logdir(self):
        """
        Return the directory where event file will be written.
        """
        return self.event_writer.get_logdir()

    def add_event(self, event, step=None, walltime=None):
        """
        Add an event to the event file.

        Args:
          event: An `Event` protocol buffer.
          step: Number. Optional global step value for training process
            to record with the event.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        # Set wall time to current time if not provided
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            # Convert step to integer if not already
            event.step = int(step)
        # Add the event to the event writer
        self.event_writer.add_event(event)

    def add_summary(self, summary, global_step=None, walltime=None):
        """
        Add a `Summary` protocol buffer to the event file.

        Args:
          summary: A `Summary` protocol buffer.
          global_step: Number. Optional global step value for training process
            to record with the summary.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        # Wrap the summary in an `Event` protocol buffer and add to event file
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step, walltime)

    def add_graph(self, graph_profile, walltime=None):
        """
        Add a `Graph` and step stats protocol buffer to the event file.

        Args:
          graph_profile: A `Graph` and step stats protocol buffer.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        # Extract graph and step stats from graph_profile
        graph = graph_profile[0]
        stepstats = graph_profile[1]
        # Add graph definition to event file
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)
        # Add step stats to event file
        trm = event_pb2.TaggedRunMetadata(
            tag="step1", run_metadata=stepstats.SerializeToString()
        )
        event = event_pb2.Event(tagged_run_metadata=trm)
        self.add_event(event, None, walltime)

    def add_onnx_graph(self, graph, walltime=None):
        """
        Add a `Graph` protocol buffer to the event file.

        Args:
          graph: A `Graph` protocol buffer.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time())
        """
        # Add ONNX graph definition to event file
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)

    def flush(self):
        """
        Flushes the event file to disk.

        Call this method to make sure that all pending events have been written to
        disk.
        """
        # Flush all pending events to disk
        self.event_writer.flush()
    def close(self):
        """
        将事件文件刷新到磁盘并关闭文件。

        当不再需要摘要写入器时，请调用此方法。
        """
        self.event_writer.close()

    def reopen(self):
        """
        重新打开 EventFileWriter。

        可以在 `close()` 后调用，以在相同目录中添加更多事件。
        新的事件将被写入新的事件文件中。
        如果 EventFileWriter 没有关闭，则不执行任何操作。
        """
        self.event_writer.reopen()
class SummaryWriter:
    """Writes entries directly to event files in the log_dir to be consumed by TensorBoard.
    
    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(
        self,
        log_dir=None,
        comment="",
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix="",
    ):
        """Initialize the SummaryWriter instance.

        Args:
            log_dir (str): Directory where event files will be written.
            comment (str): Comment to append to the log directory name.
            purge_step (int): Step at which to purge the event files.
            max_queue (int): Maximum queue size for event files.
            flush_secs (int): Flush interval in seconds for event files.
            filename_suffix (str): Suffix to append to event file names.
        """
        # Initialize instance variables with provided or default values
        self.log_dir = log_dir
        self.comment = comment
        self.purge_step = purge_step
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.filename_suffix = filename_suffix
        
        # Initialize writer variables
        self.file_writer = None  # FileWriter instance
        self.all_writers = None  # Dictionary to hold all active FileWriter instances

    def _get_file_writer(self):
        """Return the default FileWriter instance. Recreates it if closed."""
        if self.all_writers is None or self.file_writer is None:
            # Create a new FileWriter if not already initialized or closed
            self.file_writer = FileWriter(
                self.log_dir, self.max_queue, self.flush_secs, self.filename_suffix
            )
            # Add the new FileWriter instance to all_writers dictionary
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
            
            # If purge_step is provided, add corresponding events and reset purge_step
            if self.purge_step is not None:
                most_recent_step = self.purge_step
                self.file_writer.add_event(
                    Event(step=most_recent_step, file_version="brain.Event:2")
                )
                self.file_writer.add_event(
                    Event(
                        step=most_recent_step,
                        session_log=SessionLog(status=SessionLog.START),
                    )
                )
                self.purge_step = None
        
        return self.file_writer

    def get_logdir(self):
        """Return the directory where event files will be written."""
        return self.log_dir

    def add_hparams(
        self,
        hparam_dict,
        metric_dict,
        hparam_domain_discrete=None,
        run_name=None,
        global_step=None,
    ):
        """Add a set of hyperparameters to be logged.

        Args:
            hparam_dict (dict): Dictionary containing the hyperparameter names and values.
            metric_dict (dict): Dictionary containing the metric names and values.
            hparam_domain_discrete (dict, optional): Dictionary describing the discrete domains of the hyperparameters.
            run_name (str, optional): Name of the run.
            global_step (int, optional): Global step value to record with the hparams.
        """
        # Function to add hyperparameters and metrics to the event file
        pass
    ):
        """
        将一组超参数添加到TensorBoard中以进行比较。

        Args:
            hparam_dict (dict): 字典中的每个键值对分别是超参数的名称和对应的值。
              值的类型可以是`bool`、`string`、`float`、`int`或`None`。
            metric_dict (dict): 字典中的每个键值对分别是指标的名称和对应的值。请注意，
              这里使用的键在TensorBoard记录中应该是唯一的。否则，您通过`add_scalar`添加的值将会显示在hparam插件中，这通常是不希望的。
            hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) 包含超参数名称及其所有离散值的字典
            run_name (str): 运行的名称，将作为日志目录的一部分。如果未指定，将使用当前时间戳。
            global_step (int): 要记录的全局步数值。

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

        期望的结果：

        .. image:: _static/img/tensorboard/add_hparam.png
           :scale: 50 %

        """
        # 记录API使用情况到Torch的日志中
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        
        # 检查输入的参数类型是否为字典，如果不是则抛出TypeError异常
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        
        # 调用hparams函数生成实验、summary statistics input和summary statistics entropy
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        # 如果未指定运行名称，则使用当前时间戳作为运行名称
        if not run_name:
            run_name = str(time.time())
        
        # 构建日志目录路径，将其作为文件写入器的日志目录的子目录
        logdir = os.path.join(self._get_file_writer().get_logdir(), run_name)
        
        # 使用SummaryWriter，打开日志目录，使用w_hp作为写入器对象
        with SummaryWriter(log_dir=logdir) as w_hp:
            # 向文件写入器添加实验数据、summary statistics input和summary statistics entropy
            w_hp.file_writer.add_summary(exp, global_step)
            w_hp.file_writer.add_summary(ssi, global_step)
            w_hp.file_writer.add_summary(sei, global_step)
            
            # 遍历metric_dict中的每个键值对，将其作为标量添加到TensorBoard中
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v, global_step)
    ):
        """
        Add scalar data to summary.

        Args:
            tag (str): Data identifier for the scalar value
            scalar_value (float or string/blobname): Value to save (can be numeric or a string/blob name)
            global_step (int): Global step value to record the data point
            walltime (float): Optional override of default walltime with seconds after epoch of event
            new_style (boolean): Whether to use new style (tensor field) or old style (simple_value field) for data representation in tensorboard. New style can lead to faster data loading.

        Examples:
            This method shows how to use `torch.utils.tensorboard.SummaryWriter` to add scalar data to TensorBoard.
            ```
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = range(100)
            for i in x:
                writer.add_scalar('y=2x', i * 2, i)
            writer.close()
            ```

        Expected result:

        This function is expected to plot scalar data on TensorBoard, demonstrating a line graph where x-axis is the global step and y-axis is the scalar value.

        """
        # Log usage of tensorboard.logging.add_scalar API once
        torch._C._log_api_usage_once("tensorboard.logging.add_scalar")

        # Generate summary data for the scalar value with specified tag, using new or old style based on new_style parameter
        summary = scalar(
            tag, scalar_value, new_style=new_style, double_precision=double_precision
        )
        # Add the generated summary data to the file writer associated with this summary writer instance
        self._get_file_writer().add_summary(summary, global_step, walltime)
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        """Add many scalar data to summary.

        Args:
            main_tag (str): The parent name for the tags
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            r = 5
            for i in range(100):
                writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                                'xcosx':i*np.cos(i/r),
                                                'tanx': np.tan(i/r)}, i)
            writer.close()
            # This call adds three values to the same scalar plot with the tag
            # 'run_14h' in TensorBoard's scalar section.

        Expected result:

        .. image:: _static/img/tensorboard/add_scalars.png
           :scale: 50 %

        """
        # 记录 API 使用情况到 PyTorch 的内部统计
        torch._C._log_api_usage_once("tensorboard.logging.add_scalars")
        # 如果未提供 walltime，则使用当前时间作为默认值
        walltime = time.time() if walltime is None else walltime
        # 获取文件写入器的日志目录
        fw_logdir = self._get_file_writer().get_logdir()
        # 遍历标签-标量值字典
        for tag, scalar_value in tag_scalar_dict.items():
            # 构建完整的标签路径，用于唯一标识该标量数据的位置
            fw_tag = fw_logdir + "/" + main_tag.replace("/", "_") + "_" + tag
            # 确保所有写入器都已初始化
            assert self.all_writers is not None
            # 检查写入器是否已存在，如果不存在则创建新的写入器
            if fw_tag in self.all_writers.keys():
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(
                    fw_tag, self.max_queue, self.flush_secs, self.filename_suffix
                )
                self.all_writers[fw_tag] = fw
            # 将标量数据添加到 TensorBoard 的事件文件中
            fw.add_summary(scalar(main_tag, scalar_value), global_step, walltime)

    def add_tensor(
        self,
        tag,
        tensor,
        global_step=None,
        walltime=None,
    ):
        """Add tensor data to summary.

        Args:
            tag (str): Data identifier
            tensor (torch.Tensor): tensor to save
            global_step (int): Global step value to record
        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = torch.tensor([1,2,3])
            writer.add_scalar('x', x)
            writer.close()

        Expected result:
            Summary::tensor::float_val [1,2,3]
                   ::tensor::shape [3]
                   ::tag 'x'

        """
        # 记录 API 使用情况到 PyTorch 的内部统计
        torch._C._log_api_usage_once("tensorboard.logging.add_tensor")

        # 创建 tensor 的协议缓冲区消息
        summary = tensor_proto(tag, tensor)
        # 将 tensor 数据添加到 TensorBoard 的事件文件中
        self._get_file_writer().add_summary(summary, global_step, walltime)

    def add_histogram(
        self,
        tag,
        values,
        global_step=None,
        bins="tensorflow",
        walltime=None,
        max_bins=None,
    ):
        """Add histogram to summary.

        Args:
            tag (str): Data identifier
            values (torch.Tensor, numpy.array, or list): Values for histogram
            global_step (int): Global step value to record
            bins (str): One of {'tensorflow', 'auto', 'fd', ...}, default is 'tensorflow'
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            max_bins (int): Maximum number of bins

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            for i in range(10):
                writer.add_histogram('hist', np.random.rand(1000), i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_histogram.png
           :scale: 50 %

        """
        # 记录 API 使用情况到 PyTorch 的内部统计
        torch._C._log_api_usage_once("tensorboard.logging.add_histogram")
        
        # 创建直方图的协议缓冲区消息
        hist = histogram_proto(tag, values, bins, max_bins)
        # 将直方图数据添加到 TensorBoard 的事件文件中
        self._get_file_writer().add_summary(hist, global_step, walltime)
    ):
        """
        将直方图添加到摘要中。

        Args:
            tag (str): 数据标识符
                要构建直方图的值，可以是 torch.Tensor、numpy.ndarray 或字符串/对象名称
            values (torch.Tensor, numpy.ndarray, or string/blobname): 要构建直方图的值
            global_step (int): 记录的全局步骤值
            bins (str): {'tensorflow', 'auto', 'fd', ...} 中的一个字符串，用于确定直方图的箱子如何生成。
                可以在此处找到其他选项: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            walltime (float): 可选，用于覆盖默认的 walltime（time.time()）的秒数，事件的时间戳

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            writer = SummaryWriter()
            for i in range(10):
                x = np.random.random(1000)
                writer.add_histogram('distribution centers', x + i, i)
            writer.close()

        期望的结果:

        .. image:: _static/img/tensorboard/add_histogram.png
           :scale: 50 %

        """
        # 记录使用 tensorboard.logging.add_histogram API 的调用一次
        torch._C._log_api_usage_once("tensorboard.logging.add_histogram")
        if isinstance(bins, str) and bins == "tensorflow":
            bins = self.default_bins
        # 将直方图摘要添加到文件编写器中
        self._get_file_writer().add_summary(
            histogram(tag, values, bins, max_bins=max_bins), global_step, walltime
        )

    def add_histogram_raw(
        self,
        tag,
        min,
        max,
        num,
        sum,
        sum_squares,
        bucket_limits,
        bucket_counts,
        global_step=None,
        walltime=None,
    ):
        """
        Add histogram with raw data.

        Args:
            tag (str): Data identifier
                数据标识符
            min (float or int): Min value
                最小值，可以是浮点数或整数
            max (float or int): Max value
                最大值，可以是浮点数或整数
            num (int): Number of values
                值的数量
            sum (float or int): Sum of all values
                所有值的总和
            sum_squares (float or int): Sum of squares for all values
                所有值的平方和
            bucket_limits (torch.Tensor, numpy.ndarray): Upper value per bucket.
              The number of elements of it should be the same as `bucket_counts`.
                每个桶的上限值。它的元素数量应与 `bucket_counts` 相同。
            bucket_counts (torch.Tensor, numpy.ndarray): Number of values per bucket
                每个桶内的值的数量
            global_step (int): Global step value to record
                要记录的全局步骤值
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
                可选参数，覆盖默认的 walltime（time.time()）事件发生后的秒数
            see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/histogram/README.md
                参见：https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/histogram/README.md

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            writer = SummaryWriter()
            dummy_data = []
            for idx, value in enumerate(range(50)):
                dummy_data += [idx + 0.001] * value

            bins = list(range(50+2))
            bins = np.array(bins)
            values = np.array(dummy_data).astype(float).reshape(-1)
            counts, limits = np.histogram(values, bins=bins)
            sum_sq = values.dot(values)
            writer.add_histogram_raw(
                tag='histogram_with_raw_data',
                min=values.min(),
                max=values.max(),
                num=len(values),
                sum=values.sum(),
                sum_squares=sum_sq,
                bucket_limits=limits[1:].tolist(),
                bucket_counts=counts.tolist(),
                global_step=0)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_histogram_raw.png
           :scale: 50 %

        """
        # 记录 API 使用情况，仅记录一次
        torch._C._log_api_usage_once("tensorboard.logging.add_histogram_raw")
        # 检查桶限制值和桶计数值的长度是否相等，如果不相等则抛出 ValueError
        if len(bucket_limits) != len(bucket_counts):
            raise ValueError(
                "len(bucket_limits) != len(bucket_counts), see the document."
            )
        # 将直方图原始数据添加到文件写入器的摘要中
        self._get_file_writer().add_summary(
            histogram_raw(
                tag, min, max, num, sum, sum_squares, bucket_limits, bucket_counts
            ),
            global_step,
            walltime,
        )
    ):
        """
        Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (str): Data identifier
                标签名，用于标识图像数据
            img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
                图像数据，可以是 torch.Tensor, numpy.ndarray 或者字符串/文件名
            global_step (int): Global step value to record
                全局步骤值，用于记录当前步骤
            walltime (float): Optional override default walltime (time.time())
                Optional，可选项，用于覆盖默认的 walltime（time.time()）的秒数值，表示事件发生的时间
            dataformats (str): Image data format specification of the form
                CHW, HWC, HW, WH, etc.
                图像数据的格式规范，例如 CHW, HWC, HW, WH 等

        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()``
            to convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
            图像张量的形状说明，默认为 :math:`(3, H, W)`。您可以使用 ``torchvision.utils.make_grid()``
            将一个张量批量转换为 3xHxW 格式，或者调用 ``add_images`` 让我们来处理。
            张量的形状可以是 :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)`，只要相应的 ``dataformats``
            参数被传递，如 ``CHW``, ``HWC``, ``HW``。

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            img = np.zeros((3, 100, 100))
            img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
            img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

            img_HWC = np.zeros((100, 100, 3))
            img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
            img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

            writer = SummaryWriter()
            writer.add_image('my_image', img, 0)

            # If you have non-default dimension setting, set the dataformats argument.
            writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_image.png
           :scale: 50 %

        """
        # 记录使用 TensorBoard 的 add_image 方法
        torch._C._log_api_usage_once("tensorboard.logging.add_image")
        # 获取文件写入器，并添加摘要
        self._get_file_writer().add_summary(
            # 调用 image 函数，生成图像摘要，并传递给文件写入器
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime
        )

    def add_images(
        self, tag, img_tensor, global_step=None, walltime=None, dataformats="NCHW"
    ):
        """
        将批量图像数据添加到摘要中。

        注意，这需要使用 ``pillow`` 包。

        Args:
            tag (str): 数据标识符
            img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): 图像数据
            global_step (int): 要记录的全局步骤值
            walltime (float): 可选，覆盖默认的 walltime（time.time()）事件发生的秒数
              epoch 后的时间
            dataformats (str): 图像数据格式的规范，如 NCHW、NHWC、CHW、HWC、HW、WH 等。

        Shape:
            img_tensor: 默认为 :math:`(N, 3, H, W)`。如果指定了 ``dataformats``，将接受其他形状。
            例如 NCHW 或 NHWC。

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np

            img_batch = np.zeros((16, 3, 100, 100))
            for i in range(16):
                img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
                img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

            writer = SummaryWriter()
            writer.add_images('my_image_batch', img_batch, 0)
            writer.close()

        期望结果:

        .. image:: _static/img/tensorboard/add_images.png
           :scale: 30 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_images")
        self._get_file_writer().add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime
        )
    ):
        """
        将图像添加到 TensorBoard，并在图像上绘制边界框。

        Args:
            tag (str): 数据标识符
            img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): 图像数据
            box_tensor (torch.Tensor, numpy.ndarray, or string/blobname): 边界框数据（检测到的对象）
              边界框应表示为 [x1, y1, x2, y2]。
            global_step (int): 要记录的全局步骤值
            walltime (float): 可选，覆盖默认的 walltime（time.time()），
              表示自事件发生后经过的秒数
            rescale (float): 可选，覆盖默认的缩放比例
            dataformats (str): 图像数据格式规范，如 NCHW、NHWC、CHW、HWC、HW、WH 等
            labels (list of string): 每个边界框要显示的标签

        Shape:
            img_tensor: 默认为 :math:`(3, H, W)`。可以使用 ``dataformats`` 参数指定格式，
              例如 CHW 或 HWC。

            box_tensor: (torch.Tensor, numpy.ndarray, or string/blobname): NX4，其中 N 是边界框的数量，
              每行的 4 个元素表示 (xmin, ymin, xmax, ymax)。

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_image_with_boxes")
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            if len(labels) != box_tensor.shape[0]:
                labels = None
        self._get_file_writer().add_summary(
            image_boxes(
                tag,
                img_tensor,
                box_tensor,
                rescale=rescale,
                dataformats=dataformats,
                labels=labels,
            ),
            global_step,
            walltime,
        )

    def add_figure(
        self,
        tag: str,
        figure: Union["Figure", List["Figure"]],
        global_step: Optional[int] = None,
        close: bool = True,
        walltime: Optional[float] = None,
    ) -> None:
        """
        将 matplotlib 图形渲染为图像，并将其添加到摘要中。

        注意，这需要 ``matplotlib`` 包。

        Args:
            tag: 数据标识符
            figure: 单个图形或图形列表
            global_step: 要记录的全局步骤值
            close: 是否自动关闭图形
            walltime: 可选，覆盖默认的 walltime（time.time()），
              表示自事件发生后经过的秒数
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_figure")
        if isinstance(figure, list):
            self.add_image(
                tag,
                figure_to_image(figure, close),
                global_step,
                walltime,
                dataformats="NCHW",
            )
        else:
            self.add_image(
                tag,
                figure_to_image(figure, close),
                global_step,
                walltime,
                dataformats="CHW",
            )
    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        """Add video data to summary.

        Note that this requires the ``moviepy`` package.

        Args:
            tag (str): Data identifier
            vid_tensor (torch.Tensor): Video data
            global_step (int): Global step value to record
            fps (float or int): Frames per second
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Shape:
            vid_tensor: :math:`(N, T, C, H, W)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.
        """
        # 记录 API 使用情况
        torch._C._log_api_usage_once("tensorboard.logging.add_video")
        # 获取文件写入器并添加视频摘要
        self._get_file_writer().add_summary(
            video(tag, vid_tensor, fps), global_step, walltime
        )

    def add_audio(
        self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None
    ):
        """Add audio data to summary.

        Args:
            tag (str): Data identifier
            snd_tensor (torch.Tensor): Sound data
            global_step (int): Global step value to record
            sample_rate (int): sample rate in Hz
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Shape:
            snd_tensor: :math:`(1, L)`. The values should lie between [-1, 1].
        """
        # 记录 API 使用情况
        torch._C._log_api_usage_once("tensorboard.logging.add_audio")
        # 获取文件写入器并添加音频摘要
        self._get_file_writer().add_summary(
            audio(tag, snd_tensor, sample_rate=sample_rate), global_step, walltime
        )

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """Add text data to summary.

        Args:
            tag (str): Data identifier
            text_string (str): String to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Examples::

            writer.add_text('lstm', 'This is an lstm', 0)
            writer.add_text('rnn', 'This is an rnn', 10)
        """
        # 记录 API 使用情况
        torch._C._log_api_usage_once("tensorboard.logging.add_text")
        # 获取文件写入器并添加文本摘要
        self._get_file_writer().add_summary(
            text(tag, text_string), global_step, walltime
        )

    def add_onnx_graph(self, prototxt):
        """Add ONNX graph to summary.

        Args:
            prototxt (str): Path to ONNX graph prototxt file
        """
        # 记录 API 使用情况
        torch._C._log_api_usage_once("tensorboard.logging.add_onnx_graph")
        # 获取文件写入器并添加 ONNX 图摘要
        self._get_file_writer().add_onnx_graph(load_onnx_graph(prototxt))

    def add_graph(
        self, model, input_to_model=None, verbose=False, use_strict_trace=True
    ):
        """Add graph to summary.

        Args:
            model: The model to be shown as a graph
            input_to_model: A tensor that will be used as an input to the model
            verbose (bool, optional): Whether to print graph structure in console
            use_strict_trace (bool, optional): Ensure the graph is strictly traceable

        Note:
            This method is a placeholder. It's an abstract method, so it needs to be overridden by subclasses.
        """
        # 记录 API 使用情况
        torch._C._log_api_usage_once("tensorboard.logging.add_graph")
        # 获取文件写入器并添加图摘要
        self._get_file_writer().add_graph(
            model, input_to_model=input_to_model, verbose=verbose, use_strict_trace=use_strict_trace
        )
    ):
        """
        Add graph data to summary.

        Args:
            model (torch.nn.Module): Model to draw.
            input_to_model (torch.Tensor or list of torch.Tensor): A variable or a tuple of
                variables to be fed.
            verbose (bool): Whether to print graph structure in console.
            use_strict_trace (bool): Whether to pass keyword argument `strict` to
                `torch.jit.trace`. Pass False when you want the tracer to
                record your mutable container types (list, dict)
        """
        # 记录 API 使用情况，用于统计分析
        torch._C._log_api_usage_once("tensorboard.logging.add_graph")
        # 验证传入的模型是否为有效的 PyTorch 模型，应该包含 'forward' 方法
        # 将模型的计算图添加到 TensorBoard 中
        self._get_file_writer().add_graph(
            graph(model, input_to_model, verbose, use_strict_trace)
        )

    @staticmethod
    def _encode(rawstr):
        # 使用自定义方式对字符串进行 URL 编码，替代了 urllib，以处理 Python 3 和 Python 2 之间的差异
        retval = rawstr
        retval = retval.replace("%", f"%{ord('%'):02x}")
        retval = retval.replace("/", f"%{ord('/'):02x}")
        retval = retval.replace("\\", "%%%02x" % (ord("\\")))  # noqa: UP031
        return retval

    def add_embedding(
        self,
        mat,
        metadata=None,
        label_img=None,
        global_step=None,
        tag="default",
        metadata_header=None,
    ):
        # 将嵌入向量数据添加到 TensorBoard 中
        # mat: 嵌入矩阵
        # metadata: 元数据，可以是标签等
        # label_img: 可视化的图像标签
        # global_step: 全局步数，用于标识时间或迭代次数
        # tag: 嵌入数据的标签，默认为 "default"
        # metadata_header: 元数据的标题或描述信息
        ...

    def add_pr_curve(
        self,
        tag,
        labels,
        predictions,
        global_step=None,
        num_thresholds=127,
        weights=None,
        walltime=None,
    ):
        # 添加 Precision-Recall 曲线到 TensorBoard 中
        # tag: 数据的标签
        # labels: 真实标签
        # predictions: 模型预测值
        # global_step: 全局步数，用于标识时间或迭代次数
        # num_thresholds: 用于计算 PR 曲线的阈值数量
        # weights: 每个样本的权重
        # walltime: 记录事件发生的时间戳
        ...
    ):
        """
        添加 Precision-Recall 曲线。

        绘制 Precision-Recall 曲线可以帮助你理解模型在不同阈值设置下的性能表现。使用此函数，
        你需要提供每个目标的真实标签（T/F）和预测置信度（通常是模型的输出）。TensorBoard
        UI 可以让你交互地选择阈值。

        Args:
            tag (str): 数据标识符
              labels (torch.Tensor, numpy.ndarray, or string/blobname):
                真实标签数据，每个元素的二元标签。
              predictions (torch.Tensor, numpy.ndarray, or string/blobname):
                每个元素被分类为真的概率。值应在 [0, 1] 之间。
            global_step (int): 要记录的全局步骤值
            num_thresholds (int): 用于绘制曲线的阈值数量。
            walltime (float): 可选参数，覆盖默认的 walltime（time.time()）。
              事件发生后的秒数，从 epoch 开始计算。

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            labels = np.random.randint(2, size=100)  # 生成二元标签
            predictions = np.random.rand(100)
            writer = SummaryWriter()
            writer.add_pr_curve('pr_curve', labels, predictions, 0)
            writer.close()

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_pr_curve")
        labels, predictions = make_np(labels), make_np(predictions)
        self._get_file_writer().add_summary(
            pr_curve(tag, labels, predictions, num_thresholds, weights),
            global_step,
            walltime,
        )
    ):
        """
        添加原始数据的精确率-召回率曲线。

        Args:
            tag (str): 数据标识符
            true_positive_counts (torch.Tensor, numpy.ndarray, or string/blobname): 真正例数量
            false_positive_counts (torch.Tensor, numpy.ndarray, or string/blobname): 假正例数量
            true_negative_counts (torch.Tensor, numpy.ndarray, or string/blobname): 真反例数量
            false_negative_counts (torch.Tensor, numpy.ndarray, or string/blobname): 假反例数量
            precision (torch.Tensor, numpy.ndarray, or string/blobname): 精确率
            recall (torch.Tensor, numpy.ndarray, or string/blobname): 召回率
            global_step (int): 记录的全局步数值
            num_thresholds (int): 用于绘制曲线的阈值数量
            walltime (float): 可选参数，覆盖默认的 walltime（time.time()），
                              表示事件的时间戳（秒数，从 epoch 开始）
            see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_pr_curve_raw")
        self._get_file_writer().add_summary(
            pr_curve_raw(
                tag,
                true_positive_counts,
                false_positive_counts,
                true_negative_counts,
                false_negative_counts,
                precision,
                recall,
                num_thresholds,
                weights,
            ),
            global_step,
            walltime,
        )

    def add_custom_scalars_multilinechart(
        self, tags, category="default", title="untitled"
    ):
        """
        创建多线图的快捷方式。类似于 ``add_custom_scalars()``
        但唯一必需的参数是 *tags*。

        Args:
            tags (list): 在 ``add_scalar()`` 中使用过的标签列表

        Examples::

            writer.add_custom_scalars_multilinechart(['twse/0050', 'twse/2330'])
        """
        torch._C._log_api_usage_once(
            "tensorboard.logging.add_custom_scalars_multilinechart"
        )
        layout = {category: {title: ["Multiline", tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_custom_scalars_marginchart(
        self, tags, category="default", title="untitled"
    ):


These annotations provide a clear explanation of each function's purpose and parameter meanings, following the required format for documentation within the code block.
    ):
        """
        创建一个简便方法用于生成边际图。

        与 `add_custom_scalars()` 类似，但唯一必要的参数是 *tags*，
        它应该恰好包含 3 个元素。

        Args:
            tags (list): 在 `add_scalar()` 中使用的标签列表

        Examples::

            writer.add_custom_scalars_marginchart(['twse/0050', 'twse/2330', 'twse/2006'])
        """
        torch._C._log_api_usage_once(
            "tensorboard.logging.add_custom_scalars_marginchart"
        )
        assert len(tags) == 3
        layout = {category: {title: ["Margin", tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))
    ):
        """
        Add meshes or 3D point clouds to TensorBoard.

        The visualization is based on Three.js,
        so it allows users to interact with the rendered object. Besides the basic definitions
        such as vertices, faces, users can further provide camera parameter, lighting condition, etc.
        Please check https://threejs.org/docs/index.html#manual/en/introduction/Creating-a-scene for
        advanced usage.

        Args:
            tag (str): Data identifier
            vertices (torch.Tensor): List of the 3D coordinates of vertices.
            colors (torch.Tensor): Colors for each vertex
            faces (torch.Tensor): Indices of vertices within each triangle. (Optional)
            config_dict: Dictionary with ThreeJS classes names and configuration.
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Shape:
            vertices: :math:`(B, N, 3)`. (batch, number_of_vertices, channels)

            colors: :math:`(B, N, 3)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.

            faces: :math:`(B, N, 3)`. The values should lie in [0, number_of_vertices] for type `uint8`.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            vertices_tensor = torch.as_tensor([
                [1, 1, 1],
                [-1, -1, 1],
                [1, -1, -1],
                [-1, 1, -1],
            ], dtype=torch.float).unsqueeze(0)
            colors_tensor = torch.as_tensor([
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 255],
            ], dtype=torch.int).unsqueeze(0)
            faces_tensor = torch.as_tensor([
                [0, 2, 3],
                [0, 3, 1],
                [0, 1, 2],
                [1, 3, 2],
            ], dtype=torch.int).unsqueeze(0)

            writer = SummaryWriter()
            writer.add_mesh('my_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)

            writer.close()
        """
        # 记录使用了 tensorboard.logging.add_mesh API
        torch._C._log_api_usage_once("tensorboard.logging.add_mesh")
        # 调用底层的文件写入器来添加可视化的网格数据到事件文件中
        self._get_file_writer().add_summary(
            mesh(tag, vertices, colors, faces, config_dict), global_step, walltime
        )

    def flush(self):
        """
        Flushes the event file to disk.

        Call this method to make sure that all pending events have been written to
        disk.
        """
        if self.all_writers is None:
            return
        # 循环刷新所有写入器的事件到磁盘
        for writer in self.all_writers.values():
            writer.flush()

    def close(self):
        """
        Closes the event file.

        If the event file is not None, it will flush all writers and then close the file.
        """
        if self.all_writers is None:
            return  # ignore double close
        # 循环刷新所有写入器的事件到磁盘并关闭文件
        for writer in self.all_writers.values():
            writer.flush()
            writer.close()
        self.file_writer = self.all_writers = None

    def __enter__(self):
        return self
    # 定义一个特殊方法 __exit__，用于对象的上下文管理
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 调用对象的 close 方法，关闭资源
        self.close()
```