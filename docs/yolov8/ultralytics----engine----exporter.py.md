# `.\yolov8\ultralytics\engine\exporter.py`

```py
# 导入必要的库和模块

import gc  # 垃圾回收模块，用于管理内存中不再需要的对象
import json  # JSON 数据处理模块
import os  # 操作系统相关功能模块
import shutil  # 文件操作模块，用于复制、移动和删除文件
import subprocess  # 子进程管理模块，用于执行外部命令
import time  # 时间模块，提供时间相关的函数
import warnings  # 警告处理模块，用于管理警告信息

from copy import deepcopy  # 深拷贝函数，用于创建对象的完整拷贝
from datetime import datetime  # 日期时间模块，提供日期和时间的处理功能
from pathlib import Path  # 路径操作模块，用于处理文件和目录路径

import numpy as np  # 数组处理模块，提供多维数组和矩阵操作
import torch  # PyTorch 深度学习库

from ultralytics.cfg import TASK2DATA, get_cfg  # 导入特定配置和配置获取函数
from ultralytics.data import build_dataloader  # 数据加载器构建函数
from ultralytics.data.dataset import YOLODataset  # YOLO 数据集类
from ultralytics.data.utils import check_cls_dataset, check_det_dataset  # 数据集检查函数
# 导入需要的模块和函数
from ultralytics.nn.autobackend import check_class_names, default_class_names
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, WorldModel
from ultralytics.utils import (
    ARM64,
    DEFAULT_CFG,
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    PYTHON_VERSION,
    ROOT,
    WINDOWS,
    __version__,
    callbacks,
    colorstr,
    get_default_args,
    yaml_save,
)
from ultralytics.utils.checks import check_imgsz, check_is_path_safe, check_requirements, check_version
from ultralytics.utils.downloads import attempt_download_asset, get_github_assets, safe_download
from ultralytics.utils.files import file_size, spaces_in_path
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import TORCH_1_13, get_latest_opset, select_device, smart_inference_mode


# 定义一个函数，用于返回YOLOv8模型的导出格式
def export_formats():
    """YOLOv8 export formats."""
    # 引入pandas，提高“import ultralytics”的速度
    import pandas  # scope for faster 'import ultralytics'

    # 定义支持的导出格式列表
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False],
        ["TensorFlow.js", "tfjs", "_web_model", True, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
        ["NCNN", "ncnn", "_ncnn_model", True, True],
    ]
    # 返回格式列表的DataFrame形式，包含列名
    return pandas.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


# 定义一个函数，用于提取TensorFlow GraphDef模型的输出节点名称列表
def gd_outputs(gd):
    """TensorFlow GraphDef model output node names."""
    # 初始化节点名称列表和输入节点列表
    name_list, input_list = [], []
    # 遍历GraphDef对象的节点，获取节点名称和输入节点名称
    for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
        name_list.append(node.name)
        input_list.extend(node.input)
    # 返回排序后的输出节点名称列表，排除无关节点和输入节点
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))


# 定义一个装饰器函数，用于YOLOv8模型的导出
def try_export(inner_func):
    """YOLOv8 export decorator, i.e. @try_export."""
    # 获取内部函数的默认参数
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Export a model."""
        # 获取导出前缀
        prefix = inner_args["prefix"]
        try:
            # 使用Profile类记录导出时间
            with Profile() as dt:
                # 调用内部函数获取导出文件和模型对象
                f, model = inner_func(*args, **kwargs)
            # 打印导出成功信息，包括导出时间、文件大小
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{f}' ({file_size(f):.1f} MB)")
            # 返回导出的文件名和模型对象
            return f, model
        except Exception as e:
            # 打印导出失败信息，并抛出异常
            LOGGER.info(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")
            raise e

    # 返回外部函数对象
    return outer_func


# 定义一个导出类Exporter，用于导出模型
class Exporter:
    """
    A class for exporting a model.
    """

    # 在此处可以添加具体的导出方法和逻辑，根据实际需求编写
    """
    Attributes:
        args (SimpleNamespace): Configuration for the exporter.
        callbacks (list, optional): List of callback functions. Defaults to None.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the Exporter class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
            _callbacks (dict, optional): Dictionary of callback functions. Defaults to None.
        """
        # 获取配置参数并存储在self.args中
        self.args = get_cfg(cfg, overrides)

        # 如果输出格式为'coreml'或'mlmodel'，尝试修复protobuf<3.20.x的错误
        if self.args.format.lower() in {"coreml", "mlmodel"}:
            # 设置环境变量，解决TensorBoard回调之前的问题
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        # 设置回调函数列表，如果未提供_callbacks，则使用默认回调函数
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        # 将集成回调函数添加到回调列表中
        callbacks.add_integration_callbacks(self)

    @smart_inference_mode()
    def get_int8_calibration_dataloader(self, prefix=""):
        """Build and return a dataloader suitable for calibration of INT8 models."""
        # 记录信息，指示正在从指定数据集中收集INT8校准图像
        LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
        
        # 根据任务类型选择适当的数据集处理函数
        data = (check_cls_dataset if self.model.task == "classify" else check_det_dataset)(self.args.data)
        
        # 创建YOLO数据集对象，用于模型校准
        dataset = YOLODataset(
            data[self.args.split or "val"],  # 选择验证集或其他指定的数据集分割
            data=data,
            task=self.model.task,
            imgsz=self.imgsz[0],  # 图像尺寸
            augment=False,
            batch_size=self.args.batch * 2,  # TensorRT INT8校准应使用2倍批处理大小
        )
        
        # 数据集的长度
        n = len(dataset)
        # 如果数据集长度小于300，则发出警告
        if n < 300:
            LOGGER.warning(f"{prefix} WARNING ⚠️ >300 images recommended for INT8 calibration, found {n} images.")
        
        # 构建数据加载器，并返回
        return build_dataloader(dataset, batch=self.args.batch * 2, workers=0)  # 批量加载所需的参数设置

    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):
        """YOLOv8 TorchScript model export."""
        # 记录信息，指示使用torch版本开始导出TorchScript模型
        LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}...")
        
        # 设置导出文件路径
        f = self.file.with_suffix(".torchscript")
        
        # 使用torch.jit.trace对模型进行追踪，生成TorchScript表示
        ts = torch.jit.trace(self.model, self.im, strict=False)
        
        # 准备额外的文件，以便与模型一起导出
        extra_files = {"config.txt": json.dumps(self.metadata)}  # torch._C.ExtraFilesMap()
        
        # 如果设置了优化选项，则进行模型优化
        if self.args.optimize:
            LOGGER.info(f"{prefix} optimizing for mobile...")
            from torch.utils.mobile_optimizer import optimize_for_mobile

            # 对模型进行移动端优化，并保存为Lite解释器可以加载的格式
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            # 直接保存TorchScript模型
            ts.save(str(f), _extra_files=extra_files)
        
        # 返回导出的文件路径和空值（None）
        return f, None

    @try_export
    # 定义导出 ONNX 模型的方法，可选参数为前缀字符串
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """YOLOv8 ONNX export."""
        # 定义所需的第三方库依赖
        requirements = ["onnx>=1.12.0"]
        # 如果设置了简化选项，则添加相关的库依赖
        if self.args.simplify:
            requirements += ["onnxslim>=0.1.31", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]
        # 检查所需的库依赖是否满足
        check_requirements(requirements)
        # 导入 onnx 库
        import onnx  # noqa

        # 获取操作集版本号，若未指定则使用最新版本
        opset_version = self.args.opset or get_latest_opset()
        # 打印导出信息，包括 onnx 版本和操作集版本
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...")
        # 设定导出后的文件路径
        f = str(self.file.with_suffix(".onnx"))

        # 根据模型类型设置输出节点名称
        output_names = ["output0", "output1"] if isinstance(self.model, SegmentationModel) else ["output0"]
        # 获取是否启用动态形状的标志
        dynamic = self.args.dynamic
        # 若启用动态形状
        if dynamic:
            # 设置动态形状的映射关系，针对不同模型类型设定不同的动态形状
            dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
            if isinstance(self.model, SegmentationModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 116, 8400)
                dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
            elif isinstance(self.model, DetectionModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 84, 8400)

        # 导出 ONNX 模型
        torch.onnx.export(
            self.model.cpu() if dynamic else self.model,  # 若启用动态形状，导出前先将模型移至 CPU
            self.im.cpu() if dynamic else self.im,  # 若启用动态形状，导出前先将输入数据移至 CPU
            f,
            verbose=False,
            opset_version=opset_version,
            do_constant_folding=True,  # 是否执行常量折叠优化
            input_names=["images"],  # 输入节点的名称
            output_names=output_names,  # 输出节点的名称
            dynamic_axes=dynamic or None,  # 动态形状的轴信息，若未启用动态形状则为 None
        )

        # 加载导出的 ONNX 模型
        model_onnx = onnx.load(f)
        # 检查 ONNX 模型的有效性
        # onnx.checker.check_model(model_onnx)  # 检查 ONNX 模型

        # 如果设置了简化选项，则尝试使用 onnxslim 进行模型简化
        if self.args.simplify:
            try:
                import onnxslim
                # 使用 onnxslim 进行模型简化
                LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
                model_onnx = onnxslim.slim(model_onnx)

                # ONNX 模型简化器（已弃用，需在 'cmake' 和 Conda CI 环境下编译）
                # import onnxsim
                # model_onnx, check = onnxsim.simplify(model_onnx)
                # assert check, "Simplified ONNX model could not be validated"
            except Exception as e:
                # 输出简化失败的警告信息
                LOGGER.warning(f"{prefix} simplifier failure: {e}")

        # 将元数据添加到模型中
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        # 保存最终的 ONNX 模型
        onnx.save(model_onnx, f)
        # 返回导出后的 ONNX 文件路径及模型对象
        return f, model_onnx

    @try_export
    @try_export
    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):
        """YOLOv8 Paddle export."""
        # 检查所需的依赖是否已安装
        check_requirements(("paddlepaddle", "x2paddle"))
        # 导入 x2paddle 库
        import x2paddle  # noqa
        from x2paddle.convert import pytorch2paddle  # noqa

        # 记录导出开始信息，并显示 X2Paddle 的版本号
        LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
        # 准备导出的文件路径，用 '_paddle_model' 替换原文件的后缀名
        f = str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}")

        # 使用 pytorch2paddle 将 PyTorch 模型转换为 Paddle 模型
        pytorch2paddle(module=self.model, save_dir=f, jit_type="trace", input_examples=[self.im])  # export
        # 将 metadata 保存为 YAML 文件
        yaml_save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        # 返回导出的模型文件路径及空结果
        return f, None

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef:")):
        """YOLOv8 TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow."""
        # 导入 TensorFlow 库和相关功能
        import tensorflow as tf  # noqa
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa

        # 记录导出开始信息，并显示 TensorFlow 的版本号
        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        # 准备导出的文件路径，用 '.pb' 替换原文件的后缀名
        f = self.file.with_suffix(".pb")

        # 将 Keras 模型封装为 TensorFlow 函数
        m = tf.function(lambda x: keras_model(x))  # full model
        # 获取具体函数，以便后续转换为常量
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        # 将模型转换为冻结的 TensorFlow 图定义
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        # 将冻结的图定义写入到指定路径的文件中
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
        # 返回导出的模型文件路径及空结果
        return f, None

    @try_export
    def export_tflite(self, keras_model, nms, agnostic_nms, prefix=colorstr("TensorFlow Lite:")):
        """YOLOv8 TensorFlow Lite export."""
        # BUG https://github.com/ultralytics/ultralytics/issues/13436
        # 导入 TensorFlow 库
        import tensorflow as tf  # noqa

        # 记录导出开始信息，并显示 TensorFlow 的版本号
        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        # 准备保存模型的文件夹路径，用 '_saved_model' 替换原文件的后缀名
        saved_model = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        # 根据选项选择导出的 TensorFlow Lite 模型类型
        if self.args.int8:
            f = saved_model / f"{self.file.stem}_int8.tflite"  # fp32 in/out
        elif self.args.half:
            f = saved_model / f"{self.file.stem}_float16.tflite"  # fp32 in/out
        else:
            f = saved_model / f"{self.file.stem}_float32.tflite"
        # 返回导出的模型文件路径及空结果
        return str(f), None

    @try_export
    # 继续 export 方法的定义
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU:")):
        """YOLOv8 Edge TPU export https://coral.ai/docs/edgetpu/models-intro/."""
        # 输出警告信息，指出Edge TPU可能存在的已知问题
        LOGGER.warning(f"{prefix} WARNING ⚠️ Edge TPU known bug https://github.com/ultralytics/ultralytics/issues/1185")

        # 检查Edge TPU编译器的版本命令
        cmd = "edgetpu_compiler --version"
        help_url = "https://coral.ai/docs/edgetpu/compiler/"
        # 断言当前系统是Linux，否则输出错误信息并提供帮助链接
        assert LINUX, f"export only supported on Linux. See {help_url}"

        # 如果edgetpu_compiler命令返回非零状态码，说明编译器未安装
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True).returncode != 0:
            # 输出信息，说明Edge TPU导出需要安装Edge TPU编译器，并尝试从帮助链接处安装
            LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")
            # 检查系统是否安装了sudo命令
            sudo = subprocess.run("sudo --version >/dev/null", shell=True).returncode == 0  # sudo installed on system
            # 遍历安装Edge TPU编译器的命令列表，如果有sudo权限则加上sudo
            for c in (
                "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",
                'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | '
                "sudo tee /etc/apt/sources.list.d/coral-edgetpu.list",
                "sudo apt-get update",
                "sudo apt-get install edgetpu-compiler",
            ):
                subprocess.run(c if sudo else c.replace("sudo ", ""), shell=True, check=True)

        # 获取Edge TPU编译器的版本信息
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

        # 输出信息，指出使用Edge TPU编译器进行导出，并显示当前编译器的版本号
        LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
        
        # 生成导出后的Edge TPU模型文件名
        f = str(tflite_model).replace(".tflite", "_edgetpu.tflite")  # Edge TPU model

        # 构建Edge TPU编译命令
        cmd = f'edgetpu_compiler -s -d -k 10 --out_dir "{Path(f).parent}" "{tflite_model}"'
        
        # 输出信息，显示正在运行的Edge TPU编译命令
        LOGGER.info(f"{prefix} running '{cmd}'")
        
        # 运行Edge TPU编译命令
        subprocess.run(cmd, shell=True)
        
        # 为导出后的Edge TPU模型添加元数据
        self._add_tflite_metadata(f)
        
        # 返回生成的Edge TPU模型文件名和None
        return f, None
    def export_tfjs(self, prefix=colorstr("TensorFlow.js:")):
        """YOLOv8 TensorFlow.js export."""
        # 检查所需的软件包是否已安装
        check_requirements("tensorflowjs")
        # 如果是ARM64架构，修复导出到TF.js时的一个错误
        if ARM64:
            check_requirements("numpy==1.23.5")
        import tensorflow as tf
        import tensorflowjs as tfjs  # 导入TensorFlow.js库，不生成flake8警告

        # 记录导出开始的信息，包括使用的TensorFlow.js版本号
        LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")
        
        # 创建用于保存导出文件的目录名（去除后缀变成"_web_model"）
        f = str(self.file).replace(self.file.suffix, "_web_model")  # js dir
        # 设置保存*.pb文件路径
        f_pb = str(self.file.with_suffix(".pb"))  # *.pb path

        # 创建一个新的TensorFlow图（Graph），并将模型的GraphDef读入其中
        gd = tf.Graph().as_graph_def()  # TF GraphDef
        with open(f_pb, "rb") as file:
            gd.ParseFromString(file.read())
        
        # 获取输出节点的名称，并以逗号分隔输出
        outputs = ",".join(gd_outputs(gd))
        LOGGER.info(f"\n{prefix} output node names: {outputs}")

        # 根据输入的参数（half或int8），选择相应的量化方式
        quantization = "--quantize_float16" if self.args.half else "--quantize_uint8" if self.args.int8 else ""
        
        # 处理文件路径中可能存在的空格问题，使用contextlib中的函数
        with spaces_in_path(f_pb) as fpb_, spaces_in_path(f) as f_:  # exporter can not handle spaces in path
            # 构建tensorflowjs转换命令
            cmd = (
                "tensorflowjs_converter "
                f'--input_format=tf_frozen_model {quantization} --output_node_names={outputs} "{fpb_}" "{f_}"'
            )
            LOGGER.info(f"{prefix} running '{cmd}'")
            # 运行tensorflowjs转换命令
            subprocess.run(cmd, shell=True)

        # 如果导出的目录路径中含有空格，发出警告
        if " " in f:
            LOGGER.warning(f"{prefix} WARNING ⚠️ your model may not work correctly with spaces in path '{f}'.")

        # 将metadata.yaml保存到导出目录下
        yaml_save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        # 返回导出的目录路径和None
        return f, None
    def _add_tflite_metadata(self, file):
        """Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata."""
        import flatbuffers  # 导入 flatbuffers 模块

        try:
            # TFLite Support bug https://github.com/tensorflow/tflite-support/issues/954#issuecomment-2108570845
            from tensorflow_lite_support.metadata import metadata_schema_py_generated as schema  # 导入 TensorFlow Lite 元数据模块
            from tensorflow_lite_support.metadata.python import metadata  # 导入 TensorFlow Lite 元数据模块
        except ImportError:  # 捕获导入错误，ARM64 系统可能缺少 'tensorflow_lite_support' 包
            from tflite_support import metadata  # 导入 TensorFlow Lite Support 元数据模块
            from tflite_support import metadata_schema_py_generated as schema  # 导入 TensorFlow Lite 元数据模块

        # 创建模型元数据对象
        model_meta = schema.ModelMetadataT()

        # 设置模型名称、版本、作者和许可证信息
        model_meta.name = self.metadata["description"]
        model_meta.version = self.metadata["version"]
        model_meta.author = self.metadata["author"]
        model_meta.license = self.metadata["license"]

        # 标签文件处理
        tmp_file = Path(file).parent / "temp_meta.txt"
        with open(tmp_file, "w") as f:
            f.write(str(self.metadata))

        # 创建关联的文件对象
        label_file = schema.AssociatedFileT()
        label_file.name = tmp_file.name
        label_file.type = schema.AssociatedFileType.TENSOR_AXIS_LABELS

        # 创建输入元数据对象
        input_meta = schema.TensorMetadataT()
        input_meta.name = "image"
        input_meta.description = "Input image to be detected."
        input_meta.content = schema.ContentT()
        input_meta.content.contentProperties = schema.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = schema.ColorSpaceType.RGB
        input_meta.content.contentPropertiesType = schema.ContentProperties.ImageProperties

        # 创建输出元数据对象
        output1 = schema.TensorMetadataT()
        output1.name = "output"
        output1.description = "Coordinates of detected objects, class labels, and confidence score"
        output1.associatedFiles = [label_file]

        # 如果模型任务是 'segment'，则创建第二个输出元数据对象
        if self.model.task == "segment":
            output2 = schema.TensorMetadataT()
            output2.name = "output"
            output2.description = "Mask protos"
            output2.associatedFiles = [label_file]

        # 创建子图元数据对象
        subgraph = schema.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output1, output2] if self.model.task == "segment" else [output1]
        model_meta.subgraphMetadata = [subgraph]

        # 使用 flatbuffers 创建一个 Builder 对象
        b = flatbuffers.Builder(0)

        # 打包模型元数据并设置标识符
        b.Finish(model_meta.Pack(b), metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)

        # 获取元数据缓冲区
        metadata_buf = b.Output()

        # 创建 MetadataPopulator 对象并加载模型文件的元数据
        populator = metadata.MetadataPopulator.with_model_file(str(file))
        populator.load_metadata_buffer(metadata_buf)

        # 加载关联的文件列表到 MetadataPopulator
        populator.load_associated_files([str(tmp_file)])

        # 填充元数据到模型文件中
        populator.populate()

        # 删除临时标签文件
        tmp_file.unlink()
    # 定义一个方法，用于向特定事件的回调列表中添加回调函数
    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        # 将指定事件的回调函数添加到回调列表中
        self.callbacks[event].append(callback)

    # 定义一个方法，用于执行特定事件的所有回调函数
    def run_callbacks(self, event: str):
        """Execute all callbacks for a given event."""
        # 遍历指定事件的回调函数列表，依次执行每个回调函数
        for callback in self.callbacks.get(event, []):
            callback(self)
    # 定义一个名为 IOSDetectModel 的类，用于封装 Ultralytics YOLO 模型，以便导出为 Apple iOS CoreML 格式

    def __init__(self, model, im):
        # 初始化 IOSDetectModel 类，传入 YOLO 模型和示例图像 im
        super().__init__()
        _, _, h, w = im.shape  # 获取图像的批处理大小、通道数、高度和宽度信息
        self.model = model  # 将传入的 YOLO 模型保存到实例变量 self.model 中
        self.nc = len(model.names)  # 计算模型中的类别数目，并保存到实例变量 self.nc 中
        if w == h:
            self.normalize = 1.0 / w  # 如果图像是正方形，则使用标量 1.0/w 进行归一化
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # 如果图像不是正方形，则使用张量进行归一化

    def forward(self, x):
        # 实现模型的前向传播，归一化物体检测模型的预测结果，考虑输入大小相关的因素
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        return cls, xywh * self.normalize  # 返回分类结果和归一化后的坐标信息
```