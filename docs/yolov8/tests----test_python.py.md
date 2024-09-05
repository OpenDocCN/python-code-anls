# `.\yolov8\tests\test_python.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

import contextlib  # 上下文管理工具
import urllib  # URL 处理模块
from copy import copy  # 复制对象的浅拷贝
from pathlib import Path  # 处理路径的对象

import cv2  # OpenCV 库
import numpy as np  # 数组操作库
import pytest  # 测试框架
import torch  # PyTorch 深度学习库
import yaml  # YAML 格式处理库
from PIL import Image  # Python 图像库

from tests import CFG, IS_TMP_WRITEABLE, MODEL, SOURCE, TMP  # 导入测试模块
from ultralytics import RTDETR, YOLO  # 导入 YOLO 和 RTDETR 模型类
from ultralytics.cfg import MODELS, TASK2DATA, TASKS  # 导入配置相关模块
from ultralytics.data.build import load_inference_source  # 导入数据构建函数
from ultralytics.utils import (  # 导入工具函数和变量
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    LOGGER,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    checks,
)
from ultralytics.utils.downloads import download  # 导入下载函数
from ultralytics.utils.torch_utils import TORCH_1_9  # 导入 PyTorch 工具函数


def test_model_forward():
    """Test the forward pass of the YOLO model."""
    model = YOLO(CFG)  # 使用给定配置创建 YOLO 模型对象
    model(source=None, imgsz=32, augment=True)  # 测试不同参数的模型前向传播


def test_model_methods():
    """Test various methods and properties of the YOLO model to ensure correct functionality."""
    model = YOLO(MODEL)  # 使用给定模型路径创建 YOLO 模型对象

    # Model methods
    model.info(verbose=True, detailed=True)  # 调用模型的信息打印方法，详细展示
    model = model.reset_weights()  # 重置模型的权重
    model = model.load(MODEL)  # 加载指定模型
    model.to("cpu")  # 将模型转移到 CPU 设备
    model.fuse()  # 融合模型
    model.clear_callback("on_train_start")  # 清除指定的回调函数
    model.reset_callbacks()  # 重置所有回调函数

    # Model properties
    _ = model.names  # 获取模型的类别名称
    _ = model.device  # 获取模型当前设备
    _ = model.transforms  # 获取模型的数据转换
    _ = model.task_map  # 获取模型的任务映射


def test_model_profile():
    """Test profiling of the YOLO model with `profile=True` to assess performance and resource usage."""
    from ultralytics.nn.tasks import DetectionModel  # 导入检测模型类

    model = DetectionModel()  # 创建检测模型对象
    im = torch.randn(1, 3, 64, 64)  # 创建输入张量
    _ = model.predict(im, profile=True)  # 使用性能分析模式进行模型预测


@pytest.mark.skipif(not IS_TMP_WRITEABLE, reason="directory is not writeable")
def test_predict_txt():
    """Tests YOLO predictions with file, directory, and pattern sources listed in a text file."""
    txt_file = TMP / "sources.txt"  # 创建临时文件路径
    with open(txt_file, "w") as f:
        for x in [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]:
            f.write(f"{x}\n")  # 将多种数据源写入文本文件

    _ = YOLO(MODEL)(source=txt_file, imgsz=32)  # 使用文本文件中的数据源进行 YOLO 模型预测


@pytest.mark.parametrize("model_name", MODELS)
def test_predict_img(model_name):
    """Test YOLO model predictions on various image input types and sources, including online images."""
    model = YOLO(WEIGHTS_DIR / model_name)  # 使用给定模型名称加载 YOLO 模型

    im = cv2.imread(str(SOURCE))  # 读取输入图像为 numpy 数组
    assert len(model(source=Image.open(SOURCE), save=True, verbose=True, imgsz=32)) == 1  # 使用 PIL 图像进行模型预测
    assert len(model(source=im, save=True, save_txt=True, imgsz=32)) == 1  # 使用 numpy 数组进行模型预测
    assert len(model(torch.rand((2, 3, 32, 32)), imgsz=32)) == 2  # 使用 Tensor 数据进行批处理预测
    assert len(model(source=[im, im], save=True, save_txt=True, imgsz=32)) == 2  # 使用多个输入进行批处理预测
    assert len(list(model(source=[im, im], save=True, stream=True, imgsz=32))) == 2  # 使用流式数据进行预测
    assert len(model(torch.zeros(320, 640, 3).numpy().astype(np.uint8), imgsz=32)) == 1  # 使用 Tensor 转换为 numpy 数组进行预测
    batch = [
        str(SOURCE),  # 将 SOURCE 转换为字符串并存储在列表中，表示文件名
        Path(SOURCE),  # 使用 SOURCE 创建一个 Path 对象，并存储在列表中，表示路径
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg" if ONLINE else SOURCE,  # 如果 ONLINE 变量为真，则使用 GitHub 上的 URL，否则使用 SOURCE 变量，表示统一资源标识符（URI）
        cv2.imread(str(SOURCE)),  # 使用 OpenCV 读取 SOURCE 变量指定的图像，并将其存储在列表中
        Image.open(SOURCE),  # 使用 PIL 库打开 SOURCE 变量指定的图像，并将其存储在列表中
        np.zeros((320, 640, 3), dtype=np.uint8),  # 创建一个 320x640 大小，数据类型为 uint8 的全零数组，并存储在列表中，表示使用 numpy 库
    ]
    assert len(model(batch, imgsz=32)) == len(batch)  # 断言模型处理批量数据的输出长度与输入列表 batch 的长度相同
@pytest.mark.parametrize("model", MODELS)
def test_predict_visualize(model):
    """Test model prediction methods with 'visualize=True' to generate and display prediction visualizations."""
    # 使用不同的模型参数化测试模型的预测方法，设置 visualize=True 以生成和显示预测的可视化结果
    YOLO(WEIGHTS_DIR / model)(SOURCE, imgsz=32, visualize=True)


def test_predict_grey_and_4ch():
    """Test YOLO prediction on SOURCE converted to greyscale and 4-channel images with various filenames."""
    # 测试 YOLO 模型在将 SOURCE 转换为灰度图和四通道图像，并使用不同的文件名进行测试
    im = Image.open(SOURCE)
    directory = TMP / "im4"
    directory.mkdir(parents=True, exist_ok=True)

    source_greyscale = directory / "greyscale.jpg"
    source_rgba = directory / "4ch.png"
    source_non_utf = directory / "non_UTF_测试文件_tést_image.jpg"
    source_spaces = directory / "image with spaces.jpg"

    im.convert("L").save(source_greyscale)  # 将图像转换为灰度图并保存
    im.convert("RGBA").save(source_rgba)  # 将图像转换为四通道 PNG 并保存
    im.save(source_non_utf)  # 使用包含非 UTF 字符的文件名保存图像
    im.save(source_spaces)  # 使用包含空格的文件名保存图像

    # 推断过程
    model = YOLO(MODEL)
    for f in source_rgba, source_greyscale, source_non_utf, source_spaces:
        for source in Image.open(f), cv2.imread(str(f)), f:
            # 对每个文件进行模型预测，设置 save=True 和 verbose=True，imgsz=32
            results = model(source, save=True, verbose=True, imgsz=32)
            assert len(results) == 1  # 验证是否运行了一次图像预测
        f.unlink()  # 清理生成的临时文件


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_youtube():
    """Test YOLO model on a YouTube video stream, handling potential network-related errors."""
    # 在 YouTube 视频流上测试 YOLO 模型，处理可能出现的网络相关错误
    model = YOLO(MODEL)
    try:
        model.predict("https://youtu.be/G17sBkb38XQ", imgsz=96, save=True)
    # 处理因网络连接问题引起的错误，例如 'urllib.error.HTTPError: HTTP Error 429: Too Many Requests'
    except (urllib.error.HTTPError, ConnectionError) as e:
        LOGGER.warning(f"WARNING: YouTube Test Error: {e}")


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.skipif(not IS_TMP_WRITEABLE, reason="directory is not writeable")
def test_track_stream():
    """
    Tests streaming tracking on a short 10 frame video using ByteTrack tracker and different GMC methods.

    Note imgsz=160 required for tracking for higher confidence and better matches.
    """
    # 测试在短10帧视频上使用 ByteTrack 跟踪器和不同的全局运动补偿（GMC）方法进行实时跟踪

    video_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/decelera_portrait_min.mov"
    model = YOLO(MODEL)
    model.track(video_url, imgsz=160, tracker="bytetrack.yaml")  # 使用 ByteTrack 跟踪器进行跟踪
    model.track(video_url, imgsz=160, tracker="botsort.yaml", save_frames=True)  # 测试帧保存功能

    # 测试不同的全局运动补偿（GMC）方法
    for gmc in "orb", "sift", "ecc":
        with open(ROOT / "cfg/trackers/botsort.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        tracker = TMP / f"botsort-{gmc}.yaml"
        data["gmc_method"] = gmc
        with open(tracker, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
        model.track(video_url, imgsz=160, tracker=tracker)


def test_val():
    # 这是一个空测试函数，没有任何代码内容
    # 使用 YOLO 模型的验证模式进行测试
    # 实例化 YOLO 类，并调用其 val 方法，传入以下参数：
    #   - data="coco8.yaml": 指定配置文件为 "coco8.yaml"
    #   - imgsz=32: 指定图像尺寸为 32
    #   - save_hybrid=True: 设置保存混合结果为 True
    YOLO(MODEL).val(data="coco8.yaml", imgsz=32, save_hybrid=True)
def test_train_scratch():
    """Test training the YOLO model from scratch using the provided configuration."""
    # 创建一个 YOLO 模型对象，使用给定的配置 CFG
    model = YOLO(CFG)
    # 使用指定参数训练模型：数据为 coco8.yaml，训练周期为 2，图像大小为 32 像素，缓存方式为磁盘，批量大小为 -1，关闭马赛克效果，命名为 "model"
    model.train(data="coco8.yaml", epochs=2, imgsz=32, cache="disk", batch=-1, close_mosaic=1, name="model")
    # 使用模型处理 SOURCE 数据
    model(SOURCE)


def test_train_pretrained():
    """Test training of the YOLO model starting from a pre-trained checkpoint."""
    # 创建一个 YOLO 模型对象，从预训练的检查点 WEIGHTS_DIR / "yolov8n-seg.pt" 开始
    model = YOLO(WEIGHTS_DIR / "yolov8n-seg.pt")
    # 使用指定参数训练模型：数据为 coco8-seg.yaml，训练周期为 1，图像大小为 32 像素，缓存方式为 RAM，复制粘贴概率为 0.5，混合比例为 0.5，命名为 0
    model.train(data="coco8-seg.yaml", epochs=1, imgsz=32, cache="ram", copy_paste=0.5, mixup=0.5, name=0)
    # 使用模型处理 SOURCE 数据
    model(SOURCE)


def test_all_model_yamls():
    """Test YOLO model creation for all available YAML configurations in the `cfg/models` directory."""
    # 遍历 cfg/models 目录下所有的 YAML 配置文件
    for m in (ROOT / "cfg" / "models").rglob("*.yaml"):
        # 如果文件名包含 "rtdetr"
        if "rtdetr" in m.name:
            # 如果使用的是 Torch 版本 1.9 及以上
            if TORCH_1_9:
                # 创建 RTDETR 模型对象，传入 m.name 文件名，对 SOURCE 数据进行处理，图像大小为 640
                _ = RTDETR(m.name)(SOURCE, imgsz=640)  # 必须为 640
        else:
            # 创建 YOLO 模型对象，传入 m.name 文件名
            YOLO(m.name)


def test_workflow():
    """Test the complete workflow including training, validation, prediction, and exporting."""
    # 创建一个 YOLO 模型对象，使用指定的 MODEL
    model = YOLO(MODEL)
    # 训练模型：数据为 coco8.yaml，训练周期为 1，图像大小为 32 像素，优化器选择 SGD
    model.train(data="coco8.yaml", epochs=1, imgsz=32, optimizer="SGD")
    # 进行模型验证，图像大小为 32 像素
    model.val(imgsz=32)
    # 对 SOURCE 数据进行预测，图像大小为 32 像素
    model.predict(SOURCE, imgsz=32)
    # 导出模型为 TorchScript 格式
    model.export(format="torchscript")


def test_predict_callback_and_setup():
    """Test callback functionality during YOLO prediction setup and execution."""

    def on_predict_batch_end(predictor):
        """Callback function that handles operations at the end of a prediction batch."""
        # 获取 predictor.batch 的路径、图像和批量大小
        path, im0s, _ = predictor.batch
        # 将 im0s 转换为列表（如果不是），以便处理多图像情况
        im0s = im0s if isinstance(im0s, list) else [im0s]
        # 创建与预测结果、图像和批量大小相关联的元组列表
        bs = [predictor.dataset.bs for _ in range(len(path))]
        predictor.results = zip(predictor.results, im0s, bs)  # results is List[batch_size]

    # 创建一个 YOLO 模型对象，使用指定的 MODEL
    model = YOLO(MODEL)
    # 添加 on_predict_batch_end 回调函数到模型中
    model.add_callback("on_predict_batch_end", on_predict_batch_end)

    # 加载推理数据源，获取数据集的批量大小
    dataset = load_inference_source(source=SOURCE)
    bs = dataset.bs  # noqa access predictor properties
    # 对数据集进行预测，流式处理，图像大小为 160 像素
    results = model.predict(dataset, stream=True, imgsz=160)  # source already setup
    # 遍历预测结果列表
    for r, im0, bs in results:
        # 打印图像形状信息
        print("test_callback", im0.shape)
        # 打印批量大小信息
        print("test_callback", bs)
        # 获取预测结果的边界框对象
        boxes = r.boxes  # Boxes object for bbox outputs
        print(boxes)


@pytest.mark.parametrize("model", MODELS)
def test_results(model):
    """Ensure YOLO model predictions can be processed and printed in various formats."""
    # 使用指定模型 WEIGHTS_DIR / model 创建 YOLO 模型对象，并对 SOURCE 数据进行预测，图像大小为 160 像素
    results = YOLO(WEIGHTS_DIR / model)([SOURCE, SOURCE], imgsz=160)
    # 遍历预测结果列表
    for r in results:
        # 将结果转换为 CPU 上的 numpy 数组
        r = r.cpu().numpy()
        # 打印 numpy 数组的属性信息及路径
        print(r, len(r), r.path)  # print numpy attributes
        # 将结果转换为 CPU 上的 torch.float32 类型
        r = r.to(device="cpu", dtype=torch.float32)
        # 将结果保存为文本文件，保存置信度信息
        r.save_txt(txt_file=TMP / "runs/tests/label.txt", save_conf=True)
        # 将结果中的区域裁剪保存到指定目录
        r.save_crop(save_dir=TMP / "runs/tests/crops/")
        # 将结果转换为 JSON 格式，并进行规范化处理
        r.tojson(normalize=True)
        # 绘制结果的图像，返回 PIL 图像
        r.plot(pil=True)
        # 绘制结果的置信度图及边界框信息
        r.plot(conf=True, boxes=True)
        # 再次打印结果及路径信息
        print(r, len(r), r.path)  # print after methods


def test_labels_and_crops():
    # 这个函数是空的，未提供代码
    pass
    """Test output from prediction args for saving YOLO detection labels and crops; ensures accurate saving."""
    # 定义图片列表，包括源路径和指定的图像文件路径
    imgs = [SOURCE, ASSETS / "zidane.jpg"]
    # 使用预训练的 YOLO 模型处理图像列表，设置图像大小为160，保存检测结果的文本和裁剪图像
    results = YOLO(WEIGHTS_DIR / "yolov8n.pt")(imgs, imgsz=160, save_txt=True, save_crop=True)
    # 保存路径为结果中第一个元素的保存目录
    save_path = Path(results[0].save_dir)
    # 遍历每个结果
    for r in results:
        # 提取图像文件名作为标签文件名的基础
        im_name = Path(r.path).stem
        # 提取每个检测框的类别索引，转换为整数列表
        cls_idxs = r.boxes.cls.int().tolist()
        # 检查标签文件路径是否存在
        labels = save_path / f"labels/{im_name}.txt"
        assert labels.exists()  # 断言标签文件存在
        # 检查检测结果的数量是否与标签文件中的行数匹配
        assert len(r.boxes.data) == len([line for line in labels.read_text().splitlines() if line])
        # 获取所有裁剪图像的路径
        crop_dirs = list((save_path / "crops").iterdir())
        crop_files = [f for p in crop_dirs for f in p.glob("*")]
        # 断言每个类别索引对应的裁剪目录在裁剪目录中存在
        assert all(r.names.get(c) in {d.name for d in crop_dirs} for c in cls_idxs)
        # 断言裁剪文件数量与检测框数量相匹配
        assert len([f for f in crop_files if im_name in f.name]) == len(r.boxes.data)
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
# 标记为跳过测试，如果环境处于离线状态
def test_data_utils():
    """Test utility functions in ultralytics/data/utils.py, including dataset stats and auto-splitting."""
    # 导入需要测试的函数和模块
    from ultralytics.data.utils import HUBDatasetStats, autosplit
    from ultralytics.utils.downloads import zip_directory

    # from ultralytics.utils.files import WorkingDirectory
    # with WorkingDirectory(ROOT.parent / 'tests'):

    # 遍历任务列表，进行测试
    for task in TASKS:
        # 构建数据文件的路径，例如 coco8.zip
        file = Path(TASK2DATA[task]).with_suffix(".zip")  # i.e. coco8.zip
        # 下载数据文件
        download(f"https://github.com/ultralytics/hub/raw/main/example_datasets/{file}", unzip=False, dir=TMP)
        # 创建数据集统计对象
        stats = HUBDatasetStats(TMP / file, task=task)
        # 生成数据集统计信息的 JSON 文件
        stats.get_json(save=True)
        # 处理图像数据
        stats.process_images()

    # 自动划分数据集
    autosplit(TMP / "coco8")
    # 压缩指定路径下的文件夹
    zip_directory(TMP / "coco8/images/val")  # zip


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
# 标记为跳过测试，如果环境处于离线状态
def test_data_converter():
    """Test dataset conversion functions from COCO to YOLO format and class mappings."""
    # 导入需要测试的函数
    from ultralytics.data.converter import coco80_to_coco91_class, convert_coco

    # 下载 COCO 数据集的实例文件
    file = "instances_val2017.json"
    download(f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{file}", dir=TMP)
    # 将 COCO 数据集转换为 YOLO 格式
    convert_coco(labels_dir=TMP, save_dir=TMP / "yolo_labels", use_segments=True, use_keypoints=False, cls91to80=True)
    # 将 COCO80 类别映射为 COCO91 类别
    coco80_to_coco91_class()


def test_data_annotator():
    """Automatically annotate data using specified detection and segmentation models."""
    # 导入自动标注数据的函数
    from ultralytics.data.annotator import auto_annotate

    # 使用指定的检测和分割模型自动标注数据
    auto_annotate(
        ASSETS,
        det_model=WEIGHTS_DIR / "yolov8n.pt",
        sam_model=WEIGHTS_DIR / "mobile_sam.pt",
        output_dir=TMP / "auto_annotate_labels",
    )


def test_events():
    """Test event sending functionality."""
    # 导入事件发送功能模块
    from ultralytics.hub.utils import Events

    # 创建事件对象
    events = Events()
    events.enabled = True
    cfg = copy(DEFAULT_CFG)  # does not require deepcopy
    cfg.mode = "test"
    # 发送事件
    events(cfg)


def test_cfg_init():
    """Test configuration initialization utilities from the 'ultralytics.cfg' module."""
    # 导入配置初始化相关的函数
    from ultralytics.cfg import check_dict_alignment, copy_default_cfg, smart_value

    # 检查字典对齐性
    with contextlib.suppress(SyntaxError):
        check_dict_alignment({"a": 1}, {"b": 2})
    # 复制默认配置
    copy_default_cfg()
    # 删除复制的配置文件
    (Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")).unlink(missing_ok=False)
    # 对多个值应用智能化处理
    [smart_value(x) for x in ["none", "true", "false"]]


def test_utils_init():
    """Test initialization utilities in the Ultralytics library."""
    # 导入初始化工具函数
    from ultralytics.utils import get_git_branch, get_git_origin_url, get_ubuntu_version, is_github_action_running

    # 获取 Ubuntu 版本信息
    get_ubuntu_version()
    # 检查是否在 GitHub Action 环境下运行
    is_github_action_running()
    # 获取 Git 仓库的远程 URL
    get_git_origin_url()
    # 获取 Git 分支信息
    get_git_branch()


def test_utils_checks():
    """Test various utility checks for filenames, git status, requirements, image sizes, and versions."""
    # 导入各种检查函数
    from ultralytics.utils import checks

    # 检查 YOLOv5u 文件名格式
    checks.check_yolov5u_filename("yolov5n.pt")
    # 检查 Git 仓库状态
    checks.git_describe(ROOT)
    # 检查项目的要求是否符合 requirements.txt 中指定的依赖
    checks.check_requirements()  # check requirements.txt
    
    # 检查图像大小是否在指定范围内，确保宽度和高度均不超过 600 像素
    checks.check_imgsz([600, 600], max_dim=1)
    
    # 检查是否可以显示图像，若不能显示则发出警告
    checks.check_imshow(warn=True)
    
    # 检查指定模块的版本是否符合要求，这里检查 ultralytics 模块是否至少是 8.0.0 版本
    checks.check_version("ultralytics", "8.0.0")
    
    # 打印当前设置和参数，用于调试和确认运行时的配置
    checks.print_args()
@pytest.mark.skipif(WINDOWS, reason="Windows profiling is extremely slow (cause unknown)")
# 如果在 Windows 下运行，跳过此测试，原因是 Windows 上的性能分析非常缓慢（原因不明）
def test_utils_benchmarks():
    """Benchmark model performance using 'ProfileModels' from 'ultralytics.utils.benchmarks'."""
    # 导入性能分析工具 'ProfileModels' 来评估模型性能
    from ultralytics.utils.benchmarks import ProfileModels

    # 使用 ProfileModels 类来对 'yolov8n.yaml' 模型进行性能分析，设置图像大小为 32，最小运行时间为 1 秒，运行 3 次，预热 1 次
    ProfileModels(["yolov8n.yaml"], imgsz=32, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()


def test_utils_torchutils():
    """Test Torch utility functions including profiling and FLOP calculations."""
    # 导入相关模块和函数进行测试，包括性能分析和 FLOP 计算
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.utils.torch_utils import get_flops_with_torch_profiler, profile, time_sync

    # 创建一个随机张量作为输入
    x = torch.randn(1, 64, 20, 20)
    # 创建一个 Conv 模型实例
    m = Conv(64, 64, k=1, s=2)

    # 使用 profile 函数对模型 m 进行性能分析，运行 3 次
    profile(x, [m], n=3)
    # 使用 get_flops_with_torch_profiler 函数获取模型 m 的 FLOP
    get_flops_with_torch_profiler(m)
    # 执行时间同步操作
    time_sync()


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
# 如果处于离线环境，跳过此测试
def test_utils_downloads():
    """Test file download utilities from ultralytics.utils.downloads."""
    # 导入文件下载工具函数 get_google_drive_file_info
    from ultralytics.utils.downloads import get_google_drive_file_info

    # 调用 get_google_drive_file_info 函数下载特定 Google Drive 文件的信息
    get_google_drive_file_info("https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link")


def test_utils_ops():
    """Test utility operations functions for coordinate transformation and normalization."""
    # 导入坐标转换和归一化等操作函数
    from ultralytics.utils.ops import (
        ltwh2xywh,
        ltwh2xyxy,
        make_divisible,
        xywh2ltwh,
        xywh2xyxy,
        xywhn2xyxy,
        xywhr2xyxyxyxy,
        xyxy2ltwh,
        xyxy2xywh,
        xyxy2xywhn,
        xyxyxyxy2xywhr,
    )

    # 使用 make_divisible 函数，确保 17 能够被 8 整除
    make_divisible(17, torch.tensor([8]))

    # 创建随机框坐标张量
    boxes = torch.rand(10, 4)  # xywh
    # 检查通过 xywh2xyxy 和 xyxy2xywh 函数的转换后的张量是否相等
    torch.allclose(boxes, xyxy2xywh(xywh2xyxy(boxes)))
    # 检查通过 xywhn2xyxy 和 xyxy2xywhn 函数的转换后的张量是否相等
    torch.allclose(boxes, xyxy2xywhn(xywhn2xyxy(boxes)))
    # 检查通过 ltwh2xywh 和 xywh2ltwh 函数的转换后的张量是否相等
    torch.allclose(boxes, ltwh2xywh(xywh2ltwh(boxes)))
    # 检查通过 xyxy2ltwh 和 ltwh2xyxy 函数的转换后的张量是否相等
    torch.allclose(boxes, xyxy2ltwh(ltwh2xyxy(boxes)))

    # 创建带有方向信息的随机框坐标张量
    boxes = torch.rand(10, 5)  # xywhr for OBB
    # 随机生成方向信息
    boxes[:, 4] = torch.randn(10) * 30
    # 检查通过 xywhr2xyxyxyxy 和 xyxyxyxy2xywhr 函数的转换后的张量是否相等，相对误差容忍度为 1e-3
    torch.allclose(boxes, xyxyxyxy2xywhr(xywhr2xyxyxyxy(boxes)), rtol=1e-3)


def test_utils_files():
    """Test file handling utilities including file age, date, and paths with spaces."""
    # 导入文件处理工具函数，包括文件年龄、日期和带空格路径的处理
    from ultralytics.utils.files import file_age, file_date, get_latest_run, spaces_in_path

    # 获取指定文件的年龄
    file_age(SOURCE)
    # 获取指定文件的日期
    file_date(SOURCE)
    # 获取根目录下运行记录的最新一次运行
    get_latest_run(ROOT / "runs")

    # 创建一个带有空格路径的临时目录
    path = TMP / "path/with spaces"
    path.mkdir(parents=True, exist_ok=True)
    # 在带有空格路径的临时目录中执行 spaces_in_path 函数，返回处理后的新路径并打印
    with spaces_in_path(path) as new_path:
        print(new_path)


@pytest.mark.slow
def test_utils_patches_torch_save():
    """Test torch_save backoff when _torch_save raises RuntimeError to ensure robustness."""
    # 导入测试函数和 mock
    from unittest.mock import MagicMock, patch

    # 导入要测试的函数 torch_save
    from ultralytics.utils.patches import torch_save

    # 创建一个 mock 对象，模拟 RuntimeError 异常
    mock = MagicMock(side_effect=RuntimeError)

    # 使用 patch 替换 _torch_save 函数，使其在调用时抛出 RuntimeError 异常
    with patch("ultralytics.utils.patches._torch_save", new=mock):
        # 断言调用 torch_save 函数时会抛出 RuntimeError 异常
        with pytest.raises(RuntimeError):
            torch_save(torch.zeros(1), TMP / "test.pt")
    # 断言，验证 mock 对象的方法被调用的次数是否等于 4
    assert mock.call_count == 4, "torch_save was not attempted the expected number of times"
def test_nn_modules_conv():
    """Test Convolutional Neural Network modules including CBAM, Conv2, and ConvTranspose."""
    from ultralytics.nn.modules.conv import CBAM, Conv2, ConvTranspose, DWConvTranspose2d, Focus

    c1, c2 = 8, 16  # 输入通道数和输出通道数
    x = torch.zeros(4, c1, 10, 10)  # BCHW，创建一个大小为4x8x10x10的张量（批量大小x通道数x高度x宽度）

    # 运行所有未在测试中涵盖的模块
    DWConvTranspose2d(c1, c2)(x)  # 使用DWConvTranspose2d进行转置卷积操作
    ConvTranspose(c1, c2)(x)  # 使用ConvTranspose进行转置卷积操作
    Focus(c1, c2)(x)  # 使用Focus模块处理输入
    CBAM(c1)(x)  # 使用CBAM模块处理输入

    # 合并操作
    m = Conv2(c1, c2)  # 创建Conv2对象
    m.fuse_convs()  # 融合卷积操作
    m(x)  # 对输入x进行Conv2操作


def test_nn_modules_block():
    """Test various blocks in neural network modules including C1, C3TR, BottleneckCSP, C3Ghost, and C3x."""
    from ultralytics.nn.modules.block import C1, C3TR, BottleneckCSP, C3Ghost, C3x

    c1, c2 = 8, 16  # 输入通道数和输出通道数
    x = torch.zeros(4, c1, 10, 10)  # BCHW，创建一个大小为4x8x10x10的张量（批量大小x通道数x高度x宽度）

    # 运行所有未在测试中涵盖的模块
    C1(c1, c2)(x)  # 使用C1模块处理输入
    C3x(c1, c2)(x)  # 使用C3x模块处理输入
    C3TR(c1, c2)(x)  # 使用C3TR模块处理输入
    C3Ghost(c1, c2)(x)  # 使用C3Ghost模块处理输入
    BottleneckCSP(c1, c2)(x)  # 使用BottleneckCSP模块处理输入


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_hub():
    """Test Ultralytics HUB functionalities (e.g. export formats, logout)."""
    from ultralytics.hub import export_fmts_hub, logout
    from ultralytics.hub.utils import smart_request

    export_fmts_hub()  # 调用导出格式函数
    logout()  # 执行注销操作
    smart_request("GET", "https://github.com", progress=True)  # 发起一个GET请求至GitHub


@pytest.fixture
def image():
    """Load and return an image from a predefined source using OpenCV."""
    return cv2.imread(str(SOURCE))  # 使用OpenCV从预定义源加载并返回一张图像


@pytest.mark.parametrize(
    "auto_augment, erasing, force_color_jitter",
    [
        (None, 0.0, False),
        ("randaugment", 0.5, True),
        ("augmix", 0.2, False),
        ("autoaugment", 0.0, True),
    ],
)
def test_classify_transforms_train(image, auto_augment, erasing, force_color_jitter):
    """Tests classification transforms during training with various augmentations to ensure proper functionality."""
    from ultralytics.data.augment import classify_augmentations

    transform = classify_augmentations(
        size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip=0.5,
        vflip=0.5,
        auto_augment=auto_augment,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        force_color_jitter=force_color_jitter,
        erasing=erasing,
    )

    transformed_image = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    assert transformed_image.shape == (3, 224, 224)  # 断言转换后图像的形状为(3, 224, 224)
    assert torch.is_tensor(transformed_image)  # 断言转换后图像是一个PyTorch张量
    assert transformed_image.dtype == torch.float32  # 断言转换后图像的数据类型为torch.float32


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_model_tune():
    """Tune YOLO model for performance improvement."""
    YOLO("yolov8n-pose.pt").tune(data="coco8-pose.yaml", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")
    # 使用 YOLO 模型加载 "yolov8n-cls.pt" 权重文件，并进行调参和微调
    YOLO("yolov8n-cls.pt").tune(data="imagenet10", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")
# 定义测试函数，用于测试模型嵌入（embeddings）
def test_model_embeddings():
    """Test YOLO model embeddings."""
    # 创建 YOLO 检测模型对象，使用指定模型
    model_detect = YOLO(MODEL)
    # 创建 YOLO 分割模型对象，使用指定权重文件
    model_segment = YOLO(WEIGHTS_DIR / "yolov8n-seg.pt")

    # 分别测试批次大小为1和2的情况
    for batch in [SOURCE], [SOURCE, SOURCE]:  # test batch size 1 and 2
        # 断言检测模型返回的嵌入特征长度与批次大小相同
        assert len(model_detect.embed(source=batch, imgsz=32)) == len(batch)
        # 断言分割模型返回的嵌入特征长度与批次大小相同
        assert len(model_segment.embed(source=batch, imgsz=32)) == len(batch)


# 使用 pytest.mark.skipif 标记，如果条件满足，则跳过该测试
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="YOLOWorld with CLIP is not supported in Python 3.12")
# 定义测试函数，测试支持 CLIP 的 YOLO 模型
def test_yolo_world():
    """Tests YOLO world models with CLIP support, including detection and training scenarios."""
    # 创建 YOLO World 模型对象，加载指定模型
    model = YOLO("yolov8s-world.pt")  # no YOLOv8n-world model yet
    # 设置模型的分类类别为 ["tree", "window"]
    model.set_classes(["tree", "window"])
    # 运行模型进行目标检测，设定置信度阈值为 0.01
    model(SOURCE, conf=0.01)

    # 创建 YOLO Worldv2 模型对象，加载指定模型
    model = YOLO("yolov8s-worldv2.pt")  # no YOLOv8n-world model yet
    # 从预训练模型开始训练，最后阶段包括评估
    # 使用 dota8.yaml，该文件少量类别以减少 CLIP 模型推理时间
    model.train(
        data="dota8.yaml",
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
    )

    # 测试 WorWorldTrainerFromScratch
    from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

    # 创建 YOLO Worldv2 模型对象，加载指定模型
    model = YOLO("yolov8s-worldv2.yaml")  # no YOLOv8n-world model yet
    # 从头开始训练模型
    model.train(
        data={"train": {"yolo_data": ["dota8.yaml"]}, "val": {"yolo_data": ["dota8.yaml"]}},
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
        trainer=WorldTrainerFromScratch,
    )


# 定义测试函数，测试 YOLOv10 模型的训练、验证和预测步骤，使用最小配置
def test_yolov10():
    """Test YOLOv10 model training, validation, and prediction steps with minimal configurations."""
    # 创建 YOLOv10n 模型对象，加载指定模型配置文件
    model = YOLO("yolov10n.yaml")
    # 训练模型，使用 coco8.yaml 数据集，训练1轮，图像尺寸为32，使用磁盘缓存，关闭马赛克
    model.train(data="coco8.yaml", epochs=1, imgsz=32, close_mosaic=1, cache="disk")
    # 验证模型，使用 coco8.yaml 数据集，图像尺寸为32
    model.val(data="coco8.yaml", imgsz=32)
    # 进行预测，图像尺寸为32，保存文本输出和裁剪后的图像，进行数据增强
    model.predict(imgsz=32, save_txt=True, save_crop=True, augment=True)
    # 对给定的 SOURCE 数据进行预测
    model(SOURCE)
```