# `.\yolov8\ultralytics\utils\benchmarks.py`

```py
# 从 glob 模块中导入 glob 函数，用于文件路径的模糊匹配
import glob
# 导入 os 模块，提供了许多与操作系统交互的函数
import os
# 导入 platform 模块，用于获取系统平台信息
import platform
# 导入 re 模块，支持正则表达式操作
import re
# 导入 shutil 模块，提供了高级的文件操作功能
import shutil
# 导入 time 模块，提供时间相关的功能
import time
# 从 pathlib 模块中导入 Path 类，用于操作文件路径
from pathlib import Path

# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 torch.cuda 模块，用于 CUDA 相关操作
import torch.cuda
# 导入 yaml 库，用于处理 YAML 格式的文件
import yaml

# 从 ultralytics 包中导入 YOLO 和 YOLOWorld 类
from ultralytics import YOLO, YOLOWorld
# 从 ultralytics.cfg 模块中导入 TASK2DATA 和 TASK2METRIC 变量
from ultralytics.cfg import TASK2DATA, TASK2METRIC
# 从 ultralytics.engine.exporter 模块中导入 export_formats 函数
from ultralytics.engine.exporter import export_formats
# 从 ultralytics.utils 模块中导入 ARM64, ASSETS, IS_JETSON, IS_RASPBERRYPI 等变量
from ultralytics.utils import ARM64, ASSETS, IS_JETSON, IS_RASPBERRYPI, LINUX, LOGGER, MACOS, TQDM, WEIGHTS_DIR
# 从 ultralytics.utils.checks 模块中导入 IS_PYTHON_3_12, check_requirements, check_yolo 等函数和变量
from ultralytics.utils.checks import IS_PYTHON_3_12, check_requirements, check_yolo
# 从 ultralytics.utils.downloads 模块中导入 safe_download 函数
from ultralytics.utils.downloads import safe_download
# 从 ultralytics.utils.files 模块中导入 file_size 函数
from ultralytics.utils.files import file_size
# 从 ultralytics.utils.torch_utils 模块中导入 select_device 函数
from ultralytics.utils.torch_utils import select_device


def benchmark(
    model=WEIGHTS_DIR / "yolov8n.pt", data=None, imgsz=160, half=False, int8=False, device="cpu", verbose=False
):
    """
    Benchmark a YOLO model across different formats for speed and accuracy.

    Args:
        model (str | Path | optional): Path to the model file or directory. Default is
            Path(SETTINGS['weights_dir']) / 'yolov8n.pt'.
        data (str, optional): Dataset to evaluate on, inherited from TASK2DATA if not passed. Default is None.
        imgsz (int, optional): Image size for the benchmark. Default is 160.
        half (bool, optional): Use half-precision for the model if True. Default is False.
        int8 (bool, optional): Use int8-precision for the model if True. Default is False.
        device (str, optional): Device to run the benchmark on, either 'cpu' or 'cuda'. Default is 'cpu'.
        verbose (bool | float | optional): If True or a float, assert benchmarks pass with given metric.
            Default is False.
    """
    # 函数主体，用于评估 YOLO 模型在不同格式下的速度和准确性，参数详细说明在函数文档字符串中给出
    pass  # 这里是示例，实际代码会在此基础上继续开发
    def benchmark(model='yolov8n.pt', imgsz=640):
        """
        Benchmark function to evaluate model performance.
    
        Args:
            model (str or Path): Path to the model checkpoint.
            imgsz (int): Image size for inference.
    
        Returns:
            df (pandas.DataFrame): A pandas DataFrame with benchmark results for each format, including file size,
                metric, and inference time.
    
        Example:
            ```python
            from ultralytics.utils.benchmarks import benchmark
    
            benchmark(model='yolov8n.pt', imgsz=640)
            ```
        """
        import pandas as pd  # Import pandas library for DataFrame operations
        pd.options.display.max_columns = 10  # Set maximum display columns in pandas DataFrame
        pd.options.display.width = 120  # Set display width for pandas DataFrame
    
        device = select_device(device, verbose=False)  # Select device for model inference
        if isinstance(model, (str, Path)):
            model = YOLO(model)  # Initialize YOLO model if model is given as a string or Path
    
        is_end2end = getattr(model.model.model[-1], "end2end", False)  # Check if model supports end-to-end inference
    
        y = []  # Initialize an empty list to store benchmark results
        t0 = time.time()  # Record current time for benchmarking purposes
    
        check_yolo(device=device)  # Print system information relevant to YOLO
    
        # Create a pandas DataFrame 'df' with columns defined for benchmark results
        df = pd.DataFrame(y, columns=["Format", "Status❔", "Size (MB)", key, "Inference time (ms/im)", "FPS"])
    
        name = Path(model.ckpt_path).name  # Extract the name of the model checkpoint file
        # Construct a string 's' summarizing benchmark results and logging information
        s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({time.time() - t0:.2f}s)\n{df}\n"
        LOGGER.info(s)  # Log 's' to the logger file
    
        with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
            f.write(s)  # Append string 's' to the 'benchmarks.log' file
    
        if verbose and isinstance(verbose, float):
            metrics = df[key].array  # Extract the 'key' column values from the DataFrame 'df'
            floor = verbose  # Set the minimum metric floor to compare against
            # Assert that all metrics are greater than 'floor' if they are not NaN
            assert all(x > floor for x in metrics if pd.notna(x)), f"Benchmark failure: metric(s) < floor {floor}"
    
        return df  # Return the pandas DataFrame 'df' containing benchmark results
class RF100Benchmark:
    """Benchmark YOLO model performance across formats for speed and accuracy."""

    def __init__(self):
        """Function for initialization of RF100Benchmark."""
        # 初始化空列表，用于存储数据集名称
        self.ds_names = []
        # 初始化空列表，用于存储数据集配置文件路径
        self.ds_cfg_list = []
        # 初始化 RF 对象为 None
        self.rf = None
        # 定义验证指标列表
        self.val_metrics = ["class", "images", "targets", "precision", "recall", "map50", "map95"]

    def set_key(self, api_key):
        """
        Set Roboflow API key for processing.

        Args:
            api_key (str): The API key.
        """
        # 检查是否满足 Roboflow 相关的依赖
        check_requirements("roboflow")
        # 导入 Roboflow 模块
        from roboflow import Roboflow
        # 创建 Roboflow 对象并设置 API 密钥
        self.rf = Roboflow(api_key=api_key)

    def parse_dataset(self, ds_link_txt="datasets_links.txt"):
        """
        Parse dataset links and downloads datasets.

        Args:
            ds_link_txt (str): Path to dataset_links file.
        """
        # 如果存在 rf-100 目录，则删除并重新创建；否则直接创建
        (shutil.rmtree("rf-100"), os.mkdir("rf-100")) if os.path.exists("rf-100") else os.mkdir("rf-100")
        # 切换当前工作目录至 rf-100
        os.chdir("rf-100")
        # 在 rf-100 目录下创建 ultralytics-benchmarks 目录
        os.mkdir("ultralytics-benchmarks")
        # 安全下载 datasets_links.txt 文件
        safe_download("https://github.com/ultralytics/assets/releases/download/v0.0.0/datasets_links.txt")

        # 打开数据集链接文件，逐行处理
        with open(ds_link_txt, "r") as file:
            for line in file:
                try:
                    # 使用正则表达式拆分数据集链接
                    _, url, workspace, project, version = re.split("/+", line.strip())
                    # 将项目名称添加到数据集名称列表
                    self.ds_names.append(project)
                    # 组合项目和版本信息
                    proj_version = f"{project}-{version}"
                    # 如果该版本数据集尚未下载，则使用 Roboflow 对象下载到 yolov8 目录下
                    if not Path(proj_version).exists():
                        self.rf.workspace(workspace).project(project).version(version).download("yolov8")
                    else:
                        print("Dataset already downloaded.")
                    # 添加数据集配置文件路径到列表中
                    self.ds_cfg_list.append(Path.cwd() / proj_version / "data.yaml")
                except Exception:
                    continue

        return self.ds_names, self.ds_cfg_list

    @staticmethod
    def fix_yaml(path):
        """
        Function to fix YAML train and val path.

        Args:
            path (str): YAML file path.
        """
        # 使用安全加载方式读取 YAML 文件
        with open(path, "r") as file:
            yaml_data = yaml.safe_load(file)
        # 修改 YAML 文件中的训练和验证路径
        yaml_data["train"] = "train/images"
        yaml_data["val"] = "valid/images"
        # 使用安全写入方式将修改后的 YAML 数据写回文件
        with open(path, "w") as file:
            yaml.safe_dump(yaml_data, file)
    def evaluate(self, yaml_path, val_log_file, eval_log_file, list_ind):
        """
        Model evaluation on validation results.

        Args:
            yaml_path (str): YAML file path.
            val_log_file (str): val_log_file path.
            eval_log_file (str): eval_log_file path.
            list_ind (int): Index for current dataset.
        """
        # 定义跳过的符号列表，这些符号出现在日志行中时将被跳过
        skip_symbols = ["", "⚠️", "💡", "❌"]
        
        # 从 YAML 文件中读取类别名称列表
        with open(yaml_path) as stream:
            class_names = yaml.safe_load(stream)["names"]
        
        # 打开验证日志文件，读取其中的所有行
        with open(val_log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            eval_lines = []
            
            # 遍历每一行日志
            for line in lines:
                # 如果日志行包含需要跳过的符号，则跳过此行
                if any(symbol in line for symbol in skip_symbols):
                    continue
                
                # 将每行日志按空格分隔为条目列表
                entries = line.split(" ")
                # 过滤空字符串并去除每个条目结尾的换行符
                entries = list(filter(lambda val: val != "", entries))
                entries = [e.strip("\n") for e in entries]
                
                # 将符合条件的条目加入到评估结果列表中
                eval_lines.extend(
                    {
                        "class": entries[0],
                        "images": entries[1],
                        "targets": entries[2],
                        "precision": entries[3],
                        "recall": entries[4],
                        "map50": entries[5],
                        "map95": entries[6],
                    }
                    for e in entries
                    if e in class_names or (e == "all" and "(AP)" not in entries and "(AR)" not in entries)
                )
        
        # 初始化 map_val 变量为 0.0
        map_val = 0.0
        
        # 如果评估结果列表中条目数量大于 1，则进行下列操作
        if len(eval_lines) > 1:
            print("There's more dicts")
            # 遍历评估结果列表中的每一个字典
            for lst in eval_lines:
                # 如果当前字典的类别为 "all"，则将 map_val 设置为其 map50 值
                if lst["class"] == "all":
                    map_val = lst["map50"]
        else:
            print("There's only one dict res")
            # 否则，如果评估结果列表中只有一个字典，则将 map_val 设置为第一个字典的 map50 值
            map_val = [res["map50"] for res in eval_lines][0]
        
        # 将结果写入评估日志文件中，格式为 "<数据集名称>: <map_val>"
        with open(eval_log_file, "a") as f:
            f.write(f"{self.ds_names[list_ind]}: {map_val}\n")
    """
    ProfileModels class for profiling different models on ONNX and TensorRT.

    This class profiles the performance of different models, returning results such as model speed and FLOPs.

    Attributes:
        paths (list): Paths of the models to profile.
        num_timed_runs (int): Number of timed runs for the profiling. Default is 100.
        num_warmup_runs (int): Number of warmup runs before profiling. Default is 10.
        min_time (float): Minimum number of seconds to profile for. Default is 60.
        imgsz (int): Image size used in the models. Default is 640.
        half (bool): Flag indicating whether to use half-precision floating point for profiling. Default is True.
        trt (bool): Flag indicating whether to use TensorRT for profiling. Default is True.
        device (torch.device): Device used for profiling. Automatically determined if None.

    Methods:
        profile(): Profiles the models and prints the result.

    Example:
        ```py
        from ultralytics.utils.benchmarks import ProfileModels

        ProfileModels(['yolov8n.yaml', 'yolov8s.yaml'], imgsz=640).profile()
        ```
    """

    def __init__(
        self,
        paths: list,
        num_timed_runs=100,
        num_warmup_runs=10,
        min_time=60,
        imgsz=640,
        half=True,
        trt=True,
        device=None,
    ):
        """
        Initialize the ProfileModels class for profiling models.

        Args:
            paths (list): List of paths of the models to be profiled.
            num_timed_runs (int, optional): Number of timed runs for the profiling. Default is 100.
            num_warmup_runs (int, optional): Number of warmup runs before the actual profiling starts. Default is 10.
            min_time (float, optional): Minimum time in seconds for profiling a model. Default is 60.
            imgsz (int, optional): Size of the image used during profiling. Default is 640.
            half (bool, optional): Flag to indicate whether to use half-precision floating point for profiling. Default is True.
            trt (bool, optional): Flag to indicate whether to profile using TensorRT. Default is True.
            device (torch.device, optional): Device used for profiling. If None, it is determined automatically.
        """
        # 初始化各个属性，用于存储传入的参数和设置默认值
        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.min_time = min_time
        self.imgsz = imgsz
        self.half = half
        self.trt = trt  # 是否运行 TensorRT 的性能分析
        # 如果 device 为 None，则自动确定使用的设备
        self.device = device or torch.device(0 if torch.cuda.is_available() else "cpu")
    def profile(self):
        """
        Logs the benchmarking results of a model, checks metrics against floor and returns the results.
        """
        # 获取所有相关文件路径列表
        files = self.get_files()

        if not files:
            # 若没有找到匹配的 *.pt 或 *.onnx 文件，则打印消息并返回
            print("No matching *.pt or *.onnx files found.")
            return

        table_rows = []
        output = []
        for file in files:
            # 生成引擎文件名（后缀为 .engine）
            engine_file = file.with_suffix(".engine")
            if file.suffix in {".pt", ".yaml", ".yml"}:
                # 如果文件后缀是 .pt, .yaml 或 .yml，创建 YOLO 模型对象
                model = YOLO(str(file))
                model.fuse()  # 执行模型融合操作，以获取正确的参数和GFLOPs（在 model.info() 中）
                model_info = model.info()
                if self.trt and self.device.type != "cpu" and not engine_file.is_file():
                    # 如果启用 TensorRT（self.trt），且设备类型不是 CPU，并且引擎文件不存在，则导出为引擎文件
                    engine_file = model.export(
                        format="engine", half=self.half, imgsz=self.imgsz, device=self.device, verbose=False
                    )
                # 导出 ONNX 文件
                onnx_file = model.export(
                    format="onnx", half=self.half, imgsz=self.imgsz, simplify=True, device=self.device, verbose=False
                )
            elif file.suffix == ".onnx":
                # 如果文件后缀是 .onnx，获取 ONNX 模型信息
                model_info = self.get_onnx_model_info(file)
                onnx_file = file
            else:
                continue

            # 对 TensorRT 模型进行性能分析
            t_engine = self.profile_tensorrt_model(str(engine_file))
            # 对 ONNX 模型进行性能分析
            t_onnx = self.profile_onnx_model(str(onnx_file))
            # 生成表格行数据并添加到列表
            table_rows.append(self.generate_table_row(file.stem, t_onnx, t_engine, model_info))
            # 生成结果字典并添加到输出列表
            output.append(self.generate_results_dict(file.stem, t_onnx, t_engine, model_info))

        # 打印表格
        self.print_table(table_rows)
        # 返回结果输出列表
        return output

    def get_files(self):
        """
        Returns a list of paths for all relevant model files given by the user.
        """
        # 初始化文件列表
        files = []
        for path in self.paths:
            path = Path(path)
            if path.is_dir():
                # 如果路径是目录，则获取目录下所有匹配的文件路径
                extensions = ["*.pt", "*.onnx", "*.yaml"]
                files.extend([file for ext in extensions for file in glob.glob(str(path / ext))])
            elif path.suffix in {".pt", ".yaml", ".yml"}:  # add non-existing
                # 如果路径是文件且后缀符合条件，直接添加到文件列表中
                files.append(str(path))
            else:
                # 否则，获取路径下所有文件路径并添加到文件列表中
                files.extend(glob.glob(str(path)))

        # 打印正在分析的文件列表
        print(f"Profiling: {sorted(files)}")
        # 返回路径对象列表
        return [Path(file) for file in sorted(files)]

    def get_onnx_model_info(self, onnx_file: str):
        """
        Retrieves the information including number of layers, parameters, gradients and FLOPs for an ONNX model
        file.
        """
        # 暂时返回零值表示信息获取未实现
        return 0.0, 0.0, 0.0, 0.0  # return (num_layers, num_params, num_gradients, num_flops)
    def iterative_sigma_clipping(data, sigma=2, max_iters=3):
        """Applies an iterative sigma clipping algorithm to the given data."""
        # 将数据转换为 NumPy 数组
        data = np.array(data)
        # 执行最大迭代次数的循环
        for _ in range(max_iters):
            # 计算数据的平均值和标准差
            mean, std = np.mean(data), np.std(data)
            # 根据均值和标准差进行 sigma 剪切，并获取剪切后的数据
            clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
            # 如果剪切后的数据和原数据长度相同，则退出循环
            if len(clipped_data) == len(data):
                break
            # 更新数据为剪切后的数据，继续下一次迭代
            data = clipped_data
        # 返回最终剪切后的数据
        return data

    def profile_tensorrt_model(self, engine_file: str, eps: float = 1e-3):
        """Profiles the TensorRT model, measuring average run time and standard deviation among runs."""
        # 如果 TensorRT 未初始化或者引擎文件不存在，则返回默认值
        if not self.trt or not Path(engine_file).is_file():
            return 0.0, 0.0

        # 初始化模型和输入数据
        model = YOLO(engine_file)
        input_data = np.random.rand(self.imgsz, self.imgsz, 3).astype(np.float32)  # 必须是 FP32

        # 预热运行
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                model(input_data, imgsz=self.imgsz, verbose=False)
            elapsed = time.time() - start_time

        # 计算运行次数，取最大值作为 min_time 或 num_timed_runs 的倍数
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs * 50)

        # 计时运行
        run_times = []
        for _ in TQDM(range(num_runs), desc=engine_file):
            results = model(input_data, imgsz=self.imgsz, verbose=False)
            # 提取推理速度并转换为毫秒
            run_times.append(results[0].speed["inference"])

        # 对运行时间进行 sigma 剪切
        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)
        # 返回运行时间的平均值和标准差
        return np.mean(run_times), np.std(run_times)
    def profile_onnx_model(self, onnx_file: str, eps: float = 1e-3):
        """Profiles an ONNX model by executing it multiple times and returns the mean and standard deviation of run
        times.
        """
        # 检查运行环境是否满足要求，确保安装了'onnxruntime'库
        check_requirements("onnxruntime")
        import onnxruntime as ort

        # 创建会话选项对象，并设置图优化级别为最大，同时限制线程数为8
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 8  # 限制并行执行的线程数目

        # 创建 ONNX 推理会话对象，指定使用CPU执行提供者
        sess = ort.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])

        # 获取模型输入张量信息
        input_tensor = sess.get_inputs()[0]
        input_type = input_tensor.type
        # 检查输入张量是否具有动态形状
        dynamic = not all(isinstance(dim, int) and dim >= 0 for dim in input_tensor.shape)
        # 根据动态形状设置输入张量的形状
        input_shape = (1, 3, self.imgsz, self.imgsz) if dynamic else input_tensor.shape

        # 将ONNX数据类型映射到numpy数据类型
        if "float16" in input_type:
            input_dtype = np.float16
        elif "float" in input_type:
            input_dtype = np.float32
        elif "double" in input_type:
            input_dtype = np.float64
        elif "int64" in input_type:
            input_dtype = np.int64
        elif "int32" in input_type:
            input_dtype = np.int32
        else:
            raise ValueError(f"Unsupported ONNX datatype {input_type}")

        # 生成随机输入数据，以输入张量的形状和数据类型为基础
        input_data = np.random.rand(*input_shape).astype(input_dtype)
        input_name = input_tensor.name
        output_name = sess.get_outputs()[0].name

        # 预热运行，执行若干次，计算平均时间
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                sess.run([output_name], {input_name: input_data})
            elapsed = time.time() - start_time

        # 计算需要运行的总次数，确保满足最小时间要求或指定的运行次数
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs)

        # 正式计时运行
        run_times = []
        for _ in TQDM(range(num_runs), desc=onnx_file):
            start_time = time.time()
            sess.run([output_name], {input_name: input_data})
            run_times.append((time.time() - start_time) * 1000)  # 将运行时间转换为毫秒

        # 对运行时间进行迭代的sigma剪裁
        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=5)
        # 返回运行时间的均值和标准差作为性能分析结果
        return np.mean(run_times), np.std(run_times)
    # 生成包含模型性能和指标详情的表格行的格式化字符串
    def generate_table_row(self, model_name, t_onnx, t_engine, model_info):
        """Generates a formatted string for a table row that includes model performance and metric details."""
        layers, params, gradients, flops = model_info
        return (
            f"| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.2f} ± {t_onnx[1]:.2f} ms | {t_engine[0]:.2f} ± "
            f"{t_engine[1]:.2f} ms | {params / 1e6:.1f} | {flops:.1f} |"
        )

    @staticmethod
    # 生成包含模型名称、参数、GFLOPS和速度指标的字典
    def generate_results_dict(model_name, t_onnx, t_engine, model_info):
        """Generates a dictionary of model details including name, parameters, GFLOPS and speed metrics."""
        layers, params, gradients, flops = model_info
        return {
            "model/name": model_name,
            "model/parameters": params,
            "model/GFLOPs": round(flops, 3),
            "model/speed_ONNX(ms)": round(t_onnx[0], 3),
            "model/speed_TensorRT(ms)": round(t_engine[0], 3),
        }

    @staticmethod
    # 格式化并打印包含不同模型统计和性能数据的比较表格
    def print_table(table_rows):
        """Formats and prints a comparison table for different models with given statistics and performance data."""
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        header = (
            f"| Model | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | "
            f"Speed<br><sup>{gpu} TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |"
        )
        separator = (
            "|-------------|---------------------|--------------------|------------------------------|"
            "-----------------------------------|------------------|-----------------|"
        )

        # 打印表格的标题和分隔线
        print(f"\n\n{header}")
        print(separator)
        # 打印每行表格内容
        for row in table_rows:
            print(row)
```