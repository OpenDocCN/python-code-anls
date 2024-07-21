# `.\pytorch\benchmarks\inference\server.py`

```
import argparse
import asyncio
import os.path
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

import numpy as np
import pandas as pd

import torch
import torch.multiprocessing as mp


class FrontendWorker(mp.Process):
    """
    This worker will send requests to a backend process, and measure the
    throughput and latency of those requests as well as GPU utilization.
    """

    def __init__(
        self,
        metrics_dict,
        request_queue,
        response_queue,
        read_requests_event,
        batch_size,
        num_iters=10,
    ):
        super().__init__()
        self.metrics_dict = metrics_dict  # 初始化实例变量 metrics_dict，用于存储性能指标
        self.request_queue = request_queue  # 初始化实例变量 request_queue，用于存储请求队列
        self.response_queue = response_queue  # 初始化实例变量 response_queue，用于存储响应队列
        self.read_requests_event = read_requests_event  # 初始化实例变量 read_requests_event，用于读取请求事件
        self.warmup_event = mp.Event()  # 初始化实例变量 warmup_event，用于记录热身完成事件
        self.batch_size = batch_size  # 初始化实例变量 batch_size，指定每批次请求的数量
        self.num_iters = num_iters  # 初始化实例变量 num_iters，指定迭代次数，默认为10
        self.poll_gpu = True  # 初始化实例变量 poll_gpu，用于控制是否轮询 GPU 利用率
        self.start_send_time = None  # 初始化实例变量 start_send_time，用于记录发送请求的开始时间
        self.end_recv_time = None  # 初始化实例变量 end_recv_time，用于记录接收响应的结束时间

    def _run_metrics(self, metrics_lock):
        """
        This function will poll the response queue until it has received all
        responses. It records the startup latency, the average, max, min latency
        as well as througput of requests.
        """
        warmup_response_time = None  # 初始化变量 warmup_response_time，用于记录热身响应时间
        response_times = []  # 初始化列表 response_times，用于存储所有响应时间

        for i in range(self.num_iters + 1):
            response, request_time = self.response_queue.get()  # 从响应队列中获取响应和请求时间
            if warmup_response_time is None:
                self.warmup_event.set()  # 设置热身完成事件
                warmup_response_time = time.time() - request_time  # 计算热身响应时间
            else:
                response_times.append(time.time() - request_time)  # 记录每个响应的响应时间

        self.end_recv_time = time.time()  # 记录接收最后一个响应的时间
        self.poll_gpu = False  # 设置不再轮询 GPU 利用率

        response_times = np.array(response_times)  # 转换响应时间列表为 NumPy 数组
        with metrics_lock:
            self.metrics_dict["warmup_latency"] = warmup_response_time  # 记录热身响应时间到性能指标字典
            self.metrics_dict["average_latency"] = response_times.mean()  # 计算平均响应时间并记录到性能指标字典
            self.metrics_dict["max_latency"] = response_times.max()  # 计算最大响应时间并记录到性能指标字典
            self.metrics_dict["min_latency"] = response_times.min()  # 计算最小响应时间并记录到性能指标字典
            self.metrics_dict["throughput"] = (self.num_iters * self.batch_size) / (
                self.end_recv_time - self.start_send_time
            )  # 计算吞吐量并记录到性能指标字典
    def _run_gpu_utilization(self, metrics_lock):
        """
        This function will poll nvidia-smi for GPU utilization every 100ms to
        record the average GPU utilization.
        """

        def get_gpu_utilization():
            try:
                # 使用 subprocess 调用 nvidia-smi 命令，查询 GPU 利用率
                nvidia_smi_output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--id=0",
                        "--format=csv,noheader,nounits",
                    ]
                )
                # 解码输出并去除首尾空白字符，获取 GPU 利用率
                gpu_utilization = nvidia_smi_output.decode().strip()
                return gpu_utilization
            except subprocess.CalledProcessError:
                # 如果调用出错，返回 "N/A"
                return "N/A"

        gpu_utilizations = []

        while self.poll_gpu:
            # 获取当前 GPU 利用率
            gpu_utilization = get_gpu_utilization()
            if gpu_utilization != "N/A":
                # 将 GPU 利用率转换为浮点数并添加到列表中
                gpu_utilizations.append(float(gpu_utilization))

        with metrics_lock:
            # 计算 GPU 利用率的平均值，并存储到 metrics_dict 中
            self.metrics_dict["gpu_util"] = torch.tensor(gpu_utilizations).mean().item()

    def _send_requests(self):
        """
        This function will send one warmup request, and then num_iters requests
        to the backend process.
        """

        fake_data = torch.randn(self.batch_size, 3, 250, 250, requires_grad=False)
        other_data = [
            torch.randn(self.batch_size, 3, 250, 250, requires_grad=False)
            for i in range(self.num_iters)
        ]

        # 发送一批预热数据
        self.request_queue.put((fake_data, time.time()))
        # 通知后端进程轮询队列以获取预热请求
        self.read_requests_event.set()
        self.warmup_event.wait()
        # 通知后端进程轮询队列以获取剩余请求
        self.read_requests_event.set()

        # 发送假数据
        self.start_send_time = time.time()
        for i in range(self.num_iters):
            self.request_queue.put((other_data[i], time.time()))

    def run(self):
        # 用于写入 metrics_dict 的锁
        metrics_lock = threading.Lock()
        requests_thread = threading.Thread(target=self._send_requests)
        metrics_thread = threading.Thread(
            target=self._run_metrics, args=(metrics_lock,)
        )
        gpu_utilization_thread = threading.Thread(
            target=self._run_gpu_utilization, args=(metrics_lock,)
        )

        requests_thread.start()
        metrics_thread.start()

        # 只有在预热请求完成后才开始轮询 GPU 利用率
        self.warmup_event.wait()
        gpu_utilization_thread.start()

        requests_thread.join()
        metrics_thread.join()
        gpu_utilization_thread.join()
# 定义一个名为 BackendWorker 的类，用于处理请求队列中的张量数据，
# 执行计算，并将结果返回到响应队列中。

class BackendWorker:
    """
    This worker will take tensors from the request queue, do some computation,
    and then return the result back in the response queue.
    """

    def __init__(
        self,
        metrics_dict,          # 用于存储各种度量指标的字典
        request_queue,         # 请求张量的队列
        response_queue,        # 存放计算结果的队列
        read_requests_event,   # 事件，表示可以读取请求的状态
        batch_size,            # 每个批次处理的张量数量
        num_workers,           # 后端工作线程的数量
        model_dir=".",         # 模型文件的目录，默认为当前目录
        compile_model=True,    # 是否编译模型，默认为True
    ):
        super().__init__()
        self.device = "cuda:0"  # 使用 CUDA 设备，设备编号为0
        self.metrics_dict = metrics_dict  # 初始化度量指标字典
        self.request_queue = request_queue  # 初始化请求队列
        self.response_queue = response_queue  # 初始化响应队列
        self.read_requests_event = read_requests_event  # 初始化读取请求的事件
        self.batch_size = batch_size  # 初始化批处理大小
        self.num_workers = num_workers  # 初始化工作线程数量
        self.model_dir = model_dir  # 初始化模型文件目录
        self.compile_model = compile_model  # 初始化编译模型标志
        self._setup_complete = False  # 设置初始化未完成的标志为False
        self.h2d_stream = torch.cuda.Stream()  # 创建用于主机到设备内存传输的 CUDA 流对象
        self.d2h_stream = torch.cuda.Stream()  # 创建用于设备到主机内存传输的 CUDA 流对象
        # 将线程ID映射到与该工作线程关联的 CUDA 流对象的字典
        self.stream_map = dict()

    def _setup(self):
        import time

        from torchvision.models.resnet import BasicBlock, ResNet  # 导入 torchvision 库中的模型

        import torch  # 导入 PyTorch 库

        # 在 meta 设备上创建 ResNet18 模型
        with torch.device("meta"):
            m = ResNet(BasicBlock, [2, 2, 2, 2])  # 创建 ResNet18 模型对象

        # 加载预训练权重
        start_load_time = time.time()
        state_dict = torch.load(
            f"{self.model_dir}/resnet18-f37072fd.pth",  # 加载预训练模型的文件路径
            mmap=True,          # 启用内存映射模式
            map_location=self.device,  # 将加载的模型映射到指定的 CUDA 设备
        )
        self.metrics_dict["torch_load_time"] = time.time() - start_load_time  # 记录加载模型的时间
        m.load_state_dict(state_dict, assign=True)  # 加载模型的状态字典
        m.eval()  # 设置模型为评估模式

        if self.compile_model:
            start_compile_time = time.time()
            m.compile()  # 如果需要，编译模型
            end_compile_time = time.time()
            self.metrics_dict["m_compile_time"] = end_compile_time - start_compile_time  # 记录编译模型的时间
        return m  # 返回配置好的模型对象

    def model_predict(
        self,
        model,                 # 需要进行预测的模型对象
        input_buffer,          # 输入数据缓冲区
        copy_event,            # 复制数据事件
        compute_event,         # 计算数据事件
        copy_sem,              # 复制信号量，确保数据复制完成
        compute_sem,           # 计算信号量，表示计算完成
        response_list,         # 存放预测结果的列表
        request_time,          # 请求时间戳
    ):
        # 等待复制信号量，确保数据已经被复制线程记录
        copy_sem.acquire()
        self.stream_map[threading.get_native_id()].wait_event(copy_event)
        with torch.cuda.stream(self.stream_map[threading.get_native_id()]):  # 使用对应线程的 CUDA 流
            with torch.no_grad():
                response_list.append(model(input_buffer))  # 执行模型推理，并将结果添加到响应列表中
                compute_event.record()  # 记录计算事件
                compute_sem.release()  # 发布计算信号量
        del input_buffer  # 删除输入缓冲区的引用，释放内存

    def copy_data(self, input_buffer, data, copy_event, copy_sem):
        data = data.pin_memory()  # 将数据锁定在内存中，以便进行 DMA 数据传输
        with torch.cuda.stream(self.h2d_stream):  # 使用主机到设备内存传输的 CUDA 流
            input_buffer.copy_(data, non_blocking=True)  # 非阻塞地将数据从主机内存复制到设备内存
            copy_event.record()  # 记录复制事件
            copy_sem.release()   # 发布复制信号量
        # 响应处理函数，用于处理模型预测的结果并发送响应
        def respond(self, compute_event, compute_sem, response_list, request_time):
            # 等待 compute_event 被记录到 model_predict 线程中
            compute_sem.acquire()
            # 等待 d2h_stream 中的 compute_event 事件完成
            self.d2h_stream.wait_event(compute_event)
            # 使用 d2h_stream 将 response_list 中的第一个结果发送回主机
            with torch.cuda.stream(self.d2h_stream):
                self.response_queue.put((response_list[0].cpu(), request_time))

        async def run(self):
            # 初始化工作线程的 CUDA 流映射
            def worker_initializer():
                self.stream_map[threading.get_native_id()] = torch.cuda.Stream()

            # 创建工作线程池和主机到设备的数据拷贝线程池
            worker_pool = ThreadPoolExecutor(
                max_workers=self.num_workers, initializer=worker_initializer
            )
            h2d_pool = ThreadPoolExecutor(max_workers=1)
            d2h_pool = ThreadPoolExecutor(max_workers=1)

            # 等待 read_requests_event 信号
            self.read_requests_event.wait()
            # 在继续轮询 request_queue 之前，清除 read_requests_event
            # 因为我们将再次等待该事件，然后处理非预热请求
            self.read_requests_event.clear()
            while True:
                try:
                    # 从 request_queue 中获取数据和请求时间
                    data, request_time = self.request_queue.get(timeout=5)
                except Empty:
                    break

                # 如果设置尚未完成，则执行模型设置
                if not self._setup_complete:
                    model = self._setup()

                # 创建信号量和 CUDA 事件，用于数据拷贝和模型计算的同步
                copy_sem = threading.Semaphore(0)
                compute_sem = threading.Semaphore(0)
                copy_event = torch.cuda.Event()
                compute_event = torch.cuda.Event()
                response_list = []

                # 在 h2d_pool 中异步执行数据拷贝操作
                asyncio.get_running_loop().run_in_executor(
                    h2d_pool,
                    self.copy_data,
                    input_buffer,
                    data,
                    copy_event,
                    copy_sem,
                )

                # 在 worker_pool 中异步执行模型预测操作
                asyncio.get_running_loop().run_in_executor(
                    worker_pool,
                    self.model_predict,
                    model,
                    input_buffer,
                    copy_event,
                    compute_event,
                    copy_sem,
                    compute_sem,
                    response_list,
                    request_time,
                )

                # 在 d2h_pool 中异步执行响应处理操作
                asyncio.get_running_loop().run_in_executor(
                    d2h_pool,
                    self.respond,
                    compute_event,
                    compute_sem,
                    response_list,
                    request_time,
                )

                # 如果设置尚未完成，则等待 read_requests_event 信号
                if not self._setup_complete:
                    self.read_requests_event.wait()
                    self._setup_complete = True
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数：迭代次数，默认为100
    parser.add_argument("--num_iters", type=int, default=100)
    # 添加命令行参数：批量大小，默认为32
    parser.add_argument("--batch_size", type=int, default=32)
    # 添加命令行参数：模型目录，默认为当前目录
    parser.add_argument("--model_dir", type=str, default=".")
    # 添加命令行参数：是否编译，默认为True
    parser.add_argument(
        "--compile", default=True, action=argparse.BooleanOptionalAction
    )
    # 添加命令行参数：输出文件名，默认为output.csv
    parser.add_argument("--output_file", type=str, default="output.csv")
    # 添加命令行参数：是否进行性能分析，默认为False
    parser.add_argument(
        "--profile", default=False, action=argparse.BooleanOptionalAction
    )
    # 添加命令行参数：工作进程数，默认为4
    parser.add_argument("--num_workers", type=int, default=4)
    # 解析命令行参数
    args = parser.parse_args()

    # 检查是否已经下载了模型检查点文件
    downloaded_checkpoint = False
    if not os.path.isfile(f"{args.model_dir}/resnet18-f37072fd.pth"):
        # 使用 subprocess 运行 wget 下载模型检查点文件
        p = subprocess.run(
            [
                "wget",
                "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            ]
        )
        # 检查下载是否成功
        if p.returncode == 0:
            downloaded_checkpoint = True
        else:
            # 下载失败则抛出运行时错误
            raise RuntimeError("Failed to download checkpoint")

    try:
        # 设置 multiprocessing 的启动方式为 "forkserver"
        mp.set_start_method("forkserver")
        # 创建请求和响应队列，以及读取请求的事件
        request_queue = mp.Queue()
        response_queue = mp.Queue()
        read_requests_event = mp.Event()

        # 使用 multiprocessing.Manager 创建共享的 metrics_dict
        manager = mp.Manager()
        metrics_dict = manager.dict()
        metrics_dict["batch_size"] = args.batch_size  # 设置批量大小
        metrics_dict["compile"] = args.compile  # 设置是否编译

        # 创建 FrontendWorker 实例，处理前端任务
        frontend = FrontendWorker(
            metrics_dict,
            request_queue,
            response_queue,
            read_requests_event,
            args.batch_size,
            num_iters=args.num_iters,
        )
        # 创建 BackendWorker 实例，处理后端任务
        backend = BackendWorker(
            metrics_dict,
            request_queue,
            response_queue,
            read_requests_event,
            args.batch_size,
            args.num_workers,
            args.model_dir,
            args.compile,
        )

        # 启动前端任务处理
        frontend.start()

        # 如果启用了性能分析
        if args.profile:

            def trace_handler(prof):
                prof.export_chrome_trace("trace.json")

            # 使用 torch.profiler.profile 进行性能分析
            with torch.profiler.profile(on_trace_ready=trace_handler) as prof:
                asyncio.run(backend.run())
        else:
            # 否则直接运行后端任务
            asyncio.run(backend.run())

        # 等待前端任务处理完毕
        frontend.join()

        # 将 metrics_dict 转换为标准的字典格式
        metrics_dict = {k: [v] for k, v in metrics_dict._getvalue().items()}
        # 创建 DataFrame 对象
        output = pd.DataFrame.from_dict(metrics_dict, orient="columns")
        # 设置输出文件路径
        output_file = "./results/" + args.output_file
        # 检查输出文件是否为空
        is_empty = not os.path.isfile(output_file)

        # 将 DataFrame 输出到 CSV 文件中
        with open(output_file, "a+", newline="") as file:
            output.to_csv(file, header=is_empty, index=False)

    finally:
        # 最终清理：如果下载了模型检查点文件，则删除之
        if downloaded_checkpoint:
            os.remove(f"{args.model_dir}/resnet18-f37072fd.pth")
```