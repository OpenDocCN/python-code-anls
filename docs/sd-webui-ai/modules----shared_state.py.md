# `stable-diffusion-webui\modules\shared_state.py`

```py
# 导入必要的模块
import datetime
import logging
import threading
import time

# 从模块中导入特定的类或函数
from modules import errors, shared, devices
from typing import Optional

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 定义状态类
class State:
    # 初始化状态属性
    skipped = False
    interrupted = False
    job = ""
    job_no = 0
    job_count = 0
    processing_has_refined_job_count = False
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    id_live_preview = 0
    textinfo = None
    time_start = None
    server_start = None
    _server_command_signal = threading.Event()
    _server_command: Optional[str] = None

    # 初始化方法
    def __init__(self):
        # 记录服务器启动时间
        self.server_start = time.time()

    # 获取是否需要重新启动服务器的属性
    @property
    def need_restart(self) -> bool:
        # 兼容性获取器，返回是否需要重新启动服务器
        return self.server_command == "restart"

    # 设置是否需要重新启动服务器的属性
    @need_restart.setter
    def need_restart(self, value: bool) -> None:
        # 兼容性设置器，设置是否需要重新启动服务器
        if value:
            self.server_command = "restart"

    # 获取服务器命令的属性
    @property
    def server_command(self):
        return self._server_command

    # 设置服务器命令的属性
    @server_command.setter
    def server_command(self, value: Optional[str]) -> None:
        """
        Set the server command to `value` and signal that it's been set.
        """
        # 设置服务器命令，并发出信号
        self._server_command = value
        self._server_command_signal.set()

    # 等待服务器命令被设置
    def wait_for_server_command(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Wait for server command to get set; return and clear the value and signal.
        """
        # 等待服务器命令被设置，返回并清除值和信号
        if self._server_command_signal.wait(timeout):
            self._server_command_signal.clear()
            req = self._server_command
            self._server_command = None
            return req
        return None

    # 请求重新启动服务器
    def request_restart(self) -> None:
        # 中断当前操作，设置服务器命令为重新启动，并记录日志
        self.interrupt()
        self.server_command = "restart"
        log.info("Received restart request")
    # 标记当前任务为跳过状态
    def skip(self):
        self.skipped = True
        # 记录日志，显示收到跳过请求
        log.info("Received skip request")

    # 标记当前任务为中断状态
    def interrupt(self):
        self.interrupted = True
        # 记录日志，显示收到中断请求
        log.info("Received interrupt request")

    # 进入下一个任务
    def nextjob(self):
        # 如果启用了实时预览并且显示进度步数为-1，则设置当前图像
        if shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps == -1:
            self.do_set_current_image()

        # 增加任务编号
        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0

    # 返回包含任务信息的字典
    def dict(self):
        obj = {
            "skipped": self.skipped,
            "interrupted": self.interrupted,
            "job": self.job,
            "job_count": self.job_count,
            "job_timestamp": self.job_timestamp,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
        }

        return obj

    # 开始执行任务
    def begin(self, job: str = "(unknown)"):
        self.sampling_step = 0
        self.time_start = time.time()
        self.job_count = -1
        self.processing_has_refined_job_count = False
        self.job_no = 0
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.current_latent = None
        self.current_image = None
        self.current_image_sampling_step = 0
        self.id_live_preview = 0
        self.skipped = False
        self.interrupted = False
        self.textinfo = None
        self.job = job
        devices.torch_gc()
        # 记录日志，显示开始执行任务
        log.info("Starting job %s", job)

    # 结束当前任务
    def end(self):
        duration = time.time() - self.time_start
        # 记录日志，显示结束任务及执行时间
        log.info("Ending job %s (%.2f seconds)", self.job, duration)
        self.job = ""
        self.job_count = 0

        devices.torch_gc()
    # 设置当前图像，如果距离上次调用足够的采样步骤已经完成，则从当前潜在空间设置 self.current_image，并相应修改 self.id_live_preview
    def set_current_image(self):
        # 如果不允许并行处理，则直接返回
        if not shared.parallel_processing_allowed:
            return

        # 如果当前采样步骤减去上次设置当前图像的采样步骤大于等于显示进度的步骤数，并且启用了实时预览，并且显示进度的步骤数不为-1
        if self.sampling_step - self.current_image_sampling_step >= shared.opts.show_progress_every_n_steps and shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps != -1:
            # 调用 do_set_current_image 方法
            self.do_set_current_image()

    # 执行设置当前图像的操作
    def do_set_current_image(self):
        # 如果当前潜在空间为空，则直接返回
        if self.current_latent is None:
            return

        # 导入 sd_samplers 模块
        import modules.sd_samplers

        try:
            # 如果显示进度的网格为真，则将当前潜在空间样本转换为图像网格
            if shared.opts.show_progress_grid:
                self.assign_current_image(modules.sd_samplers.samples_to_image_grid(self.current_latent))
            else:
                # 否则将当前潜在空间样本转换为图像
                self.assign_current_image(modules.sd_samplers.sample_to_image(self.current_latent))

            # 更新当前图像的采样步骤为当前采样步骤
            self.current_image_sampling_step = self.sampling_step

        except Exception:
            # 在生成过程中切换模型时，VAE 可能在 CPU 上，因此创建图像会失败
            # 我们静默地忽略这个错误
            errors.record_exception()

    # 分配当前图像
    def assign_current_image(self, image):
        # 将图像分配给当前图像
        self.current_image = image
        # 增加实时预览的 ID
        self.id_live_preview += 1
```