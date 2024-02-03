# `stable-diffusion-webui\modules\shared_total_tqdm.py`

```py
# 导入 tqdm 库，用于显示进度条
import tqdm

# 从 modules 模块中导入 shared 对象
from modules import shared

# 定义 TotalTQDM 类
class TotalTQDM:
    # 初始化方法
    def __init__(self):
        # 初始化 _tqdm 属性为 None
        self._tqdm = None

    # 重置进度条
    def reset(self):
        # 创建 tqdm 进度条对象，设置描述、总数、位置和输出文件
        self._tqdm = tqdm.tqdm(
            desc="Total progress",
            total=shared.state.job_count * shared.state.sampling_steps,
            position=1,
            file=shared.progress_print_out
        )

    # 更新进度条
    def update(self):
        # 如果不需要显示多个进度条或禁用控制台进度条，则直接返回
        if not shared.opts.multiple_tqdm or shared.cmd_opts.disable_console_progressbars:
            return
        # 如果进度条对象为空，则重置进度条
        if self._tqdm is None:
            self.reset()
        # 更新进度条
        self._tqdm.update()

    # 更新总数
    def updateTotal(self, new_total):
        # 如果不需要显示多个进度条或禁用控制台进度条，则直接返回
        if not shared.opts.multiple_tqdm or shared.cmd_opts.disable_console_progressbars:
            return
        # 如果进度条对象为空，则重置进度条
        if self._tqdm is None:
            self.reset()
        # 更新进度条的总数
        self._tqdm.total = new_total

    # 清除进度条
    def clear(self):
        # 如果进度条对象不为空，则刷新并关闭进度条，并将进度条对象置为 None
        if self._tqdm is not None:
            self._tqdm.refresh()
            self._tqdm.close()
            self._tqdm = None
```