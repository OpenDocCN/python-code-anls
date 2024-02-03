# `stable-diffusion-webui\modules\shared_init.py`

```
# 导入 os 模块
import os

# 导入 torch 模块
import torch

# 从 modules 模块中导入 shared
from modules import shared
# 从 modules.shared 模块中导入 cmd_opts
from modules.shared import cmd_opts

# 初始化函数，用于在受控的方式下初始化 shared 模块内的字段
def initialize():
    """Initializes fields inside the shared module in a controlled manner.

    Should be called early because some other modules you can import mingt need these fields to be already set.
    """

    # 如果不存在指定目录，则创建目录
    os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)

    # 从 modules 模块中导入 options 和 shared_options
    from modules import options, shared_options
    # 设置 shared 模块中的 options_templates 字段为 shared_options 模块中的 options_templates
    shared.options_templates = shared_options.options_templates
    # 初始化 shared 模块中的 opts 字段
    shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
    # 设置 shared 模块中的 restricted_opts 字段为 shared_options 模块中的 restricted_opts
    shared.restricted_opts = shared_options.restricted_opts
    # 如果存在指定的配置文件，则加载配置文件
    if os.path.exists(shared.config_filename):
        shared.opts.load(shared.config_filename)

    # 从 modules 模块中导入 devices
    from modules import devices
    # 根据条件设置 devices 模块中的不同设备
    devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
        (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

    # 根据条件设置数据类型
    devices.dtype = torch.float32 if cmd_opts.no_half else torch.float16
    devices.dtype_vae = torch.float32 if cmd_opts.no_half or cmd_opts.no_half_vae else torch.float16

    # 设置 shared 模块中的 device 字段为 devices 模块中的 device
    shared.device = devices.device
    # 根据条件设置权重加载位置
    shared.weight_load_location = None if cmd_opts.lowram else "cpu"

    # 从 modules 模块中导入 shared_state
    from modules import shared_state
    # 初始化 shared 模块中的 state 字段
    shared.state = shared_state.State()

    # 从 modules 模块中导入 styles
    from modules import styles
    # 初始化 shared 模块中的 prompt_styles 字段
    shared.prompt_styles = styles.StyleDatabase(shared.styles_filename)

    # 从 modules 模块中导入 interrogate
    from modules import interrogate
    # 初始化 shared 模块中的 interrogator 字段
    shared.interrogator = interrogate.InterrogateModels("interrogate")

    # 从 modules 模块中导入 shared_total_tqdm
    from modules import shared_total_tqdm
    # 初始化 shared 模块中的 total_tqdm 字段
    shared.total_tqdm = shared_total_tqdm.TotalTQDM()

    # 从 modules 模块中导入 memmon 和 devices
    from modules import memmon, devices
    # 初始化 shared 模块中的 mem_mon 字段
    shared.mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, shared.opts)
    # 启动内存监控
    shared.mem_mon.start()
```