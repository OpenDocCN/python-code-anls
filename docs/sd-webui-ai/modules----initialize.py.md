# `stable-diffusion-webui\modules\initialize.py`

```py
# 导入必要的模块
import importlib
import logging
import sys
import warnings
from threading import Thread

# 导入自定义模块
from modules.timer import startup_timer

# 定义一个函数用于处理导入
def imports():
    # 设置日志级别
    logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)  # sshh...
    # 过滤日志消息
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    # 导入 torch 模块
    import torch  # noqa: F401
    # 记录导入 torch 的时间
    startup_timer.record("import torch")
    # 导入 pytorch_lightning 模块
    import pytorch_lightning  # noqa: F401
    # 记录导入 torch 的时间
    startup_timer.record("import torch")
    # 忽略特定警告
    warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")

    # 导入 gradio 模块
    import gradio  # noqa: F401
    # 记录导入 gradio 的时间
    startup_timer.record("import gradio")

    # 导入自定义模块
    from modules import paths, timer, import_hook, errors  # noqa: F401
    # 记录设置路径的时间
    startup_timer.record("setup paths")

    # 导入 ldm 模块
    import ldm.modules.encoders.modules  # noqa: F401
    # 记录导入 ldm 的时间
    startup_timer.record("import ldm")

    # 导入 sgm 模块
    import sgm.modules.encoders.modules  # noqa: F401
    # 记录导入 sgm 的时间
    startup_timer.record("import sgm")

    # 初始化共享资源
    from modules import shared_init
    shared_init.initialize()
    # 记录初始化共享资源的时间
    startup_timer.record("initialize shared")

    # 导入其他模块
    from modules import processing, gradio_extensons, ui  # noqa: F401
    # 记录导入其他模块的时间
    startup_timer.record("other imports")

# 检查版本信息
def check_versions():
    # 导入命令行选项
    from modules.shared_cmd_options import cmd_opts

    # 如果不跳过版本检查
    if not cmd_opts.skip_version_check:
        # 导入错误处理模块
        from modules import errors
        errors.check_versions()

# 初始化函数
def initialize():
    # 导入初始化工具模块
    from modules import initialize_util
    # 修复 torch 版本
    initialize_util.fix_torch_version()
    # 修复 asyncio 事件循环策略
    initialize_util.fix_asyncio_event_loop_policy()
    # 验证 TLS 选项
    initialize_util.validate_tls_options()
    # 配置 SIGINT 处理程序
    initialize_util.configure_sigint_handler()
    # 配置选项更改
    initialize_util.configure_opts_onchange()

    # 导入模型加载器
    from modules import modelloader
    modelloader.cleanup_models()

    # 导入 SD 模型
    from modules import sd_models
    sd_models.setup_model()
    # 记录设置 SD 模型的时间
    startup_timer.record("setup SD model")
    # 从 modules.shared_cmd_options 模块中导入 cmd_opts 变量
    from modules.shared_cmd_options import cmd_opts
    
    # 从 modules 模块中导入 codeformer_model 模块
    from modules import codeformer_model
    
    # 忽略 torch 警告信息
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
    
    # 使用 cmd_opts.codeformer_models_path 设置 codeformer_model 模块
    codeformer_model.setup_model(cmd_opts.codeformer_models_path)
    
    # 记录代码转换模型设置的时间
    startup_timer.record("setup codeformer")
    
    # 从 modules 模块中导入 gfpgan_model 模块
    from modules import gfpgan_model
    
    # 使用 cmd_opts.gfpgan_models_path 设置 gfpgan_model 模块
    gfpgan_model.setup_model(cmd_opts.gfpgan_models_path)
    
    # 记录 GAN 模型设置的时间
    startup_timer.record("setup gfpgan")
    
    # 初始化其余部分，不重新加载脚本模块
    initialize_rest(reload_script_modules=False)
# 初始化 REST API，用于初始化和重新加载 WebUI
def initialize_rest(*, reload_script_modules=False):
    # 导入共享命令选项模块
    from modules.shared_cmd_options import cmd_opts

    # 导入 sd_samplers 模块并设置采样器
    from modules import sd_samplers
    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    # 导入 extensions 模块并列出扩展
    from modules import extensions
    extensions.list_extensions()
    startup_timer.record("list extensions")

    # 导入 initialize_util 模块并恢复配置状态文件
    from modules import initialize_util
    initialize_util.restore_config_state_file()
    startup_timer.record("restore config state file")

    # 导入 shared, upscaler, scripts 模块
    from modules import shared, upscaler, scripts
    # 如果处于 UI 调试模式，则设置 UpscalerLanczos 的缩放器并加载脚本
    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        scripts.load_scripts()
        return

    # 导入 sd_models 模块并列出模型
    from modules import sd_models
    sd_models.list_models()
    startup_timer.record("list SD models")

    # 导入 localization 模块并列出本地化
    from modules import localization
    localization.list_localizations(cmd_opts.localizations_dir)
    startup_timer.record("list localizations")

    # 加载脚本
    with startup_timer.subcategory("load scripts"):
        scripts.load_scripts()

    # 如果需要重新加载脚本模块
    if reload_script_modules:
        # 重新加载以 "modules.ui" 开头的模块
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

    # 导入 modelloader 模块并加载 Upscalers
    from modules import modelloader
    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    # 导入 sd_vae 模块并刷新 VAE 列表
    from modules import sd_vae
    sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")

    # 导入 textual_inversion 模块并列出文本反演模板
    from modules import textual_inversion
    textual_inversion.textual_inversion.list_textual_inversion_templates()
    startup_timer.record("refresh textual inversion templates")

    # 导入 script_callbacks, sd_hijack_optimizations, sd_hijack 模块
    script_callbacks.on_list_optimizers(sd_hijack_optimizations.list_optimizers)
    sd_hijack.list_optimizers()
    startup_timer.record("scripts list_optimizers")

    # 导入 sd_unet 模块
    from modules import sd_unet
    # 列出所有已注册的 UNet 模型
    sd_unet.list_unets()
    # 记录脚本执行时间
    startup_timer.record("scripts list_unets")

    # 定义加载模型的函数
    def load_model():
        """
        Accesses shared.sd_model property to load model.
        After it's available, if it has been loaded before this access by some extension,
        its optimization may be None because the list of optimizaers has neet been filled
        by that time, so we apply optimization again.
        """
        
        # 访问 shared.sd_model 属性以加载模型
        shared.sd_model  # noqa: B018

        # 如果当前优化器为 None，则应用优化
        if sd_hijack.current_optimizer is None:
            sd_hijack.apply_optimizations()

        # 导入 devices 模块并进行首次计算
        from modules import devices
        devices.first_time_calculation()
    
    # 如果不跳过在启动时加载模型，则启动一个线程加载模型
    if not shared.cmd_opts.skip_load_model_at_start:
        Thread(target=load_model).start()

    # 重新加载超网络
    from modules import shared_items
    shared_items.reload_hypernetworks()
    # 记录脚本执行时间
    startup_timer.record("reload hypernetworks")

    # 初始化 UI 额外网络
    from modules import ui_extra_networks
    ui_extra_networks.initialize()
    ui_extra_networks.register_default_pages()

    # 初始化额外网络
    from modules import extra_networks
    extra_networks.initialize()
    extra_networks.register_default_extra_networks()
    # 记录脚本执行时间
    startup_timer.record("initialize extra networks")
```