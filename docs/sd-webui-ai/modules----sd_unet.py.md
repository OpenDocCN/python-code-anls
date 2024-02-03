# `stable-diffusion-webui\modules\sd_unet.py`

```
# 导入 torch.nn 模块
import torch.nn

# 从 modules 模块中导入 script_callbacks, shared, devices
from modules import script_callbacks, shared, devices

# 初始化空的 unet_options 列表
unet_options = []
# 初始化 current_unet_option 为 None
current_unet_option = None
# 初始化 current_unet 为 None
current_unet = None
# 初始化 original_forward 为 None，仅为兼容性暂时保留，未被使用

# 列出所有的 unet
def list_unets():
    # 调用 list_unets_callback 函数获取新的 unets
    new_unets = script_callbacks.list_unets_callback()

    # 清空 unet_options 列表
    unet_options.clear()
    # 将新的 unets 添加到 unet_options 列表中
    unet_options.extend(new_unets)

# 获取指定的 unet 选项
def get_unet_option(option=None):
    # 如果 option 为 None，则使用 shared.opts.sd_unet
    option = option or shared.opts.sd_unet

    # 如果 option 为 "None"，返回 None
    if option == "None":
        return None

    # 如果 option 为 "Automatic"
    if option == "Automatic":
        # 获取当前模型的名称
        name = shared.sd_model.sd_checkpoint_info.model_name

        # 从 unet_options 中筛选出模型名称与 name 相同的选项
        options = [x for x in unet_options if x.model_name == name]

        # 如果存在匹配的选项，则将 option 设置为匹配的选项的 label
        option = options[0].label if options else "None"

    # 返回 unet_options 中 label 与 option 相同的第一个选项，如果不存在则返回 None
    return next(iter([x for x in unet_options if x.label == option]), None)

# 应用指定的 unet 选项
def apply_unet(option=None):
    global current_unet_option
    global current_unet

    # 获取新的 unet 选项
    new_option = get_unet_option(option)
    # 如果新的选项与当前选项相同，则直接返回
    if new_option == current_unet_option:
        return

    # 如果当前 unet 不为 None
    if current_unet is not None:
        # 打印提示信息，关闭当前 unet
        print(f"Dectivating unet: {current_unet.option.label}")
        current_unet.deactivate()

    # 更新当前 unet 选项为新的选项
    current_unet_option = new_option
    # 如果当前 unet 选项为 None
    if current_unet_option is None:
        current_unet = None

        # 如果不是低内存模式，则将 shared.sd_model.model.diffusion_model 移动到设备上
        if not shared.sd_model.lowvram:
            shared.sd_model.model.diffusion_model.to(devices.device)

        return

    # 将 shared.sd_model.model.diffusion_model 移动到 CPU 上
    shared.sd_model.model.diffusion_model.to(devices.cpu)
    # 执行 torch 垃圾回收
    devices.torch_gc()

    # 创建当前 unet 对象，并激活
    current_unet = current_unet_option.create_unet()
    current_unet.option = current_unet_option
    print(f"Activating unet: {current_unet.option.label}")
    current_unet.activate()

# 定义 SdUnetOption 类
class SdUnetOption:
    model_name = None
    """相关检查点的名称 - 如果检查点的名称与此匹配，则此选项将自动选择为 unet"""

    label = None
    """UI 中 unet 的名称"""

    def create_unet(self):
        """返回用于制作图片时替代内置 unet 的 SdUnet 对象"""
        raise NotImplementedError()
# 定义一个名为 SdUnet 的类，继承自 torch.nn.Module
class SdUnet(torch.nn.Module):
    # 定义 forward 方法，接受输入 x、时间步长 timesteps、上下文 context，以及可变参数 *args 和关键字参数 **kwargs
    def forward(self, x, timesteps, context, *args, **kwargs):
        # 抛出 NotImplementedError 异常，提示子类需要实现该方法
        raise NotImplementedError()

    # 定义 activate 方法，空实现，子类可以覆盖该方法
    def activate(self):
        pass

    # 定义 deactivate 方法，空实现，子类可以覆盖该方法
    def deactivate(self):
        pass

# 定义一个函数 create_unet_forward，接受一个原始 forward 方法作为参数
def create_unet_forward(original_forward):
    # 定义一个内部函数 UNetModel_forward，接受 self、输入 x、时间步长 timesteps、上下文 context，以及可变参数 *args 和关键字参数 **kwargs
    def UNetModel_forward(self, x, timesteps=None, context=None, *args, **kwargs):
        # 如果当前的 current_unet 不为 None，则调用其 forward 方法，并传入相应参数
        if current_unet is not None:
            return current_unet.forward(x, timesteps, context, *args, **kwargs)

        # 否则调用原始的 forward 方法，并传入相应参数
        return original_forward(self, x, timesteps, context, *args, **kwargs)

    # 返回内部函数 UNetModel_forward
    return UNetModel_forward
```