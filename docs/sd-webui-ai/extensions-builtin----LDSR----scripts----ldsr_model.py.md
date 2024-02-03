# `stable-diffusion-webui\extensions-builtin\LDSR\scripts\ldsr_model.py`

```py
# 导入操作系统模块
import os

# 从模块中导入函数
from modules.modelloader import load_file_from_url
from modules.upscaler import Upscaler, UpscalerData
from ldsr_model_arch import LDSR
from modules import shared, script_callbacks, errors
# 导入模块并忽略 F401 错误
import sd_hijack_autoencoder  # noqa: F401
import sd_hijack_ddpm_v1  # noqa: F401

# 定义一个名为 UpscalerLDSR 的类，继承自 Upscaler 类
class UpscalerLDSR(Upscaler):
    # 初始化方法，接受用户路径参数
    def __init__(self, user_path):
        # 设置类属性 name 为 "LDSR"
        self.name = "LDSR"
        # 设置类属性 user_path 为传入的用户路径参数
        self.user_path = user_path
        # 设置类属性 model_url 为模型下载链接
        self.model_url = "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"
        # 设置类属性 yaml_url 为 YAML 文件下载链接
        self.yaml_url = "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个名为 scaler_data 的 UpscalerData 对象，传入 "LDSR"、None 和 self 作为参数
        scaler_data = UpscalerData("LDSR", None, self)
        # 将 scaler_data 添加到 scalers 列表中
        self.scalers = [scaler_data]
    def load_model(self, path: str):
        # 拼接项目路径和 project.yaml 文件路径
        yaml_path = os.path.join(self.model_path, "project.yaml")
        # 拼接项目路径和 model.pth 文件路径
        old_model_path = os.path.join(self.model_path, "model.pth")
        # 拼接项目路径和 model.ckpt 文件路径
        new_model_path = os.path.join(self.model_path, "model.ckpt")

        # 查找项目路径下的模型文件，筛选出以 .ckpt 或 .safetensors 结尾的文件
        local_model_paths = self.find_models(ext_filter=[".ckpt", ".safetensors"])
        # 获取以 model.ckpt 结尾的本地模型路径
        local_ckpt_path = next(iter([local_model for local_model in local_model_paths if local_model.endswith("model.ckpt")]), None)
        # 获取以 model.safetensors 结尾的本地模型路径
        local_safetensors_path = next(iter([local_model for local_model in local_model_paths if local_model.endswith("model.safetensors")]), None)
        # 获取以 project.yaml 结尾的本地模型路径
        local_yaml_path = next(iter([local_model for local_model in local_model_paths if local_model.endswith("project.yaml")]), None)

        # 如果 project.yaml 文件存在且大小超过 10MB，则删除该文件
        if os.path.exists(yaml_path):
            statinfo = os.stat(yaml_path)
            if statinfo.st_size >= 10485760:
                print("Removing invalid LDSR YAML file.")
                os.remove(yaml_path)

        # 如果 model.pth 文件存在，则将其重命名为 model.ckpt
        if os.path.exists(old_model_path):
            print("Renaming model from model.pth to model.ckpt")
            os.rename(old_model_path, new_model_path)

        # 如果 local_safetensors_path 不为空且文件存在，则将 model 设为 local_safetensors_path
        if local_safetensors_path is not None and os.path.exists(local_safetensors_path):
            model = local_safetensors_path
        else:
            # 否则将 model 设为 local_ckpt_path 或从 model_url 下载 model.ckpt 文件
            model = local_ckpt_path or load_file_from_url(self.model_url, model_dir=self.model_download_path, file_name="model.ckpt")

        # 将 yaml 设为 local_yaml_path 或从 yaml_url 下载 project.yaml 文件
        yaml = local_yaml_path or load_file_from_url(self.yaml_url, model_dir=self.model_download_path, file_name="project.yaml")

        # 返回加载好的 LDSR 模型和项目配置文件
        return LDSR(model, yaml)

    def do_upscale(self, img, path):
        try:
            # 加载 LDSR 模型
            ldsr = self.load_model(path)
        except Exception:
            # 加载失败时报告错误信息并返回原始图片
            errors.report(f"Failed loading LDSR model {path}", exc_info=True)
            return img
        # 获取超分辨率处理的步骤数
        ddim_steps = shared.opts.ldsr_steps
        # 返回超分辨率处理后的图片
        return ldsr.super_resolution(img, ddim_steps, self.scale)
# 定义一个名为on_ui_settings的函数
def on_ui_settings():
    # 导入gradio库，用于创建用户界面
    import gradio as gr
    
    # 向共享的选项中添加一个名为"ldsr_steps"的选项，包括默认值、描述、控件类型、控件参数和所属部分
    shared.opts.add_option("ldsr_steps", shared.OptionInfo(100, "LDSR processing steps. Lower = faster", gr.Slider, {"minimum": 1, "maximum": 200, "step": 1}, section=('upscaling', "Upscaling")))
    
    # 向共享的选项中添加一个名为"ldsr_cached"的选项，包括默认值、描述、控件类型、控件参数和所属部分
    shared.opts.add_option("ldsr_cached", shared.OptionInfo(False, "Cache LDSR model in memory", gr.Checkbox, {"interactive": True}, section=('upscaling', "Upscaling")))

# 将on_ui_settings函数注册为script_callbacks的回调函数
script_callbacks.on_ui_settings(on_ui_settings)
```