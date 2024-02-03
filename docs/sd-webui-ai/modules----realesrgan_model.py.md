# `stable-diffusion-webui\modules\realesrgan_model.py`

```
# 导入所需的模块
import os
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer

# 导入自定义模块
from modules.upscaler import Upscaler, UpscalerData
from modules.shared import cmd_opts, opts
from modules import modelloader, errors

# 定义 RealESRGAN 类，继承自 Upscaler 类
class UpscalerRealESRGAN(Upscaler):
    # 初始化方法
    def __init__(self, path):
        # 设置名称为 RealESRGAN
        self.name = "RealESRGAN"
        # 设置用户路径
        self.user_path = path
        # 调用父类的初始化方法
        super().__init__()
        try:
            # 导入所需的模块，忽略 F401 错误
            from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
            from realesrgan import RealESRGANer  # noqa: F401
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # noqa: F401
            # 启用 RealESRGAN
            self.enable = True
            # 初始化 scalers 列表
            self.scalers = []
            # 加载模型
            scalers = self.load_models(path)

            # 查找本地模型路径
            local_model_paths = self.find_models(ext_filter=[".pth"])
            # 遍历 scalers 列表
            for scaler in scalers:
                # 如果本地数据路径以 "http" 开头
                if scaler.local_data_path.startswith("http"):
                    # 获取友好的文件名
                    filename = modelloader.friendly_name(scaler.local_data_path)
                    # 在本地模型路径中查找匹配的文件
                    local_model_candidates = [local_model for local_model in local_model_paths if local_model.endswith(f"{filename}.pth")]
                    # 如果找到匹配的本地模型
                    if local_model_candidates:
                        # 更新本地数据路径为匹配的本地模型路径
                        scaler.local_data_path = local_model_candidates[0]

                # 如果 scaler 的名称在 realesrgan_enabled_models 中
                if scaler.name in opts.realesrgan_enabled_models:
                    # 将 scaler 添加到 scalers 列表中
                    self.scalers.append(scaler)

        # 捕获异常
        except Exception:
            # 报告错误信息
            errors.report("Error importing Real-ESRGAN", exc_info=True)
            # 禁用 RealESRGAN
            self.enable = False
            # 清空 scalers 列表
            self.scalers = []
    # 对图像进行放大处理，如果未启用放大功能，则直接返回原图像
    def do_upscale(self, img, path):
        # 如果未启用放大功能，则直接返回原图像
        if not self.enable:
            return img

        try:
            # 加载指定路径下的模型信息
            info = self.load_model(path)
        except Exception:
            # 报告加载 RealESRGAN 模型失败的错误信息
            errors.report(f"Unable to load RealESRGAN model {path}", exc_info=True)
            return img

        # 创建 RealESRGANer 实例，用于图像放大处理
        upsampler = RealESRGANer(
            scale=info.scale,
            model_path=info.local_data_path,
            model=info.model(),
            half=not cmd_opts.no_half and not cmd_opts.upcast_sampling,
            tile=opts.ESRGAN_tile,
            tile_pad=opts.ESRGAN_tile_overlap,
            device=self.device,
        )

        # 对图像进行放大处理
        upsampled = upsampler.enhance(np.array(img), outscale=info.scale)[0]

        # 将放大后的图像转换为 Image 对象并返回
        image = Image.fromarray(upsampled)
        return image

    # 加载指定路径下的模型信息
    def load_model(self, path):
        # 遍历已加载的模型信息列表
        for scaler in self.scalers:
            # 如果找到指定路径的模型信息
            if scaler.data_path == path:
                # 如果模型数据路径以 "http" 开头，则从 URL 加载模型文件到本地
                if scaler.local_data_path.startswith("http"):
                    scaler.local_data_path = modelloader.load_file_from_url(
                        scaler.data_path,
                        model_dir=self.model_download_path,
                    )
                # 如果本地模型文件不存在，则抛出文件未找到异常
                if not os.path.exists(scaler.local_data_path):
                    raise FileNotFoundError(f"RealESRGAN data missing: {scaler.local_data_path}")
                return scaler
        # 如果未找到指定路径的模型信息，则抛出数值错误异常
        raise ValueError(f"Unable to find model info: {path}")

    # 加载 RealESRGAN 模型
    def load_models(self, _):
        return get_realesrgan_models(self)
# 定义一个函数，用于获取 Real-ESRGAN 模型列表，参数为缩放器
def get_realesrgan_models(scaler):
    # 捕获所有异常，并报告错误信息，包括异常信息
    except Exception:
        errors.report("Error making Real-ESRGAN models list", exc_info=True)
```