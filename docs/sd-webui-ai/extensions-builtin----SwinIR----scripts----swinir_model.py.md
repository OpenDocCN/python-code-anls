# `stable-diffusion-webui\extensions-builtin\SwinIR\scripts\swinir_model.py`

```
# 导入系统、平台、numpy、torch、PIL、tqdm等模块
import sys
import platform
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 导入自定义模块：modelloader、devices、script_callbacks、shared、opts、state
from modules import modelloader, devices, script_callbacks, shared
from modules.shared import opts, state
# 导入SwinIR模型架构：SwinIR、Swin2SR
from swinir_model_arch import SwinIR
from swinir_model_arch_v2 import Swin2SR
# 导入Upscaler、UpscalerData类
from modules.upscaler import Upscaler, UpscalerData

# SwinIR模型的下载链接
SWINIR_MODEL_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"

# 获取SwinIR设备
device_swinir = devices.get_device_for('swinir')

# 定义UpscalerSwinIR类，继承自Upscaler类
class UpscalerSwinIR(Upscaler):
    def __init__(self, dirname):
        # 保持模型在SWIN_torch_compile开启时，防止每次运行都重新编译
        self._cached_model = None
        # 在更改模型（v1/v2）或设置时清除'_cached_model'
        self._cached_model_config = None
        # 定义模型名称、模型下载链接、模型名称、用户路径
        self.name = "SwinIR"
        self.model_url = SWINIR_MODEL_URL
        self.model_name = "SwinIR 4x"
        self.user_path = dirname
        # 调用父类的初始化方法
        super().__init__()
        # 初始化scalers列表
        scalers = []
        # 查找模型文件，筛选后缀为".pt"、".pth"的文件
        model_files = self.find_models(ext_filter=[".pt", ".pth"])
        # 遍历模型文件列表
        for model in model_files:
            # 如果模型以"http"开头，则使用模型名称作为名称
            if model.startswith("http"):
                name = self.model_name
            else:
                # 否则使用modelloader的friendly_name方法获取友好名称
                name = modelloader.friendly_name(model)
            # 创建UpscalerData对象，包含模型名称、模型文件、当前对象
            model_data = UpscalerData(name, model, self)
            # 将model_data添加到scalers列表中
            scalers.append(model_data)
        # 将scalers赋值给对象的scalers属性
        self.scalers = scalers
    # 对图像进行放大处理
    def do_upscale(self, img, model_file):
        # 检查是否需要编译模型，条件包括是否存在 opts.SWIN_torch_compile 属性且为真值、torch 版本大于等于 2、不在 Windows 系统上
        use_compile = hasattr(opts, 'SWIN_torch_compile') and opts.SWIN_torch_compile \
            and int(torch.__version__.split('.')[0]) >= 2 and platform.system() != "Windows"
        # 记录当前模型配置信息
        current_config = (model_file, opts.SWIN_tile)

        # 如果需要编译且当前模型配置与缓存的模型配置相同，则使用缓存的模型
        if use_compile and self._cached_model_config == current_config:
            model = self._cached_model
        else:
            # 否则清空缓存的模型
            self._cached_model = None
            try:
                # 尝试加载模型
                model = self.load_model(model_file)
            except Exception as e:
                # 加载模型失败时输出错误信息并返回原图像
                print(f"Failed loading SwinIR model {model_file}: {e}", file=sys.stderr)
                return img
            # 将模型移动到指定设备上
            model = model.to(device_swinir, dtype=devices.dtype)
            # 如果需要编译模型，则进行编译并缓存模型及配置信息
            if use_compile:
                model = torch.compile(model)
                self._cached_model = model
                self._cached_model_config = current_config
        # 使用模型对图像进行放大处理
        img = upscale(img, model)
        # 执行 torch 垃圾回收
        devices.torch_gc()
        # 返回处理后的图像
        return img
    # 加载模型，根据路径和缩放比例
    def load_model(self, path, scale=4):
        # 如果路径以"http"开头，则从URL加载模型文件
        if path.startswith("http"):
            filename = modelloader.load_file_from_url(
                url=path,
                model_dir=self.model_download_path,
                file_name=f"{self.model_name.replace(' ', '_')}.pth",
            )
        else:
            # 否则直接使用给定路径作为文件名
            filename = path
        # 如果文件名以".v2.pth"结尾，则使用Swin2SR模型
        if filename.endswith(".v2.pth"):
            model = Swin2SR(
                upscale=scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="1conv",
            )
            params = None
        else:
            # 否则使用SwinIR模型
            model = SwinIR(
                upscale=scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="3conv",
            )
            params = "params_ema"

        # 加载预训练模型
        pretrained_model = torch.load(filename)
        # 如果有参数，则加载参数
        if params is not None:
            model.load_state_dict(pretrained_model[params], strict=True)
        else:
            # 否则加载整个模型
            model.load_state_dict(pretrained_model, strict=True)
        # 返回加载的模型
        return model
# 对图像进行放大处理
def upscale(
        img,
        model,
        tile=None,
        tile_overlap=None,
        window_size=8,
        scale=4,
):
    # 如果未指定瓦片大小，则使用默认值
    tile = tile or opts.SWIN_tile
    # 如果未指定瓦片重叠区域大小，则使用默认值
    tile_overlap = tile_overlap or opts.SWIN_tile_overlap

    # 将图像转换为 NumPy 数组
    img = np.array(img)
    # 将图像通道顺序从 RGB 转换为 BGR
    img = img[:, :, ::-1]
    # 将通道维度移动到第一维
    img = np.moveaxis(img, 2, 0) / 255
    # 将 NumPy 数组转换为 PyTorch 张量
    img = torch.from_numpy(img).float()
    # 在第一维度上增加一个维度，将图像移动到指定设备上
    img = img.unsqueeze(0).to(device_swinir, dtype=devices.dtype)
    # 禁用梯度计算，并使用自动混合精度
    with torch.no_grad(), devices.autocast():
        # 获取图像的高度和宽度
        _, _, h_old, w_old = img.size()
        # 计算需要填充的高度和宽度
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        # 在垂直方向上对图像进行填充
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h_old + h_pad, :]
        # 在水平方向上对图像进行填充
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w_old + w_pad]
        # 进行推理处理，得到放大后的输出
        output = inference(img, model, tile, tile_overlap, window_size, scale)
        # 裁剪输出图像到原始大小
        output = output[..., : h_old * scale, : w_old * scale]
        # 将输出转换为 NumPy 数组，并进行后续处理
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        # 如果输出是三维的，则重新排列通道顺序
        if output.ndim == 3:
            output = np.transpose(
                output[[2, 1, 0], :, :], (1, 2, 0)
            )  # CHW-RGB to HCW-BGR
        # 将输出转换为 uint8 类型
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        # 将 NumPy 数组转换为 PIL 图像并返回
        return Image.fromarray(output, "RGB")


# 进行推理处理
def inference(img, model, tile, tile_overlap, window_size, scale):
    # 获取图像的维度信息
    b, c, h, w = img.size()
    # 确定瓦片大小不超过图像的高度和宽度
    tile = min(tile, h, w)
    # 确保瓦片大小是窗口大小的倍数
    assert tile % window_size == 0, "tile size should be a multiple of window_size"
    # 设置放大倍数
    sf = scale

    # 计算瓦片之间的步长
    stride = tile - tile_overlap
    # 计算垂直方向和水平方向的瓦片索引列表
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    # 创建用于存储结果的张量 E 和 W
    E = torch.zeros(b, c, h * sf, w * sf, dtype=devices.dtype, device=device_swinir).type_as(img)
    W = torch.zeros_like(E, dtype=devices.dtype, device=device_swinir)
    # 使用 tqdm 创建进度条，总数为 h_idx_list 和 w_idx_list 的乘积，描述为"SwinIR tiles"
    with tqdm(total=len(h_idx_list) * len(w_idx_list), desc="SwinIR tiles") as pbar:
        # 遍历 h_idx_list
        for h_idx in h_idx_list:
            # 如果状态为中断或跳过，则跳出循环
            if state.interrupted or state.skipped:
                break

            # 遍历 w_idx_list
            for w_idx in w_idx_list:
                # 如果状态为中断或跳过，则跳出循环
                if state.interrupted or state.skipped:
                    break

                # 从图像中获取指定区域的图块
                in_patch = img[..., h_idx: h_idx + tile, w_idx: w_idx + tile]
                # 使用模型处理输入图块，得到输出图块
                out_patch = model(in_patch)
                # 创建与输出图块相同形状的全 1 张量
                out_patch_mask = torch.ones_like(out_patch)

                # 将输出图块加到 E 中指定位置
                E[
                ..., h_idx * sf: (h_idx + tile) * sf, w_idx * sf: (w_idx + tile) * sf
                ].add_(out_patch)
                # 将全 1 张量加到 W 中指定位置
                W[
                ..., h_idx * sf: (h_idx + tile) * sf, w_idx * sf: (w_idx + tile) * sf
                ].add_(out_patch_mask)
                # 更新进度条
                pbar.update(1)
    # 计算 E 除以 W 的结果
    output = E.div_(W)

    # 返回计算结果
    return output
# 定义一个名为on_ui_settings的函数
def on_ui_settings():
    # 导入gradio库
    import gradio as gr

    # 添加一个名为"SWIN_tile"的选项，用于设置所有SwinIR的瓦片大小
    shared.opts.add_option("SWIN_tile", shared.OptionInfo(192, "Tile size for all SwinIR.", gr.Slider, {"minimum": 16, "maximum": 512, "step": 16}, section=('upscaling', "Upscaling")))
    # 添加一个名为"SWIN_tile_overlap"的选项，用于设置SwinIR的瓦片重叠像素数
    shared.opts.add_option("SWIN_tile_overlap", shared.OptionInfo(8, "Tile overlap, in pixels for SwinIR. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}, section=('upscaling', "Upscaling")))
    # 检查torch版本是否大于等于2且不在Windows系统上，如果是则添加一个名为"SWIN_torch_compile"的选项
    if int(torch.__version__.split('.')[0]) >= 2 and platform.system() != "Windows":    # torch.compile() require pytorch 2.0 or above, and not on Windows
        shared.opts.add_option("SWIN_torch_compile", shared.OptionInfo(False, "Use torch.compile to accelerate SwinIR.", gr.Checkbox, {"interactive": True}, section=('upscaling', "Upscaling")).info("Takes longer on first run"))

# 调用script_callbacks模块的on_ui_settings函数，并传入on_ui_settings函数作为参数
script_callbacks.on_ui_settings(on_ui_settings)
```