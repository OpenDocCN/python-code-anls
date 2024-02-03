# `stable-diffusion-webui\extensions-builtin\ScuNET\scripts\scunet_model.py`

```
# 导入必要的库
import sys

import PIL.Image
import numpy as np
import torch
from tqdm import tqdm

# 导入自定义模块
import modules.upscaler
from modules import devices, modelloader, script_callbacks, errors
from scunet_model_arch import SCUNet

# 导入额外的模块
from modules.modelloader import load_file_from_url
from modules.shared import opts

# 定义一个名为 UpscalerScuNET 的类，继承自 Upscaler 类
class UpscalerScuNET(modules.upscaler.Upscaler):
    # 初始化方法
    def __init__(self, dirname):
        # 初始化类属性
        self.name = "ScuNET"
        self.model_name = "ScuNET GAN"
        self.model_name2 = "ScuNET PSNR"
        self.model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth"
        self.model_url2 = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
        self.user_path = dirname
        # 调用父类的初始化方法
        super().__init__()
        # 查找模型文件
        model_paths = self.find_models(ext_filter=[".pth"])
        scalers = []
        add_model2 = True
        # 遍历模型文件
        for file in model_paths:
            # 根据文件路径判断模型名称
            if file.startswith("http"):
                name = self.model_name
            else:
                name = modelloader.friendly_name(file)
            # 判断是否需要添加第二个模型
            if name == self.model_name2 or file == self.model_url2:
                add_model2 = False
            try:
                # 创建 UpscalerData 对象并添加到 scalers 列表中
                scaler_data = modules.upscaler.UpscalerData(name, file, self, 4)
                scalers.append(scaler_data)
            except Exception:
                # 报告加载模型时的错误信息
                errors.report(f"Error loading ScuNET model: {file}", exc_info=True)
        # 如果需要添加第二个模型，则创建 UpscalerData 对象并添加到 scalers 列表中
        if add_model2:
            scaler_data2 = modules.upscaler.UpscalerData(self.model_name2, self.model_url2, self)
            scalers.append(scaler_data2)
        # 设置类属性 scalers
        self.scalers = scalers

    # 静态方法，禁用 Torch 的梯度计算
    @staticmethod
    @torch.no_grad()
    # 对图像进行分块推断
    def tiled_inference(img, model):
        # 获取图像的高度和宽度
        h, w = img.shape[2:]
        # 获取分块大小和重叠大小
        tile = opts.SCUNET_tile
        tile_overlap = opts.SCUNET_tile_overlap
        # 如果分块大小为0，则直接使用模型对整个图像进行推断
        if tile == 0:
            return model(img)

        # 获取设备信息
        device = devices.get_device_for('scunet')
        # 确保分块大小是窗口大小的倍数
        assert tile % 8 == 0, "tile size should be a multiple of window_size"
        sf = 1

        # 计算分块的步长
        stride = tile - tile_overlap
        # 计算高度和宽度的索引列表
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        # 初始化输出结果的张量
        E = torch.zeros(1, 3, h * sf, w * sf, dtype=img.dtype, device=device)
        W = torch.zeros_like(E, dtype=devices.dtype, device=device)

        # 使用进度条显示分块推断的进度
        with tqdm(total=len(h_idx_list) * len(w_idx_list), desc="ScuNET tiles") as pbar:
            # 遍历高度索引列表
            for h_idx in h_idx_list:

                # 遍历宽度索引列表
                for w_idx in w_idx_list:

                    # 获取当前分块的输入数据
                    in_patch = img[..., h_idx: h_idx + tile, w_idx: w_idx + tile]

                    # 使用模型对分块进行推断
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    # 将分块推断结果添加到输出张量中
                    E[
                        ..., h_idx * sf: (h_idx + tile) * sf, w_idx * sf: (w_idx + tile) * sf
                    ].add_(out_patch)
                    W[
                        ..., h_idx * sf: (h_idx + tile) * sf, w_idx * sf: (w_idx + tile) * sf
                    ].add_(out_patch_mask)
                    # 更新进度条
                    pbar.update(1)
        # 计算最终输出结果
        output = E.div_(W)

        return output
    # 对图像进行放大处理
    def do_upscale(self, img: PIL.Image.Image, selected_file):
    
        # 释放Torch的缓存
        devices.torch_gc()
        
        try:
            # 加载选定文件中的模型
            model = self.load_model(selected_file)
        except Exception as e:
            # 如果加载模型失败，则打印错误信息并返回原始图像
            print(f"ScuNET: Unable to load model from {selected_file}: {e}", file=sys.stderr)
            return img
        
        # 获取用于'ScuNET'的设备
        device = devices.get_device_for('scunet')
        # 获取SCUNET_tile的值
        tile = opts.SCUNET_tile
        # 获取图像的高度和宽度
        h, w = img.height, img.width
        # 将图像转换为NumPy数组
        np_img = np.array(img)
        # 将RGB通道转换为BGR通道
        np_img = np_img[:, :, ::-1]  # RGB to BGR
        # 将图像数组转置为CHW格式
        np_img = np_img.transpose((2, 0, 1)) / 255  # HWC to CHW
        # 将NumPy数组转换为Torch张量，并进行相应的处理
        torch_img = torch.from_numpy(np_img).float().unsqueeze(0).to(device)  # type: ignore
        
        # 如果tile大于图像的高度或宽度，则进行填充
        if tile > h or tile > w:
            _img = torch.zeros(1, 3, max(h, tile), max(w, tile), dtype=torch_img.dtype, device=torch_img.device)
            _img[:, :, :h, :w] = torch_img # pad image
            torch_img = _img
        
        # 对图像进行分块推理
        torch_output = self.tiled_inference(torch_img, model).squeeze(0)
        # 去除可能存在的填充部分
        torch_output = torch_output[:, :h * 1, :w * 1] # remove padding, if any
        # 将Torch张量转换为NumPy数组，并进行相应的处理
        np_output: np.ndarray = torch_output.float().cpu().clamp_(0, 1).numpy()
        # 释放不再需要的Torch张量
        del torch_img, torch_output
        # 释放Torch的缓存
        devices.torch_gc()
        
        # 将输出数组转置为HWC格式
        output = np_output.transpose((1, 2, 0))  # CHW to HWC
        # 将BGR通道转换为RGB通道
        output = output[:, :, ::-1]  # BGR to RGB
        # 将NumPy数组转换为PIL图像，并返回
        return PIL.Image.fromarray((output * 255).astype(np.uint8))
    # 加载模型
    def load_model(self, path: str):
        # 获取设备信息
        device = devices.get_device_for('scunet')
        # 如果路径以 "http" 开头
        if path.startswith("http"):
            # TODO: 这里根本没有使用 `path`？
            # 从 URL 加载文件
            filename = load_file_from_url(self.model_url, model_dir=self.model_download_path, file_name=f"{self.name}.pth")
        else:
            # 否则使用给定路径
            filename = path
        # 创建 SCUNet 模型对象
        model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
        # 加载模型的状态字典
        model.load_state_dict(torch.load(filename), strict=True)
        # 设置模型为评估模式
        model.eval()
        # 将模型的参数设置为不需要梯度
        for _, v in model.named_parameters():
            v.requires_grad = False
        # 将模型移动到指定设备
        model = model.to(device)

        # 返回加载的模型
        return model
# 定义一个名为on_ui_settings的函数
def on_ui_settings():
    # 导入gradio库并重命名为gr
    import gradio as gr
    # 从modules模块中导入shared对象
    from modules import shared

    # 向共享的选项对象中添加"SCUNET_tile"选项，包括默认值、描述、控件类型、控件参数和所属部分信息
    shared.opts.add_option("SCUNET_tile", shared.OptionInfo(256, "Tile size for SCUNET upscalers.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}, section=('upscaling', "Upscaling")).info("0 = no tiling"))
    # 向共享的选项对象中添加"SCUNET_tile_overlap"选项，包括默认值、描述、控件类型、控件参数和所属部分信息
    shared.opts.add_option("SCUNET_tile_overlap", shared.OptionInfo(8, "Tile overlap for SCUNET upscalers.", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}, section=('upscaling', "Upscaling")).info("Low values = visible seam"))

# 调用script_callbacks模块中的on_ui_settings函数，并传入定义的on_ui_settings函数作为参数
script_callbacks.on_ui_settings(on_ui_settings)
```