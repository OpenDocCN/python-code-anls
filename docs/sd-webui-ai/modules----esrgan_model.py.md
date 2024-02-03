# `stable-diffusion-webui\modules\esrgan_model.py`

```py
# 导入必要的库
import sys

import numpy as np
import torch
from PIL import Image

# 导入自定义模块
import modules.esrgan_model_arch as arch
from modules import modelloader, images, devices
from modules.shared import opts
from modules.upscaler import Upscaler, UpscalerData

# 将模型参数从特定模型结构转换为普通模型结构
def mod2normal(state_dict):
    # this code is copied from https://github.com/victorca25/iNNfer
    # 检查是否存在特定的键值对
    if 'conv_first.weight' in state_dict:
        # 创建新的模型参数字典
        crt_net = {}
        # 获取所有键值对的键
        items = list(state_dict)

        # 将特定键值对转换为普通模型结构的键值对
        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        # 遍历所有键值对，将特定键值对转换为普通模型结构的键值对
        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net['model.1.sub.23.weight'] = state_dict['trunk_conv.weight']
        crt_net['model.1.sub.23.bias'] = state_dict['trunk_conv.bias']
        crt_net['model.3.weight'] = state_dict['upconv1.weight']
        crt_net['model.3.bias'] = state_dict['upconv1.bias']
        crt_net['model.6.weight'] = state_dict['upconv2.weight']
        crt_net['model.6.bias'] = state_dict['upconv2.bias']
        crt_net['model.8.weight'] = state_dict['HRconv.weight']
        crt_net['model.8.bias'] = state_dict['HRconv.bias']
        crt_net['model.10.weight'] = state_dict['conv_last.weight']
        crt_net['model.10.bias'] = state_dict['conv_last.bias']
        state_dict = crt_net
    return state_dict

# 将Residual SRGAN模型参数转换为普通模型结构
def resrgan2normal(state_dict, nb=23):
    # this code is copied from https://github.com/victorca25/iNNfer
    # 检查状态字典中是否包含指定的键
    if "conv_first.weight" in state_dict and "body.0.rdb1.conv1.weight" in state_dict:
        # 初始化变量 re8x 和 crt_net
        re8x = 0
        crt_net = {}
        # 获取状态字典中所有键的列表
        items = list(state_dict)

        # 将 conv_first.weight 和 conv_first.bias 添加到 crt_net 中
        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        # 遍历状态字典中的键
        for k in items.copy():
            # 如果键中包含 'rdb'，则进行处理
            if "rdb" in k:
                # 替换键中的部分内容，生成新的键
                ori_k = k.replace('body.', 'model.1.sub.')
                ori_k = ori_k.replace('.rdb', '.RDB')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                # 将处理后的键和对应的值添加到 crt_net 中，并从 items 中移除该键
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        # 将 conv_body.weight 和 conv_body.bias 添加到 crt_net 中
        crt_net[f'model.1.sub.{nb}.weight'] = state_dict['conv_body.weight']
        crt_net[f'model.1.sub.{nb}.bias'] = state_dict['conv_body.bias']
        crt_net['model.3.weight'] = state_dict['conv_up1.weight']
        crt_net['model.3.bias'] = state_dict['conv_up1.bias']
        crt_net['model.6.weight'] = state_dict['conv_up2.weight']
        crt_net['model.6.bias'] = state_dict['conv_up2.bias']

        # 如果状态字典中包含 'conv_up3.weight'，则进行处理
        if 'conv_up3.weight' in state_dict:
            # 修改 re8x 的值为 3，并将 conv_up3.weight 和 conv_up3.bias 添加到 crt_net 中
            re8x = 3
            crt_net['model.9.weight'] = state_dict['conv_up3.weight']
            crt_net['model.9.bias'] = state_dict['conv_up3.bias']

        # 将 conv_hr.weight 和 conv_hr.bias 添加到 crt_net 中
        crt_net[f'model.{8+re8x}.weight'] = state_dict['conv_hr.weight']
        crt_net[f'model.{8+re8x}.bias'] = state_dict['conv_hr.bias']
        # 将 conv_last.weight 和 conv_last.bias 添加到 crt_net 中
        crt_net[f'model.{10+re8x}.weight'] = state_dict['conv_last.weight']
        crt_net[f'model.{10+re8x}.bias'] = state_dict['conv_last.bias']

        # 更新状态字典为 crt_net
        state_dict = crt_net
    # 返回更新后的状态字典
    return state_dict
# 推断参数的函数，根据给定的状态字典
def infer_params(state_dict):
    # 初始化变量
    scale2x = 0
    scalemin = 6
    n_uplayer = 0
    plus = False

    # 遍历状态字典中的每个键
    for block in list(state_dict):
        # 拆分键名
        parts = block.split(".")
        n_parts = len(parts)
        # 判断条件，更新变量值
        if n_parts == 5 and parts[2] == "sub":
            nb = int(parts[3])
        elif n_parts == 3:
            part_num = int(parts[1])
            if (part_num > scalemin
                and parts[0] == "model"
                and parts[2] == "weight"):
                scale2x += 1
            if part_num > n_uplayer:
                n_uplayer = part_num
                out_nc = state_dict[block].shape[0]
        if not plus and "conv1x1" in block:
            plus = True

    # 获取特定键对应的值，更新变量值
    nf = state_dict["model.0.weight"].shape[0]
    in_nc = state_dict["model.0.weight"].shape[1]
    out_nc = out_nc
    scale = 2 ** scale2x

    # 返回推断出的参数
    return in_nc, out_nc, nf, nb, plus, scale


# ESRGAN 的升频器类，继承自 Upscaler 类
class UpscalerESRGAN(Upscaler):
    # 初始化函数，接收一个目录名参数
    def __init__(self, dirname):
        # 初始化对象属性
        self.name = "ESRGAN"
        self.model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth"
        self.model_name = "ESRGAN_4x"
        self.scalers = []
        self.user_path = dirname
        # 调用父类的初始化函数
        super().__init__()
        # 查找模型文件路径
        model_paths = self.find_models(ext_filter=[".pt", ".pth"])
        scalers = []
        # 如果没有找到模型文件，创建一个 UpscalerData 对象并添加到 scalers 列表中
        if len(model_paths) == 0:
            scaler_data = UpscalerData(self.model_name, self.model_url, self, 4)
            scalers.append(scaler_data)
        # 遍历模型文件路径
        for file in model_paths:
            # 根据文件路径判断模型名称
            if file.startswith("http"):
                name = self.model_name
            else:
                name = modelloader.friendly_name(file)

            # 创建一个 UpscalerData 对象并添加到 scalers 列表中
            scaler_data = UpscalerData(name, file, self, 4)
            self.scalers.append(scaler_data)
    # 定义一个方法，用于对图像进行放大处理
    def do_upscale(self, img, selected_model):
        # 尝试加载选定的模型，如果出现异常则打印错误信息并返回原始图像
        try:
            model = self.load_model(selected_model)
        except Exception as e:
            print(f"Unable to load ESRGAN model {selected_model}: {e}", file=sys.stderr)
            return img
        # 将模型加载到指定设备上
        model.to(devices.device_esrgan)
        # 使用加载的模型对图像进行放大处理
        img = esrgan_upscale(model, img)
        # 返回处理后的图像
        return img
    # 加载模型的方法，接受一个路径参数
    def load_model(self, path: str):
        # 如果路径以"http"开头
        if path.startswith("http"):
            # TODO: 这里完全没有使用`path`？
            # 从 URL 加载文件到指定目录
            filename = modelloader.load_file_from_url(
                url=self.model_url,
                model_dir=self.model_download_path,
                file_name=f"{self.model_name}.pth",
            )
        else:
            # 否则直接使用给定的路径作为文件名
            filename = path

        # 加载模型的状态字典，根据设备类型选择在 CPU 上加载
        state_dict = torch.load(filename, map_location='cpu' if devices.device_esrgan.type == 'mps' else None)

        # 如果状态字典中包含"params_ema"
        if "params_ema" in state_dict:
            # 使用"params_ema"作为状态字典
            state_dict = state_dict["params_ema"]
        # 如果状态字典中包含"params"
        elif "params" in state_dict:
            # 使用"params"作为状态字典
            state_dict = state_dict["params"]
            # 根据文件名选择不同的卷积层数量
            num_conv = 16 if "realesr-animevideov3" in filename else 32
            # 创建一个特定参数的模型
            model = arch.SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=4, act_type='prelu')
            # 加载模型的状态字典
            model.load_state_dict(state_dict)
            # 设置模型为评估模式
            model.eval()
            # 返回加载的模型
            return model

        # 如果状态字典中包含特定的键
        if "body.0.rdb1.conv1.weight" in state_dict and "conv_first.weight" in state_dict:
            # 根据文件名选择不同的参数
            nb = 6 if "RealESRGAN_x4plus_anime_6B" in filename else 23
            # 转换状态字典中的参数
            state_dict = resrgan2normal(state_dict, nb)
        # 如果状态字典中包含"conv_first.weight"
        elif "conv_first.weight" in state_dict:
            # 转换状态字典中的参数
            state_dict = mod2normal(state_dict)
        # 如果状态字典中不包含"model.0.weight"
        elif "model.0.weight" not in state_dict:
            # 抛出异常，表示文件不是已识别的 ESRGAN 模型
            raise Exception("The file is not a recognized ESRGAN model.")

        # 推断模型参数
        in_nc, out_nc, nf, nb, plus, mscale = infer_params(state_dict)

        # 创建一个特定参数的模型
        model = arch.RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, upscale=mscale, plus=plus)
        # 加载模型的状态字典
        model.load_state_dict(state_dict)
        # 设置模型为评估模式
        model.eval()

        # 返回加载的模型
        return model
# 使用指定的模型对图像进行放大，不进行分块处理
def upscale_without_tiling(model, img):
    # 将图像转换为 NumPy 数组
    img = np.array(img)
    # 颜色通道顺序转换为 RGB
    img = img[:, :, ::-1]
    # 转置图像数组，调整通道顺序为 (2, 0, 1)，并归一化到 [0, 1] 范围
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255
    # 将 NumPy 数组转换为 PyTorch 张量，并转换为 float 类型
    img = torch.from_numpy(img).float()
    # 在第 0 维度上增加一个维度，并将张量移动到指定设备上
    img = img.unsqueeze(0).to(devices.device_esrgan)
    # 禁用梯度计算，对图像进行模型推理
    with torch.no_grad():
        output = model(img)
    # 压缩维度，转换为 float 类型，移动到 CPU，限制值在 [0, 1] 范围内，转换为 NumPy 数组
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    # 将输出转换为 [0, 255] 范围内的值，并调整通道顺序为 (0, 2)
    output = 255. * np.moveaxis(output, 0, 2)
    # 将输出转换为 uint8 类型
    output = output.astype(np.uint8)
    # 将输出的颜色通道顺序转换为 BGR
    output = output[:, :, ::-1]
    # 将 NumPy 数组转换为 PIL 图像对象，颜色模式为 RGB
    return Image.fromarray(output, 'RGB')


# 使用 ESRGAN 模型对图像进行放大，根据 opts.ESRGAN_tile 的值决定是否进行分块处理
def esrgan_upscale(model, img):
    # 如果 opts.ESRGAN_tile 为 0，则不进行分块处理
    if opts.ESRGAN_tile == 0:
        return upscale_without_tiling(model, img)

    # 将图像分割为网格，根据 opts.ESRGAN_tile 和 opts.ESRGAN_tile_overlap 进行分块
    grid = images.split_grid(img, opts.ESRGAN_tile, opts.ESRGAN_tile, opts.ESRGAN_tile_overlap)
    newtiles = []
    scale_factor = 1

    # 遍历网格中的每个块
    for y, h, row in grid.tiles:
        newrow = []
        for tiledata in row:
            x, w, tile = tiledata

            # 对每个块进行放大处理
            output = upscale_without_tiling(model, tile)
            # 计算放大比例
            scale_factor = output.width // tile.width

            newrow.append([x * scale_factor, w * scale_factor, output])
        newtiles.append([y * scale_factor, h * scale_factor, newrow])

    # 创建新的网格对象，将放大后的块组合成完整图像
    newgrid = images.Grid(newtiles, grid.tile_w * scale_factor, grid.tile_h * scale_factor, grid.image_w * scale_factor, grid.image_h * scale_factor, grid.overlap * scale_factor)
    output = images.combine_grid(newgrid)
    return output
```