# `stable-diffusion-webui\scripts\outpainting_mk_2.py`

```
# 导入 math 模块
import math

# 导入 numpy 模块，并使用 np 别名
import numpy as np
# 导入 skimage 模块
import skimage

# 导入 modules.scripts 模块，并使用 scripts 别名
import modules.scripts as scripts
# 导入 gradio 模块，并使用 gr 别名
import gradio as gr
# 从 PIL 模块中导入 Image 和 ImageDraw 类
from PIL import Image, ImageDraw

# 从 modules 包中导入 images 模块
from modules import images
# 从 modules.processing 模块中导入 Processed 和 process_images 函数
from modules.processing import Processed, process_images
# 从 modules.shared 模块中导入 opts 和 state 变量
from modules.shared import opts, state

# 从 https://github.com/parlance-zz/g-diffuser-bot 中获取的函数，用于获取匹配的噪声
def get_matched_noise(_np_src_image, np_mask_rgb, noise_q=1, color_variation=0.05):
    # 定义内部函数 _fft2，用于进行二维傅里叶变换
    def _fft2(data):
        # 如果数据维度大于2，即有多个通道
        if data.ndim > 2:
            # 创建一个与输入数据相同维度的复数数组
            out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            # 遍历每个通道
            for c in range(data.shape[2]):
                # 获取当前通道的数据
                c_data = data[:, :, c]
                # 对当前通道的数据进行傅里叶变换，并进行正交归一化
                out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
                # 对变换后的数据进行反移位
                out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
        else:  # 如果只有一个通道
            # 创建一个与输入数据相同维度的复数数组
            out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            # 对数据进行傅里叶变换，并进行正交归一化
            out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
            # 对变换后的数据进行反移位
            out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

        return out_fft

    # 定义内部函数 _ifft2，用于进行二维傅里叶逆变换
    def _ifft2(data):
        # 如果数据维度大于2，即有多个通道
        if data.ndim > 2:
            # 创建一个与输入数据相同维度的复数数组
            out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            # 遍历每个通道
            for c in range(data.shape[2]):
                # 获取当前通道的数据
                c_data = data[:, :, c]
                # 对当前通道的数据进行傅里叶逆变换，并进行正交归一化
                out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
                # 对逆变换后的数据进行反移位
                out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
        else:  # 如果只有一个通道
            # 创建一个与输入数据相同维度的复数数组
            out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            # 对数据进行傅里叶逆变换，并进行正交归一化
            out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
            # 对逆变换后的数据进行反移位
            out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

        return out_ifft
    # 根据给定的参数生成一个高斯窗口
    def _get_gaussian_window(width, height, std=3.14, mode=0):
        # 计算窗口的缩放比例
        window_scale_x = float(width / min(width, height))
        window_scale_y = float(height / min(width, height))

        # 创建一个指定大小的全零矩阵作为窗口
        window = np.zeros((width, height))
        # 计算窗口的 x 坐标
        x = (np.arange(width) / width * 2. - 1.) * window_scale_x
        # 遍历窗口的 y 坐标
        for y in range(height):
            fy = (y / height * 2. - 1.) * window_scale_y
            # 根据模式选择不同的计算方式生成高斯窗口
            if mode == 0:
                window[:, y] = np.exp(-(x ** 2 + fy ** 2) * std)
            else:
                window[:, y] = (1 / ((x ** 2 + 1.) * (fy ** 2 + 1.))) ** (std / 3.14)  # hey wait a minute that's not gaussian

        return window

    # 根据灰度掩码生成 RGB 彩色掩码
    def _get_masked_window_rgb(np_mask_grey, hardness=1.):
        np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
        # 根据硬度参数对灰度掩码进行处理
        if hardness != 1.:
            hardened = np_mask_grey[:] ** hardness
        else:
            hardened = np_mask_grey[:]
        # 将处理后的灰度掩码赋值给 RGB 彩色掩码的每个通道
        for c in range(3):
            np_mask_rgb[:, :, c] = hardened[:]
        return np_mask_rgb

    # 获取源图像的宽度、高度和通道数
    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    # 对源图像进行掩码处理
    _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2) / 3.)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3

    # 对窗口化后的图像进行处理
    windowed_image = _np_src_image * (1. - _get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb  # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color

    # 对窗口化后的图像进行傅立叶变换
    src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist

    # 创建一个具有静态种子的生成器，使得输出结果具有确定性
    rng = np.random.default_rng(0)
    # 使用高斯窗口生成噪声窗口，用于后续处理
    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    # 生成随机的彩色噪声
    noise_rgb = rng.random((width, height, num_channels))
    # 将彩色噪声转换为灰度噪声
    noise_grey = (np.sum(noise_rgb, axis=2) / 3.)
    # 根据颜色变化参数将彩色噪声与灰度噪声混合
    noise_rgb *= color_variation  # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        # 将灰度噪声混合到彩色噪声中
        noise_rgb[:, :, c] += (1. - color_variation) * noise_grey

    # 对彩色噪声进行傅里叶变换
    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        # 对每个通道的傅里叶变换结果应用噪声窗口
        noise_fft[:, :, c] *= noise_window
    # 将傅里叶逆变换得到处理后的噪声图像
    noise_rgb = np.real(_ifft2(noise_fft))
    # 对处理后的噪声图像进行傅里叶变换
    shaped_noise_fft = _fft2(noise_rgb)
    # 对噪声图像进行形状调整
    shaped_noise_fft[:, :, :] = np.absolute(shaped_noise_fft[:, :, :]) ** 2 * (src_dist ** noise_q) * src_phase  # perform the actual shaping

    # 设置亮度变化参数
    brightness_variation = 0.  # color_variation # todo: temporarily tieing brightness variation to color variation for now
    # 调整对比度
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.

    # 使用 scikit-image 进行直方图匹配
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    shaped_noise -= np.min(shaped_noise)
    shaped_noise /= np.max(shaped_noise)
    # 对噪声图像进行直方图匹配
    shaped_noise[img_mask, :] = skimage.exposure.match_histograms(shaped_noise[img_mask, :] ** 1., contrast_adjusted_np_src[ref_mask, :], channel_axis=1)
    # 将匹配后的噪声图像与原始图像进行混合
    shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb

    # 复制匹配后的噪声图像
    matched_noise = shaped_noise[:]

    # 将匹配后的噪声图像限制在 0 到 1 之间并返回
    return np.clip(matched_noise, 0., 1.)
# 定义一个名为Script的类，继承自scripts.Script类
class Script(scripts.Script):
    # 定义一个方法title，返回字符串"Outpainting mk2"
    def title(self):
        return "Outpainting mk2"

    # 定义一个方法show，接受一个布尔值is_img2img作为参数，返回is_img2img的值
    def show(self, is_img2img):
        return is_img2img

    # 定义一个方法ui，接受一个布尔值is_img2img作为参数
    def ui(self, is_img2img):
        # 如果is_img2img为False，则返回None
        if not is_img2img:
            return None

        # 创建一个HTML元素，显示推荐设置信息
        info = gr.HTML("<p style=\"margin-bottom:0.75em\">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>")

        # 创建一个滑块元素，用于设置扩展像素数
        pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=256, step=8, value=128, elem_id=self.elem_id("pixels"))
        # 创建一个滑块元素，用于设置遮罩模糊程度
        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=8, elem_id=self.elem_id("mask_blur"))
        # 创建一个复选框组元素，用于设置外部方向
        direction = gr.CheckboxGroup(label="Outpainting direction", choices=['left', 'right', 'up', 'down'], value=['left', 'right', 'up', 'down'], elem_id=self.elem_id("direction"))
        # 创建一个滑块元素，用于设置衰减指数（越低越详细）
        noise_q = gr.Slider(label="Fall-off exponent (lower=higher detail)", minimum=0.0, maximum=4.0, step=0.01, value=1.0, elem_id=self.elem_id("noise_q"))
        # 创建一个滑块元素，用于设置颜色变化
        color_variation = gr.Slider(label="Color variation", minimum=0.0, maximum=1.0, step=0.01, value=0.05, elem_id=self.elem_id("color_variation"))

        # 返回一个包含所有UI元素的列表
        return [info, pixels, mask_blur, direction, noise_q, color_variation]
```