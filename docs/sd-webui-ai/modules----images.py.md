# `stable-diffusion-webui\modules\images.py`

```py
# 导入必要的模块
from __future__ import annotations

import datetime
import pytz
import io
import math
import os
from collections import namedtuple
import re

import numpy as np
import piexif
import piexif.helper
from PIL import Image, ImageFont, ImageDraw, ImageColor, PngImagePlugin
import string
import json
import hashlib

# 导入自定义模块
from modules import sd_samplers, shared, script_callbacks, errors
from modules.paths_internal import roboto_ttf_file
from modules.shared import opts

# 定义常量LANCZOS，根据PIL版本选择不同的重采样方法
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

# 根据字体大小获取字体对象
def get_font(fontsize: int):
    try:
        return ImageFont.truetype(opts.font or roboto_ttf_file, fontsize)
    except Exception:
        return ImageFont.truetype(roboto_ttf_file, fontsize)

# 创建图像网格
def image_grid(imgs, batch_size=1, rows=None):
    # 根据参数计算行数
    if rows is None:
        if opts.n_rows > 0:
            rows = opts.n_rows
        elif opts.n_rows == 0:
            rows = batch_size
        elif opts.grid_prevent_empty_spots:
            rows = math.floor(math.sqrt(len(imgs)))
            while len(imgs) % rows != 0:
                rows -= 1
        else:
            rows = math.sqrt(len(imgs))
            rows = round(rows)
    if rows > len(imgs):
        rows = len(imgs)

    # 计算列数
    cols = math.ceil(len(imgs) / rows)

    # 调用回调函数处理图像网格
    params = script_callbacks.ImageGridLoopParams(imgs, cols, rows)
    script_callbacks.image_grid_callback(params)

    # 创建空白图像作为网格
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(params.cols * w, params.rows * h), color='black')

    # 将图像填充到网格中
    for i, img in enumerate(params.imgs):
        grid.paste(img, box=(i % params.cols * w, i // params.cols * h))

    return grid

# 定义命名元组Grid，表示图像网格的参数
Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])

# 将图像分割成网格
def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height

    # 计算非重叠区域的宽度和高度
    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    # 计算列数
    cols = math.ceil((w - overlap) / non_overlap_width)
    # 计算行数，向上取整，确保覆盖重叠区域
    rows = math.ceil((h - overlap) / non_overlap_height)

    # 计算水平方向上相邻瓦片之间的间距
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    # 计算垂直方向上相邻瓦片之间的间距
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    # 创建一个网格对象
    grid = Grid([], tile_w, tile_h, w, h, overlap)
    # 遍历每一行
    for row in range(rows):
        # 存储当前行的图像列表
        row_images = []

        # 计算当前行的起始y坐标
        y = int(row * dy)

        # 如果当前行的y坐标加上瓦片高度超过了图像高度，则调整y坐标
        if y + tile_h >= h:
            y = h - tile_h

        # 遍历当前行的每一列
        for col in range(cols):
            # 计算当前列的起始x坐标
            x = int(col * dx)

            # 如果当前列的x坐标加上瓦片宽度超过了图像宽度，则调整x坐标
            if x + tile_w >= w:
                x = w - tile_w

            # 根据起始坐标和瓦片尺寸裁剪图像，得到一个瓦片
            tile = image.crop((x, y, x + tile_w, y + tile_h))

            # 将瓦片的信息添加到当前行的图像列表中
            row_images.append([x, tile_w, tile])

        # 将当前行的图像列表添加到网格对象的瓦片列表中
        grid.tiles.append([y, tile_h, row_images])

    # 返回生成的网格对象
    return grid
# 合并网格中的图像块，生成合并后的图像
def combine_grid(grid):
    # 创建遮罩图像函数，将输入的数组转换为灰度图像
    def make_mask_image(r):
        # 将数组值映射到 0-255 范围内
        r = r * 255 / grid.overlap
        # 将数组转换为无符号 8 位整数类型
        r = r.astype(np.uint8)
        # 从数组创建灰度图像
        return Image.fromarray(r, 'L')

    # 创建水平方向的遮罩图像
    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    # 创建垂直方向的遮罩图像
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    # 创建一个新的 RGB 模式图像对象
    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    # 遍历网格中的每一行
    for y, h, row in grid.tiles:
        # 创建一个新的 RGB 模式图像对象，表示当前行的合并图像
        combined_row = Image.new("RGB", (grid.image_w, h))
        # 遍历当前行中的每个图像块
        for x, w, tile in row:
            # 如果当前图像块在行的起始位置
            if x == 0:
                # 直接粘贴图像块到合并行的起始位置
                combined_row.paste(tile, (0, 0))
                continue

            # 将图像块的左侧部分粘贴到合并行的指定位置，使用水平遮罩
            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            # 将图像块的右侧部分粘贴到合并行的指定位置
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        # 如果当前行在网格中的起始位置
        if y == 0:
            # 直接粘贴合并行到合并图像的起始位置
            combined_image.paste(combined_row, (0, 0))
            continue

        # 将合并行的上部分粘贴到合并图像的指定位置，使用垂直遮罩
        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        # 将合并行的下部分粘贴到合并图像的指定位置
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    # 返回合并后的图像
    return combined_image


# 定义网格注释类
class GridAnnotation:
    # 初始化方法，设置文本内容和是否激活状态
    def __init__(self, text='', is_active=True):
        self.text = text
        self.is_active = is_active
        self.size = None


# 绘制网格注释
def draw_grid_annotations(im, width, height, hor_texts, ver_texts, margin=0):

    # 获取激活状态下的文本颜色
    color_active = ImageColor.getcolor(opts.grid_text_active_color, 'RGB')
    # 获取非激活状态下的文本颜色
    color_inactive = ImageColor.getcolor(opts.grid_text_inactive_color, 'RGB')
    # 获取背景颜色
    color_background = ImageColor.getcolor(opts.grid_background_color, 'RGB')
    # 根据给定的文本、字体、行长度对文本进行换行处理
    def wrap(drawing, text, font, line_length):
        # 初始化行列表
        lines = ['']
        # 遍历文本中的单词
        for word in text.split():
            # 尝试将单词添加到当前行
            line = f'{lines[-1]} {word}'.strip()
            # 如果当前行的文本长度符合要求，则更新当前行
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            # 如果当前行的文本长度超过要求，则将单词添加到新行
            else:
                lines.append(word)
        # 返回处理后的行列表
        return lines

    # 在图像上绘制文本
    def draw_texts(drawing, draw_x, draw_y, lines, initial_fnt, initial_fontsize):
        # 遍历每一行文本
        for line in lines:
            # 初始化字体和字体大小
            fnt = initial_fnt
            fontsize = initial_fontsize
            # 调整字体大小，直到文本宽度符合要求或字体大小为0
            while drawing.multiline_textsize(line.text, font=fnt)[0] > line.allowed_width and fontsize > 0:
                fontsize -= 1
                fnt = get_font(fontsize)
            # 在指定位置绘制文本
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="center")

            # 如果文本不活跃，则绘制一条横线
            if not line.is_active:
                drawing.line((draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2, draw_y + line.size[1] // 2), fill=color_inactive, width=4)

            # 更新绘制位置
            draw_y += line.size[1] + line_spacing

    # 计算字体大小和行间距
    fontsize = (width + height) // 25
    line_spacing = fontsize // 2

    # 获取初始字体
    fnt = get_font(fontsize)

    # 计算左边距
    pad_left = 0 if sum([sum([len(line.text) for line in lines]) for lines in ver_texts]) == 0 else width * 3 // 4

    # 计算图像的列数和行数
    cols = im.width // width
    rows = im.height // height

    # 检查水平文本数量是否与列数相匹配
    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    # 检查垂直文本数量是否与行数相匹配
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'

    # 创建一个用于计算文本宽度的临时图像
    calc_img = Image.new("RGB", (1, 1), color_background)
    calc_d = ImageDraw.Draw(calc_img)
    # 遍历水平和垂直文本列表，以及对应的允许宽度
    for texts, allowed_width in zip(hor_texts + ver_texts, [width] * len(hor_texts) + [pad_left] * len(ver_texts):
        # 复制文本列表到 items，并清空原始文本列表
        items = [] + texts
        texts.clear()

        # 遍历 items 中的每一行文本
        for line in items:
            # 对文本进行换行处理，返回包含换行后文本的列表
            wrapped = wrap(calc_d, line.text, fnt, allowed_width)
            # 将换行后的文本转换为 GridAnnotation 对象，并添加到文本列表中
            texts += [GridAnnotation(x, line.is_active) for x in wrapped]

            # 计算每行文本的边界框大小
            for line in texts:
                bbox = calc_d.multiline_textbbox((0, 0), line.text, font=fnt)
                line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                line.allowed_width = allowed_width

    # 计算水平文本列表中每行文本的高度
    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in hor_texts]
    # 计算垂直文本列表中每行文本的高度
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in ver_texts]

    # 计算顶部填充值
    pad_top = 0 if sum(hor_text_heights) == 0 else max(hor_text_heights) + line_spacing * 2

    # 创建新的图像对象，设置背景颜色
    result = Image.new("RGB", (im.width + pad_left + margin * (cols-1), im.height + pad_top + margin * (rows-1)), color_background)

    # 遍历行和列，裁剪图像并粘贴到结果图像中
    for row in range(rows):
        for col in range(cols):
            cell = im.crop((width * col, height * row, width * (col+1), height * (row+1)))
            result.paste(cell, (pad_left + (width + margin) * col, pad_top + (height + margin) * row))

    # 创建 ImageDraw 对象
    d = ImageDraw.Draw(result)

    # 在每列的中心位置绘制水平文本
    for col in range(cols):
        x = pad_left + (width + margin) * col + width / 2
        y = pad_top / 2 - hor_text_heights[col] / 2
        draw_texts(d, x, y, hor_texts[col], fnt, fontsize)

    # 在每行的中心位置绘制垂直文本
    for row in range(rows):
        x = pad_left / 2
        y = pad_top + (height + margin) * row + height / 2 - ver_text_heights[row] / 2
        draw_texts(d, x, y, ver_texts[row], fnt, fontsize)

    # 返回结果图像
    return result
# 根据给定的参数绘制提示矩阵
def draw_prompt_matrix(im, width, height, all_prompts, margin=0):
    # 从所有提示中获取除第一个外的所有提示
    prompts = all_prompts[1:]
    # 计算分界线，向上取整
    boundary = math.ceil(len(prompts) / 2)

    # 水平方向的提示
    prompts_horiz = prompts[:boundary]
    # 垂直方向的提示
    prompts_vert = prompts[boundary:]

    # 生成水平方向的文本注释
    hor_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_horiz)] for pos in range(1 << len(prompts_horiz))]
    # 生成垂直方向的文本注释
    ver_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_vert)] for pos in range(1 << len(prompts_vert))]

    # 调用绘制网格注释的函数，返回结果
    return draw_grid_annotations(im, width, height, hor_texts, ver_texts, margin)


# 调整图像大小的函数，根据指定的调整模式、宽度和高度进行调整
def resize_image(resize_mode, im, width, height, upscaler_name=None):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """

    # 如果未提供upscaler_name，则使用默认值opts.upscaler_for_img2img
    upscaler_name = upscaler_name or opts.upscaler_for_img2img
    # 定义一个函数，用于调整图像大小
    def resize(im, w, h):
        # 如果没有指定放大器名称，或者放大器名称为"None"，或者图像模式为灰度图，则直接调整图像大小并返回
        if upscaler_name is None or upscaler_name == "None" or im.mode == 'L':
            return im.resize((w, h), resample=LANCZOS)

        # 计算图像的放大比例
        scale = max(w / im.width, h / im.height)

        # 如果放大比例大于1.0
        if scale > 1.0:
            # 查找指定名称的放大器
            upscalers = [x for x in shared.sd_upscalers if x.name == upscaler_name]
            # 如果找不到指定名称的放大器，则使用第一个放大器作为替代
            if len(upscalers) == 0:
                upscaler = shared.sd_upscalers[0]
                print(f"could not find upscaler named {upscaler_name or '<empty string>'}, using {upscaler.name} as a fallback")
            else:
                upscaler = upscalers[0]

            # 使用放大器对图像进行放大处理
            im = upscaler.scaler.upscale(im, scale, upscaler.data_path)

        # 如果调整后的图像宽高与目标宽高不一致，则再次调整图像大小
        if im.width != w or im.height != h:
            im = im.resize((w, h), resample=LANCZOS)

        # 返回调整后的图像
        return im

    # 如果调整模式为0
    if resize_mode == 0:
        # 调用resize函数对图像进行调整
        res = resize(im, width, height)

    # 如果调整模式为1
    elif resize_mode == 1:
        # 计算目标宽高比和原图宽高比
        ratio = width / height
        src_ratio = im.width / im.height

        # 根据比例计算源图像的宽高
        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        # 调整源图像的大小
        resized = resize(im, src_w, src_h)
        # 创建一个新的RGB图像
        res = Image.new("RGB", (width, height))
        # 将调整后的源图像粘贴到新图像的中心位置
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        # 计算目标图像的宽高比
        ratio = width / height
        # 计算原始图像的宽高比
        src_ratio = im.width / im.height

        # 根据比较结果确定原始图像的宽度
        src_w = width if ratio < src_ratio else im.width * height // im.height
        # 根据比较结果确定原始图像的高度
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        # 调整原始图像的大小
        resized = resize(im, src_w, src_h)
        # 创建一个新的 RGB 模式图像
        res = Image.new("RGB", (width, height))
        # 将调整后的图像粘贴到新图像中心位置
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        # 根据比较结果进行填充操作
        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                # 在顶部填充
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                # 在底部填充
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                # 在左侧填充
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                # 在右侧填充
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    # 返回处理后的图像
    return res
# 定义无效文件名字符，包括 '<>', ':"/\\|?*\n\r\t'
invalid_filename_chars = '<>:"/\\|?*\n\r\t'
# 定义无效文件名前缀，包括空格
invalid_filename_prefix = ' '
# 定义无效文件名后缀，包括空格和句号
invalid_filename_postfix = ' .'
# 定义正则表达式，用于匹配非字母字符
re_nonletters = re.compile(r'[\s' + string.punctuation + ']+')
# 定义正则表达式，用于匹配模式
re_pattern = re.compile(r"(.*?)(?:\[([^\[\]]+)\]|$)")
# 定义正则表达式，用于匹配模式参数
re_pattern_arg = re.compile(r"(.*)<([^>]*)>$")
# 定义最大文件名部分长度
max_filename_part_length = 128
# 定义一个特殊对象，用于标记无内容和跳过之前的文本
NOTHING_AND_SKIP_PREVIOUS_TEXT = object()

# 定义函数，用于清理文件名部分
def sanitize_filename_part(text, replace_spaces=True):
    # 如果文本为空，则返回空
    if text is None:
        return None

    # 如果需要替换空格，则将空格替换为下划线
    if replace_spaces:
        text = text.replace(' ', '_')

    # 清理文本中的无效字符
    text = text.translate({ord(x): '_' for x in invalid_filename_chars})
    # 去除无效前缀并限制文件名部分长度
    text = text.lstrip(invalid_filename_prefix)[:max_filename_part_length]
    # 去除无效后缀
    text = text.rstrip(invalid_filename_postfix)
    return text

# 定义文件名生成器类
class FilenameGenerator:
    # 默认时间格式
    default_time_format = '%Y%m%d%H%M%S'

    # 初始化函数，接受路径、种子、提示、图像和是否为 ZIP 文件作为参数
    def __init__(self, p, seed, prompt, image, zip=False):
        self.p = p
        self.seed = seed
        self.prompt = prompt
        self.image = image
        self.zip = zip

    # 获取 VAE 文件名的函数
    def get_vae_filename(self):
        """Get the name of the VAE file."""

        # 导入 sd_vae 模块
        import modules.sd_vae as sd_vae

        # 如果加载的 VAE 文件为空，则返回 "NoneType"
        if sd_vae.loaded_vae_file is None:
            return "NoneType"

        # 获取加载的 VAE 文件的基本文件名
        file_name = os.path.basename(sd_vae.loaded_vae_file)
        # 拆分文件名
        split_file_name = file_name.split('.')
        # 如果文件名包含多个部分且第一个部分为空，则返回第二个部分
        if len(split_file_name) > 1 and split_file_name[0] == '':
            return split_file_name[1]  # if the first character of the filename is "." then [1] is obtained.
        else:
            return split_file_name[0]
    # 检查是否存在提示信息，将提示信息转换为小写
    def hasprompt(self, *args):
        lower = self.prompt.lower()
        # 如果提示信息为空或者 self.p 为空，则返回 None
        if self.p is None or self.prompt is None:
            return None
        # 初始化输出结果字符串
        outres = ""
        # 遍历参数列表
        for arg in args:
            # 如果参数不为空
            if arg != "":
                # 将参数按 "|" 分割
                division = arg.split("|")
                # 获取期望值并转换为小写
                expected = division[0].lower()
                # 获取默认值（如果有的话）
                default = division[1] if len(division) > 1 else ""
                # 如果提示信息中包含期望值，则将期望值添加到输出结果中
                if lower.find(expected) >= 0:
                    outres = f'{outres}{expected}'
                # 否则，如果有默认值，则将默认值添加到输出结果中
                else:
                    outres = outres if default == "" else f'{outres}{default}'
        # 返回经过处理的输出结果
        return sanitize_filename_part(outres)

    # 获取没有样式的提示信息
    def prompt_no_style(self):
        # 如果提示信息为空或者 self.p 为空，则返回 None
        if self.p is None or self.prompt is None:
            return None

        # 初始化没有样式的提示信息
        prompt_no_style = self.prompt
        # 遍历样式列表，获取样式对应的提示信息
        for style in shared.prompt_styles.get_style_prompts(self.p.styles):
            if style:
                # 将样式中的部分替换为空，去除多余的逗号和空格
                for part in style.split("{prompt}"):
                    prompt_no_style = prompt_no_style.replace(part, "").replace(", ,", ",").strip().strip(',')

                # 去除样式中的内容，去除多余的逗号和空格
                prompt_no_style = prompt_no_style.replace(style, "").strip().strip(',').strip()

        # 返回经过处理的没有样式的提示信息
        return sanitize_filename_part(prompt_no_style, replace_spaces=False)

    # 获取提示信息中的单词
    def prompt_words(self):
        # 使用正则表达式将提示信息分割为单词列表
        words = [x for x in re_nonletters.split(self.prompt or "") if x]
        # 如果单词列表为空，则将 "empty" 添加到列表中
        if len(words) == 0:
            words = ["empty"]
        # 返回经过处理的单词列表的字符串形式
        return sanitize_filename_part(" ".join(words[0:opts.directories_max_prompt_words]), replace_spaces=False)
    # 定义一个方法用于获取当前时间，并根据传入参数格式化时间
    def datetime(self, *args):
        # 获取当前时间
        time_datetime = datetime.datetime.now()

        # 确定时间格式，如果有传入参数且不为空，则使用传入参数，否则使用默认时间格式
        time_format = args[0] if (args and args[0] != "") else self.default_time_format
        # 尝试根据第二个参数获取时区，如果参数个数大于1，则使用传入的时区，否则时区为None
        try:
            time_zone = pytz.timezone(args[1]) if len(args) > 1 else None
        except pytz.exceptions.UnknownTimeZoneError:
            time_zone = None

        # 将当前时间转换为指定时区的时间
        time_zone_time = time_datetime.astimezone(time_zone)
        # 根据指定格式格式化时间
        try:
            formatted_time = time_zone_time.strftime(time_format)
        except (ValueError, TypeError):
            formatted_time = time_zone_time.strftime(self.default_time_format)

        # 返回格式化后的时间，确保文件名部分合法
        return sanitize_filename_part(formatted_time, replace_spaces=False)

    # 定义一个方法用于计算图像的哈希值
    def image_hash(self, *args):
        # 获取哈希值的长度，如果有传入参数且不为空，则使用传入参数，否则长度为None
        length = int(args[0]) if (args and args[0] != "") else None
        # 计算图像的哈希值并截取指定长度返回
        return hashlib.sha256(self.image.tobytes()).hexdigest()[0:length]

    # 定义一个方法用于计算字符串的哈希值
    def string_hash(self, text, *args):
        # 获取哈希值的长度，如果有传入参数且不为空，则使用传入参数，否则长度为8
        length = int(args[0]) if (args and args[0] != "") else 8
        # 计算字符串的哈希值并截取指定长度返回
        return hashlib.sha256(text.encode()).hexdigest()[0:length]
    # 定义一个方法，用于对输入的字符串进行处理
    def apply(self, x):
        # 初始化结果字符串为空
        res = ''

        # 遍历正则表达式匹配到的所有内容
        for m in re_pattern.finditer(x):
            # 获取匹配到的文本和模式
            text, pattern = m.groups()

            # 如果模式为空，则将文本直接添加到结果字符串中并继续下一次循环
            if pattern is None:
                res += text
                continue

            # 初始化模式参数列表
            pattern_args = []
            # 循环匹配模式参数
            while True:
                m = re_pattern_arg.match(pattern)
                if m is None:
                    break

                pattern, arg = m.groups()
                pattern_args.insert(0, arg)

            # 获取模式对应的处理函数
            fun = self.replacements.get(pattern.lower())
            # 如果存在对应的处理函数
            if fun is not None:
                try:
                    # 尝试调用处理函数并获取替换结果
                    replacement = fun(self, *pattern_args)
                except Exception:
                    # 处理函数调用出现异常时，设置替换结果为None，并记录错误信息
                    replacement = None
                    errors.report(f"Error adding [{pattern}] to filename", exc_info=True)

                # 如果替换结果为特定值，跳过当前文本的处理
                if replacement == NOTHING_AND_SKIP_PREVIOUS_TEXT:
                    continue
                # 如果替换结果不为空，则将文本和替换结果添加到结果字符串中并继续下一次循环
                elif replacement is not None:
                    res += text + str(replacement)
                    continue

            # 如果未找到对应的处理函数，则将文本和模式添加到结果字符串中
            res += f'{text}[{pattern}]'

        # 返回处理后的结果字符串
        return res
# 获取下一个序列号以用于保存图像在指定目录中

def get_next_sequence_number(path, basename):
    """
    确定并返回在指定目录中保存图像时要使用的下一个序列号。

    序列从0开始。
    """
    result = -1
    如果基本名称不为空，则在基本名称后添加“-”
    if basename != '':
        basename = f"{basename}-"

    获取基本名称的长度
    prefix_length = len(basename)
    遍历指定路径下的所有文件
    for p in os.listdir(path):
        如果文件名以基本名称开头
        if p.startswith(basename):
            获取文件名（如果定义了基本名称，则先删除基本名称，以便序列号始终是第一个元素）
            parts = os.path.splitext(p[prefix_length:])[0].split('-')
            尝试将第一个部分转换为整数，并将其与结果比较取最大值
            try:
                result = max(int(parts[0]), result)
            如果无法转换为整数，则跳过
            except ValueError:
                pass

    返回结果加1作为下一个序列号
    return result + 1


# 保存带有生成信息的图像到文件名中

def save_image_with_geninfo(image, geninfo, filename, extension=None, existing_pnginfo=None, pnginfo_section_name='parameters'):
    """
    将图像保存到文件名中，包括生成信息作为文本信息。
    对于PNG图像，使用pnginfo_section_name参数将生成信息添加到现有的pnginfo字典中。
    对于JPG图像，没有字典，生成信息只是替换EXIF描述。
    """

    如果未指定扩展名，则根据文件名获取扩展名
    if extension is None:
        extension = os.path.splitext(filename)[1]

    获取图像格式
    image_format = Image.registered_extensions()[extension]

    如果扩展名为'.png'
    if extension.lower() == '.png':
        如果未提供现有的pnginfo字典，则设置为None
        existing_pnginfo = existing_pnginfo or {}
        如果启用了pnginfo
        if opts.enable_pnginfo:
            使用pnginfo_section_name作为键，将生成信息添加到现有的pnginfo字典中
            existing_pnginfo[pnginfo_section_name] = geninfo

        如果启用了pnginfo
        if opts.enable_pnginfo:
            创建PngInfo对象，并将现有的pnginfo字典中的键值对添加到其中
            pnginfo_data = PngImagePlugin.PngInfo()
            for k, v in (existing_pnginfo or {}).items():
                pnginfo_data.add_text(k, str(v))
        否则，将pnginfo_data设置为None
        else:
            pnginfo_data = None

        保存图像到文件中，指定格式、质量和pnginfo数据
        image.save(filename, format=image_format, quality=opts.jpeg_quality, pnginfo=pnginfo_data)
    # 如果文件扩展名为.jpg、.jpeg或.webp，则执行以下操作
    elif extension.lower() in (".jpg", ".jpeg", ".webp"):
        # 如果图片的模式为RGBA，则转换为RGB模式
        if image.mode == 'RGBA':
            image = image.convert("RGB")
        # 如果图片的模式为I;16，则进行特定的像素点处理，并转换为RGB或L模式
        elif image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("RGB" if extension.lower() == ".webp" else "L")

        # 保存图片到指定文件名，指定格式、质量和是否无损
        image.save(filename, format=image_format, quality=opts.jpeg_quality, lossless=opts.webp_lossless)

        # 如果启用PNG信息且存在生成信息，则处理EXIF信息并插入到图片中
        if opts.enable_pnginfo and geninfo is not None:
            exif_bytes = piexif.dump({
                "Exif": {
                    piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(geninfo or "", encoding="unicode")
                },
            })

            piexif.insert(exif_bytes, filename)
    # 如果文件扩展名为.gif，则保存图片到指定文件名，并添加注释信息
    elif extension.lower() == ".gif":
        image.save(filename, format=image_format, comment=geninfo)
    # 其他情况下，保存图片到指定文件名，指定格式和质量
    else:
        image.save(filename, format=image_format, quality=opts.jpeg_quality)
# 保存图像到指定路径

def save_image(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):
    """Save an image.

    Args:
        image (`PIL.Image`):
            要保存的图像。
        path (`str`):
            保存图像的目录。注意，选项 `save_to_dirs` 将使图像保存到子目录中。
        basename (`str`):
            将应用于 `filename pattern` 的基本文件名。
        seed, prompt, short_filename,
        extension (`str`):
            图像文件扩展名，默认为 `png`。
        pngsectionname (`str`):
            指定 `info` 将保存在其中的部分的名称。
        info (`str` or `PngImagePlugin.iTXt`):
            PNG 信息块。
        existing_info (`dict`):
            附加的 PNG 信息。`existing_info == {pngsectionname: info, ...}`
        no_prompt:
            TODO 我不知道它的含义。
        p (`StableDiffusionProcessing`)
        forced_filename (`str`):
            如果指定，将忽略 `basename` 和文件名模式。
        save_to_dirs (bool):
            如果为真，则图像将保存到 `path` 的子目录中。

    Returns: (fullfn, txt_fullfn)
        fullfn (`str`):
            保存图像的完整路径。
        txt_fullfn (`str` or None):
            如果为此图像保存了文本文件，则这将是其完整路径。否则为 None。
    """
    # 创建文件名生成器
    namegen = FilenameGenerator(p, seed, prompt, image)

    # WebP 和 JPG 格式的最大尺寸限制分别为 16383 和 65535。切换到具有更高限制的 PNG 格式
    # 检查图片的高度和宽度是否超过限制，并且文件扩展名为 jpg 或 jpeg 或者高度和宽度是否超过另一组限制并且文件扩展名为 webp
    if (image.height > 65535 or image.width > 65535) and extension.lower() in ("jpg", "jpeg") or (image.height > 16383 or image.width > 16383) and extension.lower() == "webp":
        # 如果图片尺寸过大，则保存为 PNG 格式
        print('Image dimensions too large; saving as PNG')
        extension = ".png"

    # 如果保存路径为空，则根据条件设置保存路径
    if save_to_dirs is None:
        save_to_dirs = (grid and opts.grid_save_to_dirs) or (not grid and opts.save_to_dirs and not no_prompt)

    # 如果需要保存到目录，则生成目录名并拼接路径
    if save_to_dirs:
        dirname = namegen.apply(opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
        path = os.path.join(path, dirname)

    # 创建目录，如果目录已存在则不报错
    os.makedirs(path, exist_ok=True)

    # 如果强制文件名为空，则根据条件设置文件名
    if forced_filename is None:
        # 根据条件设置文件名修饰
        if short_filename or seed is None:
            file_decoration = ""
        elif opts.save_to_dirs:
            file_decoration = opts.samples_filename_pattern or "[seed]"
        else:
            file_decoration = opts.samples_filename_pattern or "[seed]-[prompt_spaces]"

        file_decoration = namegen.apply(file_decoration) + suffix

        # 根据条件是否添加序号
        add_number = opts.save_images_add_number or file_decoration == ''

        # 如果需要添加序号，则生成带序号的文件名
        if file_decoration != "" and add_number:
            file_decoration = f"-{file_decoration}"

        # 如果需要添加序号，则生成带序号的文件名
        if add_number:
            basecount = get_next_sequence_number(path, basename)
            fullfn = None
            for i in range(500):
                fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
                fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
                if not os.path.exists(fullfn):
                    break
        else:
            fullfn = os.path.join(path, f"{file_decoration}.{extension}")
    else:
        # 如果有强制文件名，则使用强制文件名
        fullfn = os.path.join(path, f"{forced_filename}.{extension}")

    # 如果已存在图片信息，则使用已存在的信息
    pnginfo = existing_info or {}
    # 如果有新的信息，则更新图片信息
    if info is not None:
        pnginfo[pnginfo_section_name] = info

    # 创建图片保存参数对象
    params = script_callbacks.ImageSaveParams(image, p, fullfn, pnginfo)
    # 调用图片保存前的回调函数
    script_callbacks.before_image_saved_callback(params)
    # 从参数中获取图像对象
    image = params.image
    # 从参数中获取完整文件名
    fullfn = params.filename
    # 从 PNG 信息中获取指定部分的信息
    info = params.pnginfo.get(pnginfo_section_name, None)

    # 定义一个函数，原子性地保存图像
    def _atomically_save_image(image_to_save, filename_without_extension, extension):
        """
        保存图像为带有 .tmp 扩展名的临时文件，以避免在另一个进程检测到目录中的新图像时出现竞争条件
        """
        # 生成临时文件路径
        temp_file_path = f"{filename_without_extension}.tmp"

        # 调用函数保存图像和相关信息
        save_image_with_geninfo(image_to_save, info, temp_file_path, extension, existing_pnginfo=params.pnginfo, pnginfo_section_name=pnginfo_section_name)

        # 生成最终文件名
        filename = filename_without_extension + extension
        # 如果保存图像的替换操作不是"Replace"
        if shared.opts.save_images_replace_action != "Replace":
            n = 0
            # 如果文件已存在，则在文件名后添加序号
            while os.path.exists(filename):
                n += 1
                filename = f"{filename_without_extension}-{n}{extension}"
        # 替换临时文件为最终文件
        os.replace(temp_file_path, filename)

    # 获取完整文件名去除扩展名后的部分和扩展名
    fullfn_without_extension, extension = os.path.splitext(params.filename)
    # 如果操作系统支持 statvfs 函数
    if hasattr(os, 'statvfs'):
        # 获取文件名的最大长度
        max_name_len = os.statvfs(path).f_namemax
        # 根据文件名的最大长度截取文件名
        fullfn_without_extension = fullfn_without_extension[:max_name_len - max(4, len(extension))]
        params.filename = fullfn_without_extension + extension
        fullfn = params.filename
    # 调用保存图像的函数
    _atomically_save_image(image, fullfn_without_extension, extension)

    # 将已保存的完整文件名存储在图像对象中
    image.already_saved_as = fullfn

    # 判断图像是否超过目标尺寸
    oversize = image.width > opts.target_side_length or image.height > opts.target_side_length
    # 如果需要为4chan导出，并且图片过大或者文件大小超过设定的阈值
    if opts.export_for_4chan and (oversize or os.stat(fullfn).st_size > opts.img_downscale_threshold * 1024 * 1024):
        # 计算图片宽高比
        ratio = image.width / image.height
        resize_to = None
        # 如果图片过大且宽高比大于1
        if oversize and ratio > 1:
            resize_to = round(opts.target_side_length), round(image.height * opts.target_side_length / image.width)
        # 如果图片过大
        elif oversize:
            resize_to = round(image.width * opts.target_side_length / image.height), round(opts.target_side_length)

        # 如果需要调整大小
        if resize_to is not None:
            try:
                # 使用LANCZOS方法调整图片大小，如果图片模式为I;16可能会抛出异常
                image = image.resize(resize_to, LANCZOS)
            except Exception:
                # 如果调整大小失败，使用默认方法调整大小
                image = image.resize(resize_to)
        try:
            # 以原子方式将图片保存为缩小的JPG格式
            _atomically_save_image(image, fullfn_without_extension, ".jpg")
        except Exception as e:
            # 如果保存图片为缩小的JPG格式失败，显示错误信息
            errors.display(e, "saving image as downscaled JPG")

    # 如果需要保存文本信息并且信息不为空
    if opts.save_txt and info is not None:
        # 构建文本文件名
        txt_fullfn = f"{fullfn_without_extension}.txt"
        # 打开文本文件并写入信息
        with open(txt_fullfn, "w", encoding="utf8") as file:
            file.write(f"{info}\n")
    else:
        txt_fullfn = None

    # 调用图像保存回调函数
    script_callbacks.image_saved_callback(params)

    # 返回图片文件名和文本文件名
    return fullfn, txt_fullfn
# 定义要忽略的图片信息关键字集合
IGNORED_INFO_KEYS = {
    'jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif',
    'loop', 'background', 'timestamp', 'duration', 'progressive', 'progression',
    'icc_profile', 'chromaticity', 'photoshop',
}

# 从图片中读取信息并返回元组
def read_info_from_image(image: Image.Image) -> tuple[str | None, dict]:
    # 复制图片信息字典
    items = (image.info or {}).copy()

    # 弹出 'parameters' 字段
    geninfo = items.pop('parameters', None)

    # 如果图片信息中包含 'exif'
    if "exif" in items:
        # 获取 'exif' 数据
        exif_data = items["exif"]
        try:
            # 尝试加载 'exif' 数据
            exif = piexif.load(exif_data)
        except OSError:
            # 内存 / exif 数据无效，piexif 尝试从文件中读取
            exif = None
        # 获取 'Exif' 中的用户评论
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
        try:
            # 尝试加载用户评论
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            # 解码用户评论
            exif_comment = exif_comment.decode('utf8', errors="ignore")

        # 如果存在用户评论
        if exif_comment:
            # 将用户评论添加到图片信息字典中
            items['exif comment'] = exif_comment
            geninfo = exif_comment
    # 如果图片信息中包含 'comment'（用于 GIF）
    elif "comment" in items:
        # 解码 'comment' 字段
        geninfo = items["comment"].decode('utf8', errors="ignore")

    # 移除要忽略的信息字段
    for field in IGNORED_INFO_KEYS:
        items.pop(field, None)

    # 如果 'Software' 字段为 "NovelAI"
    if items.get("Software", None) == "NovelAI":
        try:
            # 尝试解析 JSON 格式的 'Comment' 字段
            json_info = json.loads(items["Comment"])
            # 获取 'sampler' 对应的采样器
            sampler = sd_samplers.samplers_map.get(json_info["sampler"], "Euler a")

            # 生成信息字符串
            geninfo = f"""{items["Description"]}
Negative prompt: {json_info["uc"]}
Steps: {json_info["steps"]}, Sampler: {sampler}, CFG scale: {json_info["scale"]}, Seed: {json_info["seed"]}, Size: {image.width}x{image.height}, Clip skip: 2, ENSD: 31337"""
        except Exception:
            # 报告解析 NovelAI 图片生成参数时出错
            errors.report("Error parsing NovelAI image generation parameters", exc_info=True)

    # 返回生成的信息和图片信息字典
    return geninfo, items

# 处理图片数据
def image_data(data):
    # 导入 gradio 库
    import gradio as gr

    try:
        # 打开图片数据并创建 Image 对象
        image = Image.open(io.BytesIO(data))
        # 读取图片信息并返回文本信息和空值
        textinfo, _ = read_info_from_image(image)
        return textinfo, None
    # 捕获任何异常并忽略，不做任何处理
    except Exception:
        pass

    try:
        # 尝试将数据解码为 UTF-8 编码的文本
        text = data.decode('utf8')
        # 断言文本长度小于10000
        assert len(text) < 10000
        # 返回解码后的文本和空值
        return text, None

    # 捕获任何异常并忽略，不做任何处理
    except Exception:
        pass

    # 调用 gr.update() 方法并返回结果和空值
    return gr.update(), None
# 定义函数，用于将图像中的透明部分替换为指定背景颜色，返回一个没有透明度的 RGB 模式图像
def flatten(img, bgcolor):
    # 检查图像的模式是否为 RGBA
    if img.mode == "RGBA":
        # 创建一个与原图像大小相同的 RGBA 模式图像，背景颜色为指定颜色
        background = Image.new('RGBA', img.size, bgcolor)
        # 将原图像粘贴到背景图像上，使用原图像的透明度作为掩码
        background.paste(img, mask=img)
        # 将处理后的图像赋值给原图像变量
        img = background

    # 将图像转换为 RGB 模式并返回
    return img.convert('RGB')
```