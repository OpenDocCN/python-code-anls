# `stable-diffusion-webui\modules\textual_inversion\image_embedding.py`

```py
# 导入必要的库
import base64
import json
import warnings

import numpy as np
import zlib
from PIL import Image, ImageDraw
import torch

# 自定义 JSON 编码器，用于处理 torch.Tensor 对象
class EmbeddingEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {'TORCHTENSOR': obj.cpu().detach().numpy().tolist()}
        return json.JSONEncoder.default(self, obj)

# 自定义 JSON 解码器，用于处理 torch.Tensor 对象
class EmbeddingDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, *args, object_hook=self.object_hook, **kwargs)

    def object_hook(self, d):
        if 'TORCHTENSOR' in d:
            return torch.from_numpy(np.array(d['TORCHTENSOR']))
        return d

# 将数据编码为 base64 字符串
def embedding_to_b64(data):
    d = json.dumps(data, cls=EmbeddingEncoder)
    return base64.b64encode(d.encode())

# 从 base64 字符串解码数据
def embedding_from_b64(data):
    d = base64.b64decode(data)
    return json.loads(d, cls=EmbeddingDecoder)

# 线性同余生成器函数
def lcg(m=2**32, a=1664525, c=1013904223, seed=0):
    while True:
        seed = (a * seed + c) % m
        yield seed % 255

# 对块数据进行异或操作
def xor_block(block):
    g = lcg()
    randblock = np.array([next(g) for _ in range(np.product(block.shape))]).astype(np.uint8).reshape(block.shape)
    return np.bitwise_xor(block.astype(np.uint8), randblock & 0x0F)

# 对块数据进行样式处理
def style_block(block, sequence):
    im = Image.new('RGB', (block.shape[1], block.shape[0]))
    draw = ImageDraw.Draw(im)
    i = 0
    for x in range(-6, im.size[0], 8):
        for yi, y in enumerate(range(-6, im.size[1], 8)):
            offset = 0
            if yi % 2 == 0:
                offset = 4
            shade = sequence[i % len(sequence)]
            i += 1
            draw.ellipse((x+offset, y, x+6+offset, y+6), fill=(shade, shade, shade))

    fg = np.array(im).astype(np.uint8) & 0xF0

    return block ^ fg

# 将数据嵌入到图像中
def insert_image_data_embed(image, data):
    d = 3
    # 压缩数据并转换为 numpy 数组
    data_compressed = zlib.compress(json.dumps(data, cls=EmbeddingEncoder).encode(), level=9)
    data_np_ = np.frombuffer(data_compressed, np.uint8).copy()
    # 将数据向右移动4位，相当于除以16，得到高位数据
    data_np_high = data_np_ >> 4
    # 将数据与0x0F进行按位与操作，得到低位数据
    data_np_low = data_np_ & 0x0F

    # 获取图像的高度
    h = image.size[1]
    # 计算下一个大小，使得低位数据的行数能够整除图像高度h
    next_size = data_np_low.shape[0] + (h-(data_np_low.shape[0] % h))
    # 计算下一个大小，使得低位数据的行数能够整除图像高度h以及图像宽度d
    next_size = next_size + ((h*d)-(next_size % (h*d)))

    # 调整低位数据的大小，使其满足要求
    data_np_low = np.resize(data_np_low, next_size)
    # 将低位数据重新组织成(h, -1, d)的形状
    data_np_low = data_np_low.reshape((h, -1, d))

    # 调整高位数据的大小，使其满足要求
    data_np_high = np.resize(data_np_high, next_size)
    # 将高位数据重新组织成(h, -1, d)的形状
    data_np_high = data_np_high.reshape((h, -1, d))

    # 获取数据中的边缘样式
    edge_style = list(data['string_to_param'].values())[0].cpu().detach().numpy().tolist()[0][:1024]
    # 将边缘样式归一化到0-255范围，并转换为无符号整数类型
    edge_style = (np.abs(edge_style)/np.max(np.abs(edge_style))*255).astype(np.uint8)

    # 对低位数据应用样式块和异或块处理
    data_np_low = style_block(data_np_low, sequence=edge_style)
    data_np_low = xor_block(data_np_low)
    # 对高位数据应用样式块和异或块处理
    data_np_high = style_block(data_np_high, sequence=edge_style[::-1])
    data_np_high = xor_block(data_np_high)

    # 将低位数据转换为图像
    im_low = Image.fromarray(data_np_low, mode='RGB')
    # 将高位数据转换为图像
    im_high = Image.fromarray(data_np_high, mode='RGB')

    # 创建一个背景图像，将低位图像、原始图像和高位图像拼接在一起
    background = Image.new('RGB', (image.size[0]+im_low.size[0]+im_high.size[0]+2, image.size[1]), (0, 0, 0))
    background.paste(im_low, (0, 0))
    background.paste(image, (im_low.size[0]+1, 0))
    background.paste(im_high, (im_low.size[0]+1+image.size[0]+1, 0))

    # 返回拼接后的背景图像
    return background
# 根据给定的阈值将图像裁剪为黑色背景
def crop_black(img, tol=0):
    # 创建一个布尔掩码，表示图像中是否所有通道的像素值都大于阈值
    mask = (img > tol).all(2)
    # 获取水平和垂直方向上的掩码
    mask0, mask1 = mask.any(0), mask.any(1)
    # 获取裁剪后的图像的起始和结束列、行索引
    col_start, col_end = mask0.argmax(), mask.shape[1]-mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), mask.shape[0]-mask1[::-1].argmax()
    # 返回裁剪后的图像
    return img[row_start:row_end, col_start:col_end]


# 从嵌入式图像中提取数据
def extract_image_data_embed(image):
    d = 3
    # 将图像转换为 RGB 模式，获取像素数据并重塑为三维数组
    outarr = crop_black(np.array(image.convert('RGB').getdata()).reshape(image.size[1], image.size[0], d).astype(np.uint8)) & 0x0F
    # 查找全黑列的索引
    black_cols = np.where(np.sum(outarr, axis=(0, 2)) == 0)
    # 如果找不到数据块，则打印消息并返回 None
    if black_cols[0].shape[0] < 2:
        print('No Image data blocks found.')
        return None

    # 提取数据块的下半部分和上半部分
    data_block_lower = outarr[:, :black_cols[0].min(), :].astype(np.uint8)
    data_block_upper = outarr[:, black_cols[0].max()+1:, :].astype(np.uint8)

    # 对数据块进行异或操作
    data_block_lower = xor_block(data_block_lower)
    data_block_upper = xor_block(data_block_upper)

    # 合并数据块并解压缩
    data_block = (data_block_upper << 4) | (data_block_lower)
    data_block = data_block.flatten().tobytes()

    data = zlib.decompress(data_block)
    # 将解压缩后的数据转换为 JSON 格式
    return json.loads(data, cls=EmbeddingDecoder)


# 在图像上叠加标题和页脚信息
def caption_image_overlay(srcimage, title, footerLeft, footerMid, footerRight, textfont=None):
    # 导入获取字体的函数
    from modules.images import get_font
    # 如果传入了 textfont 参数，则发出警告
    if textfont:
        warnings.warn(
            'passing in a textfont to caption_image_overlay is deprecated and does nothing',
            DeprecationWarning,
            stacklevel=2,
        )
    # 导入 cos 函数
    from math import cos

    # 复制源图像
    image = srcimage.copy()
    fontsize = 32
    factor = 1.5
    # 创建渐变图像
    gradient = Image.new('RGBA', (1, image.size[1]), color=(0, 0, 0, 0))
    # 为每一行像素设置渐变值
    for y in range(image.size[1]):
        mag = 1-cos(y/image.size[1]*factor)
        mag = max(mag, 1-cos((image.size[1]-y)/image.size[1]*factor*1.1))
        gradient.putpixel((0, y), (0, 0, 0, int(mag*255)))
    # 将渐变图像叠加到源图像上
    image = Image.alpha_composite(image.convert('RGBA'), gradient.resize(image.size))

    # 创建图像绘制对象
    draw = ImageDraw.Draw(image)

    # 获取指定字体大小的字体对象
    font = get_font(fontsize)
    padding = 10
    # 获取标题文本的边界框信息，包括左上角和右下角坐标以及宽高
    _, _, w, h = draw.textbbox((0, 0), title, font=font)
    # 根据图片宽度、边距和标题文本宽度计算合适的字体大小
    fontsize = min(int(fontsize * (((image.size[0]*0.75)-(padding*4))/w)), 72)
    # 根据新的字体大小获取字体对象
    font = get_font(fontsize)
    # 获取标题文本的边界框信息
    _, _, w, h = draw.textbbox((0, 0), title, font=font)
    # 在图片上绘制标题文本
    draw.text((padding, padding), title, anchor='lt', font=font, fill=(255, 255, 255, 230))

    # 获取左侧页脚文本的边界框信息
    _, _, w, h = draw.textbbox((0, 0), footerLeft, font=font)
    # 根据图片宽度、边距和左侧页脚文本宽度计算合适的字体大小
    fontsize_left = min(int(fontsize * (((image.size[0]/3)-(padding))/w)), 72
    # 获取中间页脚文本的边界框信息
    _, _, w, h = draw.textbbox((0, 0), footerMid, font=font)
    # 根据图片宽度、边距和中间页脚文本宽度计算合适的字体大小
    fontsize_mid = min(int(fontsize * (((image.size[0]/3)-(padding))/w)), 72
    # 获取右侧页脚文本的边界框信息
    _, _, w, h = draw.textbbox((0, 0), footerRight, font=font)
    # 根据图片宽度、边距和右侧页脚文本宽度计算合适的字体大小
    fontsize_right = min(int(fontsize * (((image.size[0]/3)-(padding))/w)), 72

    # 根据左侧、中间和右侧页脚文本的字体大小选择最小的作为最终字体大小
    font = get_font(min(fontsize_left, fontsize_mid, fontsize_right))

    # 在图片底部左侧绘制左侧页脚文本
    draw.text((padding, image.size[1]-padding), footerLeft, anchor='ls', font=font, fill=(255, 255, 255, 230))
    # 在图片底部中间绘制中间页脚文本
    draw.text((image.size[0]/2, image.size[1]-padding), footerMid, anchor='ms', font=font, fill=(255, 255, 255, 230))
    # 在图片底部右侧绘制右侧页脚文本
    draw.text((image.size[0]-padding, image.size[1]-padding), footerRight, anchor='rs', font=font, fill=(255, 255, 255, 230))

    # 返回添加文本后的图片
    return image
# 如果当前脚本作为主程序运行
if __name__ == '__main__':

    # 打开测试嵌入图像文件
    testEmbed = Image.open('test_embedding.png')
    # 从图像中提取嵌入的数据
    data = extract_image_data_embed(testEmbed)
    # 断言提取的数据不为空
    assert data is not None

    # 从 Base64 编码的字符串中提取嵌入的数据
    data = embedding_from_b64(testEmbed.text['sd-ti-embedding'])
    # 断言提取的数据不为空
    assert data is not None

    # 创建一个新的 RGBA 模式图像
    image = Image.new('RGBA', (512, 512), (255, 255, 200, 255))
    # 在图像上添加标题和页脚信息
    cap_image = caption_image_overlay(image, 'title', 'footerLeft', 'footerMid', 'footerRight')

    # 创建一个测试嵌入数据字典
    test_embed = {'string_to_param': {'*': torch.from_numpy(np.random.random((2, 4096)))}}

    # 在图像中插入嵌入数据
    embedded_image = insert_image_data_embed(cap_image, test_embed)

    # 从嵌入图像中提取数据
    retrived_embed = extract_image_data_embed(embedded_image)

    # 断言提取的数据与测试数据相同
    assert str(retrived_embed) == str(test_embed)

    # 再次在图像中插入嵌入数据
    embedded_image2 = insert_image_data_embed(cap_image, retrived_embed)

    # 断言两次插入的图像数据相同
    assert embedded_image == embedded_image2

    # 创建一个线性同余生成器
    g = lcg()
    # 生成共享的随机数列表
    shared_random = np.array([next(g) for _ in range(100)]).astype(np.uint8).tolist()

    # 参考的随机数列表
    reference_random = [253, 242, 127,  44, 157,  27, 239, 133,  38,  79, 167,   4, 177,
                         95, 130,  79,  78,  14,  52, 215, 220, 194, 126,  28, 240, 179,
                        160, 153, 149,  50, 105,  14,  21, 218, 199,  18,  54, 198, 193,
                         38, 128,  19,  53, 195, 124,  75, 205,  12,   6, 145,   0,  28,
                         30, 148,   8,  45, 218, 171,  55, 249,  97, 166,  12,  35,   0,
                         41, 221, 122, 215, 170,  31, 113, 186,  97, 119,  31,  23, 185,
                         66, 140,  30,  41,  37,  63, 137, 109, 216,  55, 159, 145,  82,
                         204, 86,  73, 222,  44, 198, 118, 240,  97]

    # 断言生成的随机数列表与参考的随机数列表相同
    assert shared_random == reference_random

    # 计算生成的随机数列表的总和
    hunna_kay_random_sum = sum(np.array([next(g) for _ in range(100000)]).astype(np.uint8).tolist())

    # 断言生成的随机数列表的总和为指定值
    assert 12731374 == hunna_kay_random_sum
```