# `.\PaddleOCR\applications\PCB字符识别\gen_data\gen.py`

```
# 版权声明
#
# 版权所有 (c) 2020 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言下的权限和限制。
"""
# 引用来源
# https://github.com/zcswdt/Color_OCR_image_generator

import os
import random
from PIL import Image, ImageDraw, ImageFont
import json
import argparse

def get_char_lines(txt_root_path):
    """
    desc:获取语料行
    """
    # 获取指定路径下的所有文本文件
    txt_files = os.listdir(txt_root_path)
    char_lines = []
    for txt in txt_files:
        # 逐行读取文本文件内容
        f = open(os.path.join(txt_root_path, txt), mode='r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        for line in lines:
            char_lines.append(line.strip())
        return char_lines

def get_horizontal_text_picture(image_file, chars, fonts_list, cf):
    """
    desc:生成水平文本图片
    """
    # 打开图像文件
    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_w, img_h = img.size

    # 随机选择字体
    font_path = random.choice(fonts_list)
    # 随机选择字体大小
    font_size = random.randint(cf.font_min_size, cf.font_max_size)
    font = ImageFont.truetype(font_path, font_size)

    ch_w = []
    ch_h = []
    for ch in chars:
        # 获取字符的边界框
        left, top, right, bottom = font.getbbox(ch)
        wt, ht = right - left, bottom - top
        ch_w.append(wt)
        ch_h.append(ht)
    f_w = sum(ch_w)
    f_h = max(ch_h)

    # 添加空格
    char_space_width = max(ch_w)
    f_w += (char_space_width * (len(chars) - 1))
    # 生成随机的 x1 坐标，范围在图片宽度减去字体宽度内
    x1 = random.randint(0, img_w - f_w)
    # 生成随机的 y1 坐标，范围在图片高度减去字体高度内
    y1 = random.randint(0, img_h - f_h)
    # 计算 x2 坐标
    x2 = x1 + f_w
    # 计算 y2 坐标
    y2 = y1 + f_h

    # 设置裁剪区域的左上角坐标
    crop_y1 = y1
    crop_x1 = x1
    # 设置裁剪区域的右下角坐标
    crop_y2 = y2
    crop_x2 = x2

    # 初始化最佳颜色为黑色
    best_color = (0, 0, 0)
    # 创建一个可以在图像上绘制的对象
    draw = ImageDraw.Draw(img)
    # 遍历字符列表，并在图像上绘制每个字符
    for i, ch in enumerate(chars):
        draw.text((x1, y1), ch, best_color, font=font)
        # 更新 x1 坐标，考虑字符宽度和字符间距
        x1 += (ch_w[i] + char_space_width)
    # 根据裁剪区域坐标裁剪图像
    crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    # 返回裁剪后的图像和字符列表
    return crop_img, chars
# 生成垂直文本图片
def get_vertical_text_picture(image_file, chars, fonts_list, cf):
    """
    desc:gen vertical text picture
    """
    # 打开图片文件
    img = Image.open(image_file)
    # 如果图片模式不是 RGB，则转换为 RGB 模式
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # 获取图片宽度和高度
    img_w, img_h = img.size
    # 随机选择字体文件路径
    font_path = random.choice(fonts_list)
    # 随机选择字体大小
    font_size = random.randint(cf.font_min_size, cf.font_max_size)
    # 使用字体文件路径和字体大小创建字体对象
    font = ImageFont.truetype(font_path, font_size)

    # 计算每个字符的宽度和高度
    ch_w = []
    ch_h = []
    for ch in chars:
        left, top, right, bottom = font.getbbox(ch)
        wt, ht = right - left, bottom - top
        ch_w.append(wt)
        ch_h.append(ht)
    # 获取字符中最大的宽度和总高度
    f_w = max(ch_w)
    f_h = sum(ch_h)

    # 随机生成文本图片的起始坐标和结束坐标
    x1 = random.randint(0, img_w - f_w)
    y1 = random.randint(0, img_h - f_h)
    x2 = x1 + f_w
    y2 = y1 + f_h

    crop_y1 = y1
    crop_x1 = x1
    crop_y2 = y2
    crop_x2 = x2

    best_color = (0, 0, 0)
    draw = ImageDraw.Draw(img)
    i = 0
    # 在图片上绘制每个字符
    for ch in chars:
        draw.text((x1, y1), ch, best_color, font=font)
        y1 = y1 + ch_h[i]
        i = i + 1
    # 裁剪图片
    crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    # 将裁剪后的图片旋转 90 度
    crop_img = crop_img.transpose(Image.ROTATE_90)
    return crop_img, chars

# 获取所有字体文件路径
def get_fonts(fonts_path):
    """
    desc: get all fonts
    """
    # 获取字体文件夹下所有文件名
    font_files = os.listdir(fonts_path)
    fonts_list=[]
    # 遍历所有字体文件，获取完整的字体文件路径
    for font_file in font_files:
        font_path=os.path.join(fonts_path, font_file)
        fonts_list.append(font_path)
    return fonts_list

if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数：生成图片的数量
    parser.add_argument('--num_img', type=int, default=30, help="Number of images to generate")
    # 添加参数：字体最小大小
    parser.add_argument('--font_min_size', type=int, default=11)
    # 添加参数：字体最大大小
    parser.add_argument('--font_max_size', type=int, default=12,
                        help="Help adjust the size of the generated text and the size of the picture")
    # 添加命令行参数'--bg_path'，指定生成的文本图片将粘贴到此文件夹中的图片上
    parser.add_argument('--bg_path', type=str, default='./background',
                        help='The generated text pictures will be pasted onto the pictures of this folder')
    # 添加命令行参数'--det_bg_path'，指定生成的文本图片将使用此文件夹中的图片作为背景
    parser.add_argument('--det_bg_path', type=str, default='./det_background',
                        help='The generated text pictures will use the pictures of this folder as the background')
    # 添加命令行参数'--fonts_path'，指定用于生成图片的字体
    parser.add_argument('--fonts_path', type=str, default='../../StyleText/fonts',
                        help='The font used to generate the picture')
    # 添加命令行参数'--corpus_path'，指定用于生成文本图片的语料库
    parser.add_argument('--corpus_path', type=str, default='./corpus',
                        help='The corpus used to generate the text picture')
    # 添加命令行参数'--output_dir'，指定图片保存的目录
    parser.add_argument('--output_dir', type=str, default='./output/', help='Images save dir')


    # 解析命令行参数
    cf = parser.parse_args()
    # 如果输出目录不存在，则创建
    if not os.path.exists(cf.output_dir):
        os.mkdir(cf.output_dir)

    # 获取语料库
    txt_root_path = cf.corpus_path
    char_lines = get_char_lines(txt_root_path=txt_root_path)

    # 获取所有字体
    fonts_path = cf.fonts_path
    fonts_list = get_fonts(fonts_path)

    # 获取背景图片
    img_root_path = cf.bg_path
    imnames=os.listdir(img_root_path)
    
    # 获取检测背景图片
    det_bg_path = cf.det_bg_path
    bg_pics = os.listdir(det_bg_path)

    # 打开OCR检测文件
    det_val_file = open(cf.output_dir + 'det_gt_val.txt', 'w', encoding='utf-8')
    det_train_file = open(cf.output_dir + 'det_gt_train.txt', 'w', encoding='utf-8')
    # 检测图片保存目录
    det_save_dir = 'imgs/'
    if not os.path.exists(cf.output_dir + det_save_dir):
        os.mkdir(cf.output_dir + det_save_dir)
    det_val_save_dir = 'imgs_val/'
    if not os.path.exists(cf.output_dir + det_val_save_dir):
        os.mkdir(cf.output_dir + det_val_save_dir)

    # 打开OCR识别文件
    rec_val_file = open(cf.output_dir + 'rec_gt_val.txt', 'w', encoding='utf-8')
    rec_train_file = open(cf.output_dir + 'rec_gt_train.txt', 'w', encoding='utf-8')
    # 识别图片保存目录
    rec_save_dir = 'rec_imgs/'
    # 如果输出目录下的记录保存目录不存在，则创建该目录
    if not os.path.exists(cf.output_dir + rec_save_dir):
        os.mkdir(cf.output_dir + rec_save_dir)
    # 设置验证图像保存目录名称
    rec_val_save_dir = 'rec_imgs_val/'
    # 如果输出目录下的验证图像保存目录不存在，则创建该目录
    if not os.path.exists(cf.output_dir + rec_val_save_dir):
        os.mkdir(cf.output_dir + rec_val_save_dir)

    # 计算验证数据集所占比例
    val_ratio = cf.num_img * 0.2  # val dataset ratio

    # 打印开始生成数据集的提示信息
    print('start generating...')
```