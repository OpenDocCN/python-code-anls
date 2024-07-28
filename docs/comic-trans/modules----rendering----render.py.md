# `.\comic-translate\modules\rendering\render.py`

```py
# 导入必要的库和模块
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from typing import Tuple, List
from .hyphen_textwrap import wrap as hyphen_wrap  # 导入文本换行模块
from ..utils.textblock import TextBlock  # 导入文本块工具类
from ..detection import make_bubble_mask, bubble_interior_bounds  # 导入气泡检测相关函数
from ..utils.textblock import adjust_blks_size  # 导入调整文本块大小的函数

def cv2_to_pil(cv2_image: np.ndarray):
    # 将 OpenCV 图像从 BGR 转换为 RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # 将 NumPy 数组转换为 PIL 图像
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def pil_to_cv2(pil_image: Image):
    # 将 PIL 图像转换为 NumPy 数组
    numpy_image = np.array(pil_image)
    
    # PIL 图像默认为 RGB，OpenCV 使用 BGR，因此需要转换颜色空间
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    return cv2_image

def pil_word_wrap(image: Image, tbbox_top_left: Tuple, font_pth: str, init_font_size, text: str, roi_width, roi_height, align: str, spacing):
    """将长文本分行并减小字体大小，直到所有文本适合指定的边界框内。"""
    mutable_message = text
    font_size = init_font_size
    font = ImageFont.truetype(font_pth, font_size)

    def eval_metrics(txt, font):
        """快速计算文本的宽度和高度的辅助函数。"""
        (left, top, right, bottom) = ImageDraw.Draw(image).multiline_textbbox(xy=tbbox_top_left, text=txt, font=font, align=align, spacing=spacing)
        return (right-left, bottom-top)

    while font_size > 1:
        font = font.font_variant(size=font_size)
        width, height = eval_metrics(mutable_message, font)
        if height > roi_height:
            font_size -= 0.75  # 减小字体大小
            mutable_message = text  # 恢复原始文本
        elif width > roi_width:
            columns = len(mutable_message)
            while columns > 0:
                columns -= 1
                if columns == 0:
                    break
                # 使用文本换行模块对文本进行换行处理
                mutable_message = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True)) 
                wrapped_width, _ = eval_metrics(mutable_message, font)
                if wrapped_width <= roi_width:
                    break
            if columns < 1:
                font_size -= 0.75  # 减小字体大小
                mutable_message = text  # 恢复原始文本
        else:
            break

    return mutable_message, font_size

def draw_text(image: np.ndarray, blk_list: List[TextBlock], font_pth: str, init_font_size, colour: str = "#000"):
    image = cv2_to_pil(image)  # 转换为 PIL 图像
    draw = ImageDraw.Draw(image)  # 创建 ImageDraw 对象

    font = ImageFont.truetype(font_pth, size=init_font_size)  # 使用指定字体和大小创建字体对象
    # 对每个文本块进行处理，blk_list 是文本块列表
    for blk in blk_list:
        # 从文本块中获取位置和尺寸信息
        x1, y1, width, height = blk.xywh
        # 文本框左上角的坐标
        tbbox_top_left = (x1, y1)

        # 获取文本块的翻译文本和字体大小
        translation = blk.translation
        # 如果没有翻译文本或者翻译文本长度为1，则跳过此文本块
        if not translation or len(translation) == 1:
            continue

        # 使用 PIL 进行文字自动换行处理，并返回调整后的翻译文本和字体大小
        translation, font_size = pil_word_wrap(image, tbbox_top_left, font_pth, init_font_size, translation, width, height, align=blk.alignment, spacing=blk.line_spacing)
        
        # 根据调整后的字体大小设置字体变体
        font = font.font_variant(size=font_size)

        # 字体检测的变通方法。在文本周围绘制白色偏移以增强字体识别效果
        offsets = [(dx, dy) for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2) if dx != 0 or dy != 0]
        # 在文本周围绘制带有白色偏移的多行文本
        for dx, dy in offsets:
            draw.multiline_text((tbbox_top_left[0] + dx, tbbox_top_left[1] + dy), translation, font=font, fill="#FFF", align=blk.alignment, spacing=1)
        # 绘制多行文本到指定位置，使用指定的颜色、字体和对齐方式
        draw.multiline_text(tbbox_top_left, translation, colour, font, align=blk.alignment, spacing=1)
        
    # 将 PIL 图像转换为 OpenCV 图像格式
    image = pil_to_cv2(image)
    # 将图像从 BGR 转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 返回处理后的图像
    return image
# 根据传入的文本块列表、原始图像和修复后的图像，确定最佳的文本渲染区域
def get_best_render_area(blk_list: List[TextBlock], img, inpainted_img):
    # 使用对话框检测来找到最佳的文本渲染区域
    for blk in blk_list:
        # 如果文本块属于文本气泡类
        if blk.text_class == 'text_bubble':
            # 获取文本气泡的边界框坐标
            bx1, by1, bx2, by2 = blk.bubble_xyxy
            # 从修复后的图像中提取出文本气泡的干净帧
            bubble_clean_frame = inpainted_img[by1:by2, bx1:bx2]
            # 创建文本气泡的遮罩
            bubble_mask = make_bubble_mask(bubble_clean_frame)
            # 计算文本绘制的边界
            text_draw_bounds = bubble_interior_bounds(bubble_mask)

            # 获取调整后的文本绘制边界坐标
            bdx1, bdy1, bdx2, bdy2 = text_draw_bounds

            # 将文本绘制边界坐标转换回原始图像坐标系
            bdx1 += bx1
            bdy1 += by1
            bdx2 += bx1
            bdy2 += by1

            # 如果文本块的源语言为日语
            if blk.source_lang == 'ja':
                # 更新文本块的坐标为调整后的文本绘制边界
                blk.xyxy[:] = [bdx1, bdy1, bdx2, bdy2]
                # 调整文本块的大小
                adjust_blks_size(blk_list, img.shape, -5, -5)
            else:
                # 否则，获取原始文本块的坐标
                tx1, ty1, tx2, ty2 = blk.xyxy

                # 计算文本块的新水平范围
                nx1 = max(bdx1, tx1)
                nx2 = min(bdx2, tx2)
                
                # 更新文本块的坐标为新的水平范围，保持垂直范围不变
                blk.xyxy[:] = [nx1, ty1, nx2, ty2]

    # 返回更新后的文本块列表
    return blk_list
```