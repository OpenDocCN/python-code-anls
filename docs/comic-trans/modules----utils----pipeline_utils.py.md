# `.\comic-translate\modules\utils\pipeline_utils.py`

```py
# 导入所需模块
import cv2
import numpy as np
import os
import base64
# 导入本地模块和函数
from .textblock import TextBlock, sort_textblock_rectangles
from ..detection import does_rectangle_fit, do_rectangles_overlap
# 导入类型提示
from typing import List
# 导入图像修复算法
from ..inpainting.lama import LaMa
from ..inpainting.schema import Config
# 导入用户界面消息模块
from app.ui.messages import Messages

# 支持的语言代码映射
language_codes = {
    "Korean": "ko",
    "Japanese": "ja",
    "Chinese": "zh",
    "Simplified Chinese": "zh-CN",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "Russian": "ru",
    "French": "fr",
    "German": "de",
    "Dutch": "nl",
    "Spanish": "es",
    "Italian": "it",
    "Turkish": "tr",
    "Polish": "pl",
    "Portuguese": "pt",
    "Brazilian Portuguese": "pt-br",
}

# 图像修复算法映射
inpaint_map = {
    "LaMa": LaMa
}

# 根据页面设置获取配置信息
def get_config(settings_page):
    strategy_settings = settings_page.get_hd_strategy_settings()
    if strategy_settings['strategy'] == settings_page.ui.tr("Resize"):
        # 创建一个调整大小的配置对象
        config = Config(hd_strategy="Resize", hd_strategy_resize_limit = strategy_settings['resize_limit'])
    elif strategy_settings['strategy'] == settings_page.ui.tr("Crop"):
        # 创建一个裁剪的配置对象
        config = Config(hd_strategy="Crop", hd_strategy_crop_margin = strategy_settings['crop_margin'],
                        hd_strategy_crop_trigger_size = strategy_settings['crop_trigger_size'])
    else:
        # 创建一个原始尺寸的配置对象
        config = Config(hd_strategy="Original")

    return config

# 根据语言名称获取语言代码
def get_language_code(lng: str):
    lng_cd = language_codes.get(lng, None)
    return lng_cd

# 将 RGBA 列表转换为十六进制颜色表示
def rgba2hex(rgba_list):
    r,g,b,a = [int(num) for num in rgba_list]
    return "#{:02x}{:02x}{:02x}{:02x}".format(r, g, b, a)

# 编码图像数组为 base64 字符串
def encode_image_array(img_array: np.ndarray):
    _, img_bytes = cv2.imencode('.png', img_array)
    return base64.b64encode(img_bytes).decode('utf-8')

# 将多个文本块列表转换为单个文本块列表
def lists_to_blk_list(blk_list: List[TextBlock], texts_bboxes: List, texts_string: List):

    for blk in blk_list:
        blk_entries = []
        group = list(zip(texts_bboxes, texts_string))  

        # 遍历文本框和对应文本的组合
        for line, text in group:
            if blk.bubble_xyxy is not None:
                # 如果存在气泡边界框，检查文本框是否适合或重叠
                if does_rectangle_fit(blk.bubble_xyxy, line):
                    blk_entries.append((line, text))  
                elif do_rectangles_overlap(blk.bubble_xyxy, line):
                    blk_entries.append((line, text)) 
            elif do_rectangles_overlap(blk.xyxy, line):
                # 检查文本框是否与区域边界框重叠
                blk_entries.append((line, text)) 

        # 对文本条目进行排序和连接
        sorted_entries = sort_textblock_rectangles(blk_entries, blk.source_lang_direction)
        if blk.source_lang in ['ja', 'zh']:
            blk.text = ''.join(text for bbox, text in sorted_entries)
        else:
            blk.text = ' '.join(text for bbox, text in sorted_entries)

    return blk_list

# 确保坐标在图像边界内
def ensure_within_bounds(coords, im_width, im_height, width_expansion_percentage: int, height_expansion_percentage: int):
    x1, y1, x2, y2 = coords

    width = x2 - x1
    height = y2 - y1
    # 根据给定的宽度扩展百分比计算宽度的扩展偏移量
    width_expansion_offset = int((width * width_expansion_percentage) / 100)
    # 根据给定的高度扩展百分比计算高度的扩展偏移量
    height_expansion_offset = int((height * height_expansion_percentage) / 100)

    # 计算扩展后的左上角 x 坐标，确保不超出图像边界
    x1 = max(x1 - width_expansion_offset, 0)
    # 计算扩展后的右下角 x 坐标，确保不超出图像边界
    x2 = min(x2 + width_expansion_offset, im_width)
    # 计算扩展后的左上角 y 坐标，确保不超出图像边界
    y1 = max(y1 - height_expansion_offset, 0)
    # 计算扩展后的右下角 y 坐标，确保不超出图像边界
    y2 = min(y2 + height_expansion_offset, im_height)

    # 返回经过边界扩展后的坐标范围
    return x1, y1, x2, y2
# 根据输入的图像和文本块列表生成遮罩
def generate_mask(img: np.ndarray, blk_list: List[TextBlock], default_kernel_size=5):
    # 获取图像的高度、宽度和通道数
    h, w, c = img.shape
    # 创建一个初始全黑的遮罩
    mask = np.zeros((h, w), dtype=np.uint8)

    # 遍历每一个文本块
    for blk in blk_list:
        # 获取文本块的轮廓点
        seg = blk.segm_pts

        # 如果轮廓点为空，跳过该文本块
        if seg.size == 0:
            continue
        
        # 如果文本块的来源语言是英语，将默认核大小设置为1
        if blk.source_lang == 'en':
            default_kernel_size = 1
        
        # 默认的核大小
        kernel_size = default_kernel_size
        
        # 如果文本块属于文本气泡类别
        if blk.text_class == 'text_bubble':
            # 访问文本气泡的边界框坐标
            bbox = blk.bubble_xyxy
            # 计算遮罩到边界框边缘的最小距离
            min_distance_to_bbox = min(
                np.min(seg[:, 0]) - bbox[0],  # 左侧
                bbox[2] - np.max(seg[:, 0]),  # 右侧
                np.min(seg[:, 1]) - bbox[1],  # 上侧
                bbox[3] - np.max(seg[:, 1])   # 下侧
            )
            # 根据需要调整核大小
            if default_kernel_size >= min_distance_to_bbox:
                kernel_size = max(1, int(min_distance_to_bbox - (0.2 * min_distance_to_bbox)))

        # 创建一个基于核大小的膨胀核
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 创建单个文本块的遮罩并进行膨胀
        single_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(single_mask, [seg], 255)
        single_mask = cv2.dilate(single_mask, kernel, iterations=1)

        # 将膨胀后的遮罩与全局遮罩进行按位或运算
        mask = cv2.bitwise_or(mask, single_mask)
        np.expand_dims(mask, axis=-1)

    # 返回最终合并的遮罩
    return mask

# 验证 OCR 设置是否有效
def validate_ocr(main_page, source_lang):
    # 获取主页面的设置页面
    settings_page = main_page.settings_page
    # 获取所有设置项
    settings = settings_page.get_all_settings()

    # 映射源语言到标准英语名称
    source_lang_en = main_page.lang_mapping.get(source_lang, source_lang)

    # 获取当前使用的 OCR 工具
    ocr_tool = settings['tools']['ocr']

    # 验证 Microsoft OCR 的 API 密钥是否存在
    if ocr_tool == settings_page.ui.tr("Microsoft OCR") and not settings["credentials"]["Microsoft Azure"]["api_key_ocr"]:
        Messages.show_api_key_ocr_error(main_page)
        return False
    
    # 验证 Google Cloud Vision 的 API 密钥是否存在
    if ocr_tool == settings_page.ui.tr("Google Cloud Vision") and not settings["credentials"]["Google Cloud"]["api_key"]:
        Messages.show_api_key_ocr_error(main_page)
        return False

    # 验证 Microsoft OCR 的终端点是否存在
    if ocr_tool == settings_page.ui.tr('Microsoft OCR') and not settings['credentials']['Microsoft Azure']['endpoint']:
        Messages.show_endpoint_url_error(main_page)
        return False

    # 验证 GPT OCR 的设置是否存在，适用于指定的源语言
    if source_lang_en in ["French", "German", "Dutch", "Russian", "Spanish", "Italian"]:
        if ocr_tool == settings_page.ui.tr('Default') and not settings['credentials']['Open AI GPT']['api_key']:
            Messages.show_api_key_ocr_gpt4v_error(main_page)
            return False
    
    # 如果所有验证均通过，返回 True
    return True

# 验证翻译器设置是否有效
def validate_translator(main_page, source_lang, target_lang):
    # 获取主页面的设置页面对象
    settings_page = main_page.settings_page
    # 获取所有设置信息
    settings = settings_page.get_all_settings()

    # 获取翻译工具的设置信息
    translator_tool = settings['tools']['translator']

    # 验证翻译工具的 API 密钥

    # 如果翻译工具为 DeepL 并且没有设置 DeepL 的 API 密钥，则显示 API 密钥错误信息并返回 False
    if translator_tool == settings_page.ui.tr("DeepL") and not settings["credentials"]["DeepL"]["api_key"]:
        Messages.show_api_key_translator_error(main_page)
        return False
    
    # 如果翻译工具为 Microsoft Translator 并且没有设置 Microsoft Azure 的 API 密钥，则显示 API 密钥错误信息并返回 False
    if translator_tool == settings_page.ui.tr("Microsoft Translator") and not settings["credentials"]["Microsoft Azure"]["api_key_translator"]:
        Messages.show_api_key_translator_error(main_page)
        return False

    # 如果翻译工具为 Yandex 并且没有设置 Yandex 的 API 密钥，则显示 API 密钥错误信息并返回 False
    if translator_tool == settings_page.ui.tr("Yandex") and not settings["credentials"]["Yandex"]["api_key"]:
        Messages.show_api_key_translator_error(main_page)
        return False
    
    # 如果翻译工具包含 'GPT' 并且没有设置 Open AI GPT 的 API 密钥，则显示 API 密钥错误信息并返回 False
    if 'GPT' in translator_tool and not settings['credentials']['Open AI GPT']['api_key']:
        Messages.show_api_key_translator_error(main_page)
        return False
    
    # 如果翻译工具包含 'Gemini' 并且没有设置 Google Gemini 的 API 密钥，则显示 API 密钥错误信息并返回 False
    if 'Gemini' in translator_tool and not settings['credentials']['Google Gemini']['api_key']:
        Messages.show_api_key_translator_error(main_page)
        return False
    
    # 如果翻译工具包含 'Claude' 并且没有设置 Anthropic Claude 的 API 密钥，则显示 API 密钥错误信息并返回 False
    if 'Claude' in translator_tool and not settings['credentials']['Anthropic Claude']['api_key']:
        Messages.show_api_key_translator_error(main_page)
        return False
    
    # 检查 DeepL 和目标语言为繁体中文的不兼容性，如果满足条件则显示 DeepL 和繁体中文的错误信息并返回 False
    if translator_tool == 'DeepL' and target_lang == main_page.tr('Traditional Chinese'):
        Messages.show_deepl_ch_error(main_page)
        return False

    # 添加 Google Translate 和巴西葡萄牙语的不兼容性检查
    if translator_tool == 'Google Translate':
        # 如果源语言或目标语言为巴西葡萄牙语，则显示 Google Translate 和巴西葡萄牙语的错误信息并返回 False
        if source_lang == main_page.tr('Brazilian Portuguese') or target_lang == main_page.tr('Brazilian Portuguese'):
            Messages.show_googlet_ptbr_error(main_page)
            return False
        
    # 如果所有验证都通过，则返回 True
    return True
# 检查主页面的文本渲染设置中是否选择了字体，若未选择则显示字体选择错误消息，并返回 False
def font_selected(main_page):
    if not main_page.settings_page.get_text_rendering_settings()['font']:
        Messages.select_font_error(main_page)
        return False
    return True

# 验证 OCR 和翻译设置是否有效，若任一验证失败则返回 False
def validate_settings(main_page, source_lang, target_lang):
    if not validate_ocr(main_page, source_lang):
        return False
    if not validate_translator(main_page, source_lang, target_lang):
        return False
    if not font_selected(main_page):
        return False
    
    return True

# 根据设置页面的文本渲染设置，为文本块列表设置对齐方式
def set_alignment(blk_list, settings_page):
    text_render_settings = settings_page.get_text_rendering_settings()
    for blk in blk_list:
        alignment = text_render_settings['alignment']
        if alignment == settings_page.ui.tr("Center"):
            blk.alignment = "center"
        elif alignment == settings_page.ui.tr("Left"):
            blk.alignment = "left"
        elif alignment == settings_page.ui.tr("Right"):
            blk.alignment = "right"

# 检查目录是否为空，返回布尔值
def is_directory_empty(directory):
    # 遍历目录结构
    for root, dirs, files in os.walk(directory):
        # 如果找到任何文件，则目录非空
        if files:
            return False
    # 若未找到文件，则检查是否存在子目录
    for root, dirs, files in os.walk(directory):
        if dirs:
            # 递归检查子目录
            for dir in dirs:
                if not is_directory_empty(os.path.join(root, dir)):
                    return False
    # 若所有子目录都为空，则目录为空
    return True
```