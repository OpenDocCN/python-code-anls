# `.\comic-translate\pipeline.py`

```py
# 导入必要的模块
import os
import cv2, shutil
from datetime import datetime
from typing import List
from PySide6 import QtWidgets
from PySide6 import QtCore, QtGui

# 导入自定义模块和函数
from modules.detection import TextBlockDetector
from modules.ocr.ocr import OCRProcessor
from modules.translator import Translator
from modules.utils.textblock import TextBlock, sort_blk_list
from modules.rendering.render import get_best_render_area
from modules.utils.pipeline_utils import inpaint_map, get_config
from modules.rendering.render import draw_text, get_best_render_area
from modules.utils.pipeline_utils import generate_mask, get_language_code, set_alignment, is_directory_empty
from modules.utils.translator_utils import get_raw_translation, get_raw_text, format_translations
from modules.utils.archives import make

# 定义一个名为ComicTranslatePipeline的类
class ComicTranslatePipeline:
    # 初始化方法，接受main_page参数
    def __init__(self, main_page):
        self.main_page = main_page

    # 加载文本块的边界框坐标到图形视图中
    def load_box_coords(self, blk_list: List[TextBlock]):
        # 检查图像视图中是否有图片
        if self.main_page.image_viewer.hasPhoto():
            # 遍历传入的文本块列表
            for blk in blk_list:
                # 获取文本块的坐标信息
                x1, y1, x2, y2 = blk.xyxy
                # 创建Qt的矩形对象
                rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
                # 在图像视图中创建矩形图元对象
                rect_item = QtWidgets.QGraphicsRectItem(rect, self.main_page.image_viewer._photo)
                # 设置矩形的填充颜色为透明粉红色
                rect_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 192, 203, 100)))  # Transparent pink
                # 将创建的矩形图元对象添加到主页面的矩形列表中
                self.main_page.image_viewer._rectangles.append(rect_item)

    # 检测文本块方法
    def detect_blocks(self, load_rects=True):
        # 检查图像视图中是否有图片
        if self.main_page.image_viewer.hasPhoto():
            # 根据GPU设置选择设备
            device = 0 if self.main_page.settings_page.is_gpu_enabled() else 'cpu'
            # 创建文本块检测器对象
            block_detector = TextBlockDetector('models/detection/comic-speech-bubble-detector.pt', 
                                               'models/detection/comic-text-segmenter.pt', device)
            # 获取图像视图中的OpenCV格式图片
            image = self.main_page.image_viewer.get_cv2_image()
            # 使用文本块检测器检测文本块
            blk_list = block_detector.detect(image)
            # 返回检测到的文本块列表及是否加载矩形的标志
            return blk_list, load_rects

    # 当文本块检测完成时的回调方法
    def on_blk_detect_complete(self, result): 
        # 解包结果元组
        blk_list, load_rects = result
        # 获取当前选择的源语言
        source_lang = self.main_page.s_combo.currentText()
        # 获取源语言对应的英语名称
        source_lang_english = self.main_page.lang_mapping.get(source_lang, source_lang)
        # 根据源语言是否为日语确定文本方向
        rtl = True if source_lang_english == 'Japanese' else False
        # 根据文本方向对文本块列表进行排序
        blk_list = sort_blk_list(blk_list, rtl)
        # 将排序后的文本块列表保存到主页面的blk_list属性中
        self.main_page.blk_list = blk_list
        # 如果需要加载矩形，则调用加载文本块边界框坐标的方法
        if load_rects:
            self.load_box_coords(blk_list)
    # 手动修复图片中的缺陷
    def manual_inpaint(self):
        # 获取主页面上的图像查看器和设置页面对象
        image_viewer = self.main_page.image_viewer
        settings_page = self.main_page.settings_page
        # 获取需要修复的区域掩码
        mask = image_viewer.get_mask_for_inpainting()
        # 获取当前显示的图像数据
        image = image_viewer.get_cv2_image()

        # 根据是否启用 GPU 设置设备类型
        device = 'cuda' if settings_page.is_gpu_enabled() else 'cpu'
        # 获取选定修复工具类型
        inpainter_key = settings_page.get_tool_selection('inpainter')
        # 根据选定的修复工具类型从映射中获取相应的类
        InpainterClass = inpaint_map[inpainter_key]
        # 获取配置信息
        config = get_config(settings_page)
        # 创建修复工具对象
        inpainter = InpainterClass(device)

        # 对图像进行修复
        inpaint_input_img = inpainter(image, mask, config)
        # 转换修复后的图像数据格式为 OpenCV 可接受的格式
        inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img)

        # 返回修复后的图像数据
        return inpaint_input_img
    
    # 完成修复后的处理
    def inpaint_complete(self, result):
        # 获取修复后的图像和原始图像
        inpainted, original_image = result
        # 在主页面上设置修复后的图像显示
        self.main_page.set_cv2_image(inpainted)
        # 获取最佳渲染区域
        get_best_render_area(self.main_page.blk_list, original_image, inpainted)
        # 加载图像框坐标
        self.load_box_coords(self.main_page.blk_list)
    
    # 执行修复操作并返回修复结果
    def inpaint(self):
        # 获取当前显示的图像
        image = self.main_page.image_viewer.get_cv2_image()
        # 执行手动修复操作
        inpainted = self.manual_inpaint()
        # 返回修复后的图像和原始图像
        return inpainted, image

    # 执行修复并在完成后设置图像
    def inpaint_and_set(self):
        # 检查图像查看器是否包含图片和已绘制的元素
        if self.main_page.image_viewer.hasPhoto() and self.main_page.image_viewer.has_drawn_elements():
            # 获取当前显示的图像
            image = self.main_page.image_viewer.get_cv2_image()

            # 执行手动修复操作
            inpainted = self.manual_inpaint()
            # 在主页面上设置修复后的图像显示
            self.main_page.set_cv2_image(inpainted)

            # 获取最佳渲染区域
            get_best_render_area(self.main_page.blk_list, image, inpainted)
            # 加载图像框坐标
            self.load_box_coords(self.main_page.blk_list)

    # 对图像执行OCR识别
    def OCR_image(self):
        # 获取当前选择的源语言
        source_lang = self.main_page.s_combo.currentText()
        # 检查图像查看器是否包含图片和矩形框
        if self.main_page.image_viewer.hasPhoto() and self.main_page.image_viewer._rectangles:
            # 获取当前显示的图像
            image = self.main_page.image_viewer.get_cv2_image()
            # 更新块列表
            self.main_page.update_blk_list()
            # 创建OCR处理器对象并进行处理
            ocr = OCRProcessor(self.main_page, source_lang)
            ocr.process(image, self.main_page.blk_list)

    # 对图像执行翻译操作
    def translate_image(self):
        # 获取当前选择的源语言和目标语言
        source_lang = self.main_page.s_combo.currentText()
        target_lang = self.main_page.t_combo.currentText()
        # 检查图像查看器是否包含图片和块列表
        if self.main_page.image_viewer.hasPhoto() and self.main_page.blk_list:
            # 获取设置页面对象
            settings_page = self.main_page.settings_page
            # 获取当前显示的图像
            image = self.main_page.image_viewer.get_cv2_image()
            # 获取语言模型设置中的额外上下文信息
            extra_context = settings_page.get_llm_settings()['extra_context']

            # 创建翻译器对象并执行翻译
            translator = Translator(self.main_page, source_lang, target_lang)
            translator.translate(self.main_page.blk_list, image, extra_context)

            # 获取目标语言对应的英语名称
            target_lang_en = self.main_page.lang_mapping.get(target_lang, None)
            # 获取目标语言的语言代码
            trg_lng_cd = get_language_code(target_lang_en)
            # 获取文本渲染设置信息
            text_rendering_settings = settings_page.get_text_rendering_settings()
            upper_case = text_rendering_settings['upper_case']
            # 格式化翻译结果
            format_translations(self.main_page.blk_list, trg_lng_cd, upper_case=upper_case)
    # 在指定目录下构建保存路径，用于存放翻译后的漫画图像文件
    def skip_save(self, directory, timestamp, base_name, extension, archive_bname, image):
        # 构建完整的保存路径，包括时间戳和归档基础名称
        path = os.path.join(directory, f"comic_translate_{timestamp}", "translated_images", archive_bname)
        # 如果路径不存在，则创建
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        # 将图像从BGR格式转换为RGB格式，然后保存到指定路径
        image_save = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(path, f"{base_name}_translated{extension}"), image_save)
    
    # 记录被跳过的图像路径到文本文件中
    def log_skipped_image(self, directory, timestamp, image_path):
        # 打开文件以追加模式写入，指定UTF-8编码
        with open(os.path.join(directory, f"comic_translate_{timestamp}", "skipped_images.txt"), 'a', encoding='UTF-8') as file:
            # 将图像路径写入文件，并在末尾添加换行符
            file.write(image_path + "\n")
```