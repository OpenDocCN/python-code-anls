# `.\comic-translate\comic.py`

```py
import os  # 导入操作系统功能模块
import cv2, shutil  # 导入 OpenCV 图像处理库和文件操作模块
import tempfile  # 导入临时文件处理模块
import numpy as np  # 导入数值计算库 NumPy
from typing import Callable, Tuple, List  # 导入类型提示相关模块

from PySide6 import QtWidgets  # 导入 PySide6 的 UI 组件模块
from PySide6 import QtCore  # 导入 PySide6 的核心模块
from PySide6.QtCore import QCoreApplication  # 导入 Qt 核心模块的应用程序类
from PySide6.QtCore import QSettings  # 导入 Qt 核心模块的设置类
from PySide6.QtCore import QThreadPool  # 导入 Qt 核心模块的线程池类
from PySide6.QtCore import QTranslator, QLocale  # 导入 Qt 核心模块的国际化相关类

from app.ui.dayu_widgets import dayu_theme  # 导入 Dayu UI 框架的主题模块
from app.ui.dayu_widgets.clickable_card import ClickMeta  # 导入 Dayu UI 框架的可点击卡片模块
from app.ui.dayu_widgets.qt import MPixmap  # 导入 Dayu UI 框架的图片处理模块
from app.ui.main_window import ComicTranslateUI  # 导入主窗口 UI 模块
from app.ui.messages import Messages  # 导入消息提示模块
from app.thread_worker import GenericWorker  # 导入通用工作线程模块
from app.ui.dayu_widgets.message import MMessage  # 导入 Dayu UI 框架的消息提示模块

from modules.detection import do_rectangles_overlap  # 导入矩形检测模块
from modules.utils.textblock import TextBlock  # 导入文本块处理工具模块
from modules.rendering.render import draw_text  # 导入绘制文本模块
from modules.utils.file_handler import FileHandler  # 导入文件处理工具模块
from modules.utils.pipeline_utils import set_alignment, font_selected, validate_settings, \
                                         validate_ocr, validate_translator, get_language_code  # 导入管道工具函数
from modules.utils.archives import make  # 导入归档处理函数
from modules.utils.download import get_models, mandatory_models  # 导入模型下载相关函数
from modules.utils.translator_utils import format_translations, is_there_text  # 导入翻译工具函数
from pipeline import ComicTranslatePipeline  # 导入漫画翻译管道模块

for model in mandatory_models:
    get_models(model)  # 调用函数以获取必需模型

class ComicTranslate(ComicTranslateUI):
    image_processed = QtCore.Signal(int, object, str)  # 定义图像处理完成的信号，带有整数、对象和字符串参数
    progress_update = QtCore.Signal(int, int, int, int, bool)  # 定义进度更新的信号，带有整数、整数、整数、整数和布尔型参数
    image_skipped = QtCore.Signal(str, str, str)  # 定义图像跳过的信号，带有三个字符串参数

    def __init__(self, parent=None):
        super(ComicTranslate, self).__init__(parent)

        self.image_files = []  # 初始化图像文件列表
        self.current_image_index = -1  # 初始化当前图像索引为 -1
        self.image_states = {}  # 初始化图像状态字典

        self.blk_list = []  # 初始化块列表
        self.image_data = {}  # 存储每个图像的最新版本数据
        self.image_history = {}  # 存储每个图像的撤销/重做历史记录
        self.current_history_index = {}  # 每个图像的撤销/重做历史记录的当前位置
        self.displayed_images = set()  # 新建一个集合以跟踪显示的图像
        self.current_text_block = None  # 当前文本块为空

        self.pipeline = ComicTranslatePipeline(self)  # 初始化漫画翻译管道对象
        self.file_handler = FileHandler()  # 初始化文件处理器对象
        self.threadpool = QThreadPool()  # 初始化 Qt 线程池对象
        self.current_worker = None  # 当前工作线程为空

        self.image_skipped.connect(self.on_image_skipped)  # 连接图像跳过信号与相应槽函数
        self.image_processed.connect(self.on_image_processed)  # 连接图像处理完成信号与相应槽函数
        self.progress_update.connect(self.update_progress)  # 连接进度更新信号与相应槽函数

        self.image_cards = []  # 初始化图像卡片列表
        self.current_highlighted_card = None  # 当前高亮卡片为空

        self.connect_ui_elements()  # 连接 UI 元素
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)  # 设置焦点策略为强焦点

        self.load_main_page_settings()  # 加载主页面设置
        self.settings_page.load_settings()  # 加载设置页面的设置
    def connect_ui_elements(self):
        # 当文件选择发生变化时，连接线程加载图像函数
        self.tool_browser.sig_files_changed.connect(self.thread_load_images)
        # 当保存文件选择发生变化时，连接保存当前图像函数
        self.save_browser.sig_file_changed.connect(self.save_current_image)
        # 当保存所有文件选择发生变化时，连接保存并生成函数
        self.save_all_browser.sig_file_changed.connect(self.save_and_make)
        # 当拖动文件选择发生变化时，连接线程加载图像函数
        self.drag_browser.sig_files_changed.connect(self.thread_load_images)
        
        # 当手动模式单选按钮被点击时，连接手动模式选择函数
        self.manual_radio.clicked.connect(self.manual_mode_selected)
        # 当自动模式单选按钮被点击时，连接批处理模式选择函数
        self.automatic_radio.clicked.connect(self.batch_mode_selected)

        # 连接按钮组中的按钮
        self.hbutton_group.get_button_group().buttons()[0].clicked.connect(lambda: self.block_detect())
        self.hbutton_group.get_button_group().buttons()[1].clicked.connect(self.ocr)
        self.hbutton_group.get_button_group().buttons()[2].clicked.connect(self.translate_image)
        self.hbutton_group.get_button_group().buttons()[3].clicked.connect(self.load_segmentation_points)
        self.hbutton_group.get_button_group().buttons()[4].clicked.connect(self.inpaint_and_set)
        self.hbutton_group.get_button_group().buttons()[5].clicked.connect(self.render_text)

        # 连接返回按钮组中的按钮
        self.return_buttons_group.get_button_group().buttons()[0].clicked.connect(self.undo_image)
        self.return_buttons_group.get_button_group().buttons()[1].clicked.connect(self.redo_image)

        # 连接其他按钮和小部件
        self.translate_button.clicked.connect(self.start_batch_process)
        self.cancel_button.clicked.connect(self.cancel_current_task)
        self.set_all_button.clicked.connect(self.set_src_trg_all)
        self.clear_rectangles_button.clicked.connect(self.image_viewer.clear_rectangles)
        self.clear_brush_strokes_button.clicked.connect(self.image_viewer.clear_brush_strokes)

        # 连接文本编辑小部件
        self.s_text_edit.textChanged.connect(self.update_text_block)
        self.t_text_edit.textChanged.connect(self.update_text_block)

        self.s_combo.currentTextChanged.connect(self.save_src_trg)
        self.t_combo.currentTextChanged.connect(self.save_src_trg)

        # 连接图像查看器的信号
        self.image_viewer.rectangle_selected.connect(self.handle_rectangle_selection)
        self.image_viewer.rectangle_changed.connect(self.handle_rectangle_change)

    def save_src_trg(self):
        # 获取当前源语言和目标语言
        source_lang = self.s_combo.currentText()
        target_lang = self.t_combo.currentText()
        # 如果当前图像索引有效，则更新当前文件的语言状态
        if self.current_image_index >= 0:
            current_file = self.image_files[self.current_image_index]
            self.image_states[current_file]['source_lang'] = source_lang
            self.image_states[current_file]['target_lang'] = target_lang

    def set_src_trg_all(self):
        # 获取当前源语言和目标语言
        source_lang = self.s_combo.currentText()
        target_lang = self.t_combo.currentText()
        # 更新所有图像文件的语言状态
        for image_path in self.image_files:
            self.image_states[image_path]['source_lang'] = source_lang
            self.image_states[image_path]['target_lang'] = target_lang
    # 当批处理模式被选择时执行的方法：禁用水平按钮组，启用翻译按钮和取消按钮
    def batch_mode_selected(self):
        self.disable_hbutton_group()  # 禁用水平按钮组
        self.translate_button.setEnabled(True)  # 启用翻译按钮
        self.cancel_button.setEnabled(True)  # 启用取消按钮

    # 当手动模式被选择时执行的方法：启用水平按钮组，禁用翻译按钮和取消按钮
    def manual_mode_selected(self):
        self.enable_hbutton_group()  # 启用水平按钮组
        self.translate_button.setEnabled(False)  # 禁用翻译按钮
        self.cancel_button.setEnabled(False)  # 禁用取消按钮
    
    # 当图像处理完成时的回调方法：如果索引匹配当前图像索引，则设置显示 OpenCV 图像，否则更新图像历史记录和数据字典
    def on_image_processed(self, index: int, rendered_image: np.ndarray, image_path: str):
        if index == self.current_image_index:
            self.set_cv2_image(rendered_image)  # 设置显示 OpenCV 图像
        else:
            self.update_image_history(image_path, rendered_image)  # 更新图像历史记录
            self.image_data[image_path] = rendered_image  # 更新图像数据字典

    # 当图像被跳过时的回调方法：根据跳过原因和错误生成消息，并显示信息提示框
    def on_image_skipped(self, image_path: str, skip_reason: str, error: str):
        message = { 
            "Text Blocks": QCoreApplication.translate('Messages', 'No Text Blocks Detected.\nSkipping:') + f" {image_path}\n{error}", 
            "OCR": QCoreApplication.translate('Messages', 'Could not OCR detected text.\nSkipping:') + f" {image_path}\n{error}",
            "Translator": QCoreApplication.translate('Messages', 'Could not get translations.\nSkipping:') + f" {image_path}\n{error}"        
        }

        text = message.get(skip_reason, f"Unknown skip reason: {skip_reason}. Error: {error}")  # 获取消息文本
        # 显示信息提示框，显示消息内容，父窗口为当前对象，持续时间5秒，可关闭
        MMessage.info(
            text=text,
            parent=self,
            duration=5,
            closable=True
        )

    # 当手动操作完成时的方法：隐藏加载指示器，启用水平按钮组
    def on_manual_finished(self):
        self.loading.setVisible(False)  # 隐藏加载指示器
        self.enable_hbutton_group()  # 启用水平按钮组
    
    # 启动一个线程来执行指定的回调函数，并设置相关的回调处理函数
    def run_threaded(self, callback: Callable, result_callback: Callable=None, error_callback: Callable=None, finished_callback: Callable=None, *args, **kwargs):
        worker = GenericWorker(callback, *args, **kwargs)  # 创建一个通用的工作线程对象

        # 设置结果回调函数，使用 QTimer.singleShot 0延迟执行确保在事件循环中处理结果
        if result_callback:
            worker.signals.result.connect(lambda result: QtCore.QTimer.singleShot(0, lambda: result_callback(result)))
        # 设置错误回调函数
        if error_callback:
            worker.signals.error.connect(lambda error: QtCore.QTimer.singleShot(0, lambda: error_callback(error)))
        # 设置完成回调函数
        if finished_callback:
            worker.signals.finished.connect(finished_callback)
        
        self.current_worker = worker  # 将当前工作线程设置为新创建的工作线程
        self.threadpool.start(worker)  # 使用线程池启动工作线程

    # 取消当前正在运行的任务的方法：如果存在当前工作线程，则取消它
    def cancel_current_task(self):
        if self.current_worker:
            self.current_worker.cancel()  # 取消当前工作线程
        # 不需要启用必要的小部件/按钮，因为线程已经有完成回调处理这些

    # 默认的错误处理器方法：处理错误元组，显示错误信息对话框，并隐藏加载指示器，启用水平按钮组
    def default_error_handler(self, error_tuple: Tuple):
        exctype, value, traceback_str = error_tuple
        error_msg = f"An error occurred:\n{exctype.__name__}: {value}"
        error_msg_trcbk = f"An error occurred:\n{exctype.__name__}: {value}\n\nTraceback:\n{traceback_str}"
        print(error_msg_trcbk)  # 打印详细的错误和回溯信息到控制台
        QtWidgets.QMessageBox.critical(self, "Error", error_msg)  # 显示错误信息对话框
        self.loading.setVisible(False)  # 隐藏加载指示器
        self.enable_hbutton_group()  # 启用水平按钮组
    # 启动批处理过程的方法
    def start_batch_process(self):
        # 遍历所有图片文件路径
        for image_path in self.image_files:
            # 获取当前图片路径对应的源语言和目标语言
            source_lang = self.image_states[image_path]['source_lang']
            target_lang = self.image_states[image_path]['target_lang']

            # 验证设置是否有效，如果无效则返回
            if not validate_settings(self, source_lang, target_lang):
                return
            
        # 禁用翻译按钮
        self.translate_button.setEnabled(False)
        # 显示进度条
        self.progress_bar.setVisible(True) 
        # 在线程中运行批处理方法，传入空参数，设置默认错误处理器和批处理完成后的回调函数
        self.run_threaded(self.pipeline.batch_process, None, self.default_error_handler, self.on_batch_process_finished)

    # 批处理完成后的回调方法
    def on_batch_process_finished(self):
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        # 启用翻译按钮
        self.translate_button.setEnabled(True)
        # 显示翻译完成消息框
        Messages.show_translation_complete(self)

    # 禁用水平按钮组的所有按钮
    def disable_hbutton_group(self):
        for button in self.hbutton_group.get_button_group().buttons():
            button.setEnabled(False)

    # 启用水平按钮组的所有按钮
    def enable_hbutton_group(self):
        for button in self.hbutton_group.get_button_group().buttons():
            button.setEnabled(True)

    # 执行区块检测的方法，可选择是否加载区块
    def block_detect(self, load_rects: bool = True):
        # 显示加载中的标志
        self.loading.setVisible(True)
        # 禁用水平按钮组的所有按钮
        self.disable_hbutton_group()
        # 在线程中运行区块检测方法，传入检测完成后的回调函数、默认错误处理器和手动完成时的回调函数，可以选择是否加载区块
        self.run_threaded(self.pipeline.detect_blocks, self.pipeline.on_blk_detect_complete, 
                          self.default_error_handler, self.on_manual_finished, load_rects)

    # 完成OCR和翻译的方法
    def finish_ocr_translate(self):
        # 找到对应的矩形区域，并选择该区域
        rect = self.find_corresponding_rect(self.blk_list[0], 0.5)
        self.image_viewer.select_rectangle(rect)
        # 设置工具为“box”
        self.set_tool('box')
        # 手动完成操作
        self.on_manual_finished()

    # 执行OCR识别的方法
    def ocr(self):
        # 获取当前选择的源语言
        source_lang = self.s_combo.currentText()
        # 如果OCR验证失败，则返回
        if not validate_ocr(self, source_lang):
            return
        # 显示加载中的标志
        self.loading.setVisible(True)
        # 禁用水平按钮组的所有按钮
        self.disable_hbutton_group()
        # 在线程中运行OCR识别方法，设置默认错误处理器和OCR完成后的翻译方法
        self.run_threaded(self.pipeline.OCR_image, None, self.default_error_handler, self.finish_ocr_translate)

    # 执行图片翻译的方法
    def translate_image(self):
        # 获取当前选择的源语言和目标语言
        source_lang = self.s_combo.currentText()
        target_lang = self.t_combo.currentText()
        # 如果区块列表中没有文本或翻译验证失败，则返回
        if not is_there_text(self.blk_list) or not validate_translator(self, source_lang, target_lang):
            return
        # 显示加载中的标志
        self.loading.setVisible(True)
        # 禁用水平按钮组的所有按钮
        self.disable_hbutton_group()
        # 在线程中运行图片翻译方法，设置默认错误处理器和OCR完成后的翻译方法
        self.run_threaded(self.pipeline.translate_image, None, self.default_error_handler, self.finish_ocr_translate)

    # 执行修复和设置的方法
    def inpaint_and_set(self):
        # 如果图像查看器中有照片并且已经绘制了元素
        if self.image_viewer.hasPhoto() and self.image_viewer.has_drawn_elements():
            # 显示加载中的标志
            self.loading.setVisible(True)
            # 禁用水平按钮组的所有按钮
            self.disable_hbutton_group()
            # 在线程中运行修复方法，设置修复完成后的回调函数、默认错误处理器和手动完成时的回调函数
            self.run_threaded(self.pipeline.inpaint, self.pipeline.inpaint_complete, 
                              self.default_error_handler, self.on_manual_finished)
    # 使用多线程加载给定的文件路径中的图像数据
    def load_images_threaded(self, file_paths: List[str]):
        # 将文件路径列表设置为文件处理器的属性
        self.file_handler.file_paths = file_paths
        # 准备文件路径列表，确保文件可用性
        file_paths = self.file_handler.prepare_files()

        # 初始化一个空列表，用于存储加载后的图像数据和对应的文件路径
        loaded_images = []
        # 遍历每个文件路径
        for file_path in file_paths:
            # 使用 OpenCV 读取图像文件
            cv2_image = cv2.imread(file_path)
            # 如果成功读取图像（不为None）
            if cv2_image is not None:
                # 将图像从 BGR 格式转换为 RGB 格式
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                # 将文件路径和处理后的图像添加到加载后的图像列表中
                loaded_images.append((file_path, cv2_image))

        # 返回加载后的图像列表
        return loaded_images

    # 使用线程加载图像数据的包装方法
    def thread_load_images(self, file_paths: List[str]):
        # 在线程中运行加载图像的方法，并指定回调函数和错误处理函数
        self.run_threaded(self.load_images_threaded, self.on_images_loaded, self.default_error_handler, None, file_paths)

    # 当图像加载完成时调用的方法
    def on_images_loaded(self, loaded_images: List[Tuple[str, np.ndarray]]):
        # 清空现有的图像数据和状态
        self.image_files = []
        self.image_states.clear()
        self.image_data.clear()
        self.image_history.clear()
        self.current_history_index.clear()
        self.blk_list = []
        self.displayed_images.clear()
        # 清空图像查看器中的矩形和画笔笔划
        self.image_viewer.clear_rectangles()
        self.image_viewer.clear_brush_strokes()
        # 清空文本编辑器中的文本
        self.s_text_edit.clear()
        self.t_text_edit.clear()

        # 重置当前图像索引为-1
        self.current_image_index = -1

        # 遍历加载后的图像列表
        for file_path, cv2_image in loaded_images:
            # 将文件路径添加到图像文件列表中
            self.image_files.append(file_path)
            # 将图像数据添加到图像数据字典中
            self.image_data[file_path] = cv2_image
            # 初始化图像历史记录，当前历史索引，并保存图像状态
            self.image_history[file_path] = [cv2_image.copy()]
            self.current_history_index[file_path] = 0
            self.save_image_state(file_path)

        # 更新图像卡片显示
        self.update_image_cards()

        # 如果成功加载了图像，则显示第一张图像
        if self.image_files:
            self.display_image(0)
        else:
            # 如果未成功加载任何图像，则清空图像查看器
            self.image_viewer.clear_scene()

        # 重置图像查看器的变换
        self.image_viewer.resetTransform()
        self.image_viewer.fitInView()

    # 更新图像卡片显示的方法
    def update_image_cards(self):
        # 清空现有的图像卡片
        for i in reversed(range(self.image_card_layout.count())):
            widget = self.image_card_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        
        # 重置图像卡片列表
        self.image_cards = []

        # 添加新的图像卡片
        for index, file_path in enumerate(self.image_files):
            # 获取文件路径的基本名称作为卡片的标题
            file_name = os.path.basename(file_path)
            # 创建一个新的图像卡片对象
            card = ClickMeta(extra=False, avatar_size=(35, 50))
            # 设置图像卡片的数据和标题
            card.setup_data({
                "title": file_name,
                "avatar": MPixmap(file_path)
            })
            # 将卡片与点击事件绑定，并传递索引作为参数
            card.connect_clicked(lambda idx=index: self.on_card_clicked(idx))
            # 将卡片添加到图像卡片布局中
            self.image_card_layout.insertWidget(self.image_card_layout.count() - 1, card)
            # 将卡片对象添加到图像卡片列表中
            self.image_cards.append(card)
    # 高亮显示指定索引的卡片
    def highlight_card(self, index: int):
        # 检查索引是否在有效范围内
        if 0 <= index < len(self.image_cards):
            # 如果存在已经高亮的卡片，取消其高亮状态
            if self.current_highlighted_card:
                self.current_highlighted_card.set_highlight(False)
            
            # 设置新卡片为高亮状态
            self.image_cards[index].set_highlight(True)
            self.current_highlighted_card = self.image_cards[index]

    # 处理卡片点击事件
    def on_card_clicked(self, index: int):
        self.highlight_card(index)  # 调用高亮显示卡片的方法
        self.display_image(index)   # 显示指定索引的图片

    # 保存当前视图状态到指定文件的图片状态字典中
    def save_image_state(self, file: str):
        self.image_states[file] = {
            'viewer_state': self.image_viewer.save_state(),
            'source_text': self.s_text_edit.toPlainText(),
            'source_lang': self.s_combo.currentText(),
            'target_text': self.t_text_edit.toPlainText(),
            'target_lang': self.t_combo.currentText(),
            'brush_strokes': self.image_viewer.save_brush_strokes(),
            'undo_brush_stack': self.image_viewer._undo_brush_stack,
            'redo_brush_stack': self.image_viewer._redo_brush_stack,
            'blk_list': self.blk_list
        }

    # 保存当前图片状态到图片状态字典中
    def save_current_image_state(self):
        if self.current_image_index >= 0:
            current_file = self.image_files[self.current_image_index]
            self.save_image_state(current_file)

    # 加载指定文件路径的图片状态
    def load_image_state(self, file_path: str):
        cv2_image = self.image_data[file_path]  # 获取指定文件路径的图像数据

        # 设置OpenCV图像数据到视图器中
        self.set_cv2_image(cv2_image)

        # 如果文件路径在图片状态字典中，则恢复相关状态信息
        if file_path in self.image_states:
            state = self.image_states[file_path]
            self.image_viewer.load_state(state['viewer_state'])
            self.s_text_edit.setPlainText(state['source_text'])
            self.s_combo.setCurrentText(state['source_lang'])
            self.t_text_edit.setPlainText(state['target_text'])
            self.t_combo.setCurrentText(state['target_lang'])
            self.image_viewer.load_brush_strokes(state['brush_strokes'])
            self.blk_list = state['blk_list']
        else:
            # 如果文件路径不在字典中，清空相关状态信息
            self.s_text_edit.clear()
            self.t_text_edit.clear()

    # 显示指定索引的图片
    def display_image(self, index: int):
        # 检查索引是否在有效范围内
        if 0 <= index < len(self.image_files):
            self.save_current_image_state()  # 保存当前图片的状态
            self.current_image_index = index  # 更新当前图片索引
            file_path = self.image_files[index]  # 获取指定索引对应的文件路径
            
            # 检查图片是否已经显示过
            first_time_display = file_path not in self.displayed_images
            
            # 加载图片状态到视图中
            self.load_image_state(file_path)
            
            # 切换至图片视图
            self.central_stack.setCurrentWidget(self.image_viewer)
            self.central_stack.layout().activate()
            
            # 如果是第一次显示该图片，则调整视图以适应图片
            if first_time_display:
                self.image_viewer.fitInView()
                self.displayed_images.add(file_path)  # 标记该图片已显示过
    # 对块检测结果进行分段处理
    def blk_detect_segment(self, result): 
        # 将结果解包为块列表和加载的矩形区域
        blk_list, load_rects = result
        # 将块列表保存到对象的属性中
        self.blk_list = blk_list
        # 遍历每一个块
        for blk in self.blk_list:
            # 获取当前块的分段点集合
            segm_pts = blk.segm_pts
            # 如果分段点集合不为空，则在图像视图中绘制分段线
            if segm_pts.size > 0:
                self.image_viewer.draw_segmentation_lines(segm_pts)

    # 加载分段点
    def load_segmentation_points(self):
        # 检查图像视图是否有照片
        if self.image_viewer.hasPhoto():
            # 设置工具为刷子工具
            self.set_tool('brush')
            # 禁用水平按钮组
            self.disable_hbutton_group()
            # 清除图像视图中的矩形区域
            self.image_viewer.clear_rectangles()
            # 如果块列表不为空
            if self.blk_list:
                # 遍历每一个块
                for blk in self.blk_list:
                    # 获取当前块的分段点集合
                    segm_pts = blk.segm_pts
                    # 如果分段点集合不为空，则在图像视图中绘制分段线
                    if segm_pts.size > 0:
                        self.image_viewer.draw_segmentation_lines(segm_pts)
                # 启用水平按钮组
                self.enable_hbutton_group()
            else:
                # 显示加载中的提示
                self.loading.setVisible(True)
                # 禁用水平按钮组
                self.disable_hbutton_group()
                # 启动线程进行块检测，传入处理函数 blk_detect_segment，并设置默认错误处理函数和手动完成时的回调函数
                self.run_threaded(self.pipeline.detect_blocks, self.blk_detect_segment, 
                                  self.default_error_handler, self.on_manual_finished)
                
    # 更新图像历史记录
    def update_image_history(self, file_path: str, cv2_img: np.ndarray):
        # 检查新图像与当前图像是否不同
        if not np.array_equal(self.image_data[file_path], cv2_img):
            # 更新图像数据
            self.image_data[file_path] = cv2_img
            # 获取当前文件路径的历史记录
            history = self.image_history[file_path]
            # 获取当前历史记录的索引
            current_index = self.current_history_index[file_path]
            # 如果当前索引不是最后一个，移除其后的历史记录
            del history[current_index + 1:]
            # 将新图像添加到历史记录末尾
            history.append(cv2_img.copy())
            # 更新当前历史记录索引为最后一个
            self.current_history_index[file_path] = len(history) - 1

    # 设置 OpenCV 图像
    def set_cv2_image(self, cv2_img: np.ndarray):
        # 如果当前图像索引大于等于0
        if self.current_image_index >= 0:
            # 获取当前图像文件路径
            file_path = self.image_files[self.current_image_index]
            # 更新图像历史记录
            self.update_image_history(file_path, cv2_img)
            # 在图像视图中显示 OpenCV 图像
            self.image_viewer.display_cv2_image(cv2_img)

    # 撤销图像操作
    def undo_image(self):
        # 如果当前图像索引大于等于0
        if self.current_image_index >= 0:
            # 获取当前图像文件路径
            file_path = self.image_files[self.current_image_index]
            # 获取当前历史记录的索引
            current_index = self.current_history_index[file_path]
            # 反向遍历历史记录，直到找到不同的图像
            while current_index > 0:
                current_index -= 1
                cv2_img = self.image_history[file_path][current_index]
                # 如果找到不同的图像，则更新当前历史记录索引，并更新图像数据和显示图像
                if not np.array_equal(self.image_data[file_path], cv2_img):
                    self.current_history_index[file_path] = current_index
                    self.image_data[file_path] = cv2_img
                    self.image_viewer.display_cv2_image(cv2_img)
                    break
    # 重新显示当前图像的历史记录中的下一个图像
    def redo_image(self):
        if self.current_image_index >= 0:
            file_path = self.image_files[self.current_image_index]
            current_index = self.current_history_index[file_path]
            # 循环直到找到与当前图像数据不同的历史记录图像
            while current_index < len(self.image_history[file_path]) - 1:
                current_index += 1
                cv2_img = self.image_history[file_path][current_index]
                # 如果当前图像数据与历史记录不同，则更新图像数据并显示
                if not np.array_equal(self.image_data[file_path], cv2_img):
                    self.current_history_index[file_path] = current_index
                    self.image_data[file_path] = cv2_img
                    self.image_viewer.display_cv2_image(cv2_img)
                    break

    # 更新文本块列表
    def update_blk_list(self):
        # 获取图像查看器中当前的矩形框坐标
        current_rectangles = self.image_viewer.get_rectangle_coordinates()
        
        # 创建一个新列表来存储更新后的文本块
        updated_blk_list = []
        
        # 遍历现有文本块列表，与当前矩形框进行比对
        for blk in self.blk_list:
            blk_rect = tuple(blk.xyxy)
            
            # 检查该文本块是否仍然存在或已经略有修改
            for i, curr_rect in enumerate(current_rectangles):
                if do_rectangles_overlap(blk_rect, curr_rect, iou_threshold=0.5):
                    # 更新文本块的坐标
                    blk.xyxy[:] = list(curr_rect)
                    updated_blk_list.append(blk)
                    current_rectangles.pop(i)
                    break
        
        # 对于剩余的矩形框，创建新的文本块并添加到更新后的列表中
        for new_rect in current_rectangles:
            new_blk = TextBlock(np.array(new_rect))
            updated_blk_list.append(new_blk)
        
        # 使用更新后的配置更新 self.blk_list
        self.blk_list = updated_blk_list

    # 查找与给定矩形框重叠的对应文本块
    def find_corresponding_text_block(self, rect: Tuple[float], iou_threshold: int):
        for blk in self.blk_list:
            if do_rectangles_overlap(rect, blk.xyxy, iou_threshold):
                return blk
        return None

    # 查找与给定文本块重叠的对应矩形框
    def find_corresponding_rect(self, tblock: TextBlock, iou_threshold: int):
        for rect in self.image_viewer._rectangles:
            x1, y1, w, h = rect.rect().getRect()
            rect_coord = (x1, y1, x1 + w, y1 + h)
            if do_rectangles_overlap(rect_coord, tblock.xyxy, iou_threshold):
                return rect
        return None
    # 处理矩形选择事件，根据给定的矩形参数确定选择区域
    def handle_rectangle_selection(self, rect: QtCore.QRectF):
        # 从 QRectF 对象获取左上角坐标 (x1, y1) 和宽度 w、高度 h
        x1, y1, w, h = rect.getRect()
        # 将矩形参数重新定义为 (x1, y1, x2, y2) 的形式
        rect = (x1, y1, x1 + w, y1 + h)
        
        # 查找与当前选中矩形对应的文本块，相似度阈值为 0.5
        self.current_text_block = self.find_corresponding_text_block(rect, 0.5)
        
        if self.current_text_block:
            # 断开文本编辑框的文本改变信号，避免循环触发更新
            self.s_text_edit.textChanged.disconnect(self.update_text_block)
            self.t_text_edit.textChanged.disconnect(self.update_text_block)
            # 设置源文本编辑框和目标文本编辑框的文本内容为当前文本块的文本和翻译
            self.s_text_edit.setPlainText(self.current_text_block.text)
            self.t_text_edit.setPlainText(self.current_text_block.translation)
            # 重新连接文本编辑框的文本改变信号到更新文本块函数
            self.s_text_edit.textChanged.connect(self.update_text_block)
            self.t_text_edit.textChanged.connect(self.update_text_block)
        else:
            # 若未找到对应文本块，清空文本编辑框，并将当前文本块设为 None
            self.s_text_edit.clear()
            self.t_text_edit.clear()
            self.current_text_block = None

    # 更新当前文本块的文本内容和翻译内容
    def update_text_block(self):
        if self.current_text_block:
            self.current_text_block.text = self.s_text_edit.toPlainText()
            self.current_text_block.translation = self.t_text_edit.toPlainText()

    # 更新进度条显示状态，根据当前处理索引、总图像数、步骤索引、总步骤数和是否更改名称来确定进度
    def update_progress(self, index: int, total_images: int, step: int, total_steps: int, change_name: bool):
        # 分配图像处理和归档的权重比例（根据需求进行调整）
        image_processing_weight = 0.9
        archiving_weight = 0.1

        # 获取归档信息列表和总归档数目
        archive_info_list = self.file_handler.archive_info
        total_archives = len(archive_info_list)

        if change_name:
            if index < total_images:
                # 更新进度条的显示格式为图像处理进度
                im_path = self.image_files[index]
                im_name = os.path.basename(im_path)
                self.progress_bar.setFormat(QCoreApplication.translate('Messages', 'Processing:') + f" {im_name} . . . %p%")
            else:
                # 更新进度条的显示格式为归档进度
                archive_index = index - total_images
                self.progress_bar.setFormat(QCoreApplication.translate('Messages', 'Archiving:') + f" {archive_index + 1}/{total_archives} . . . %p%")

        if index < total_images:
            # 计算图像处理进度
            task_progress = (index / total_images) * image_processing_weight
            step_progress = (step / total_steps) * (1 / total_images) * image_processing_weight
        else:
            # 计算归档进度
            archive_index = index - total_images
            task_progress = image_processing_weight + (archive_index / total_archives) * archiving_weight
            step_progress = (step / total_steps) * (1 / total_archives) * archiving_weight

        # 计算总体进度百分比并更新进度条的值
        progress = (task_progress + step_progress) * 100 
        self.progress_bar.setValue(int(progress))

    # 渲染完成后的回调函数，设置图像并显示相关界面元素
    def on_render_complete(self, rendered_image: np.ndarray):
        self.set_cv2_image(rendered_image)
        self.loading.setVisible(False)
        self.enable_hbutton_group()
    # 渲染文本内容到图像上
    def render_text(self):
        # 检查是否有图片显示器中的照片并且存在文本块列表
        if self.image_viewer.hasPhoto() and self.blk_list:
            # 如果没有选择字体则返回
            if not font_selected(self):
                return
            # 显示加载状态
            self.loading.setVisible(True)
            # 禁用水平按钮组
            self.disable_hbutton_group()
            # 获取当前图像的 OpenCV 格式
            inpaint_image = self.image_viewer.get_cv2_image()
            # 获取文本渲染设置
            text_rendering_settings = self.settings_page.get_text_rendering_settings()
            # 获取字体和颜色
            font = text_rendering_settings['font']
            font_color = text_rendering_settings['color']
            # 获取是否大写设置
            upper = text_rendering_settings['upper_case']
            # 构建字体文件路径
            font_path = font_path = f'fonts/{font}'
            # 设置文本块的对齐方式
            set_alignment(self.blk_list, self.settings_page)

            # 获取目标语言并进行映射
            target_lang = self.t_combo.currentText()
            target_lang_en = self.lang_mapping.get(target_lang, None)
            trg_lng_cd = get_language_code(target_lang_en)
            # 格式化翻译内容
            format_translations(self.blk_list, trg_lng_cd, upper_case=upper)

            # 在线程中运行绘制文本函数
            self.run_threaded(draw_text, self.on_render_complete, self.default_error_handler, 
                              None, inpaint_image, self.blk_list, font_path, 40, colour=font_color)
            
    # 处理矩形变化事件
    def handle_rectangle_change(self, new_rect: QtCore.QRectF):
        # 查找在blk_list中与给定矩形重叠的文本块
        for blk in self.blk_list:
            if do_rectangles_overlap(blk.xyxy, (new_rect.left(), new_rect.top(), new_rect.right(), new_rect.bottom()), 0.2):
                # 更新文本块的坐标
                blk.xyxy[:] = [new_rect.left(), new_rect.top(), new_rect.right(), new_rect.bottom()]
                break

    # 保存当前图像到指定文件路径
    def save_current_image(self, file_path: str):
        curr_image = self.image_viewer.get_cv2_image()
        cv2.imwrite(file_path, curr_image)

    # 保存并生成处理结果
    def save_and_make(self, output_path: str):
        # 在线程中运行保存并生成工作函数
        self.run_threaded(self.save_and_make_worker, None, self.default_error_handler, None, output_path)

    # 保存并生成工作函数
    def save_and_make_worker(self, output_path: str):
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        try:
            # 保存图片
            for file_path in self.image_files:
                bname = os.path.basename(file_path) 
                cv2_img = self.image_data[file_path]
                cv2_img_save = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                sv_pth = os.path.join(temp_dir, bname)
                cv2.imwrite(sv_pth, cv2_img_save)
            
            # 调用生成函数
            make(temp_dir, output_path)
        finally:
            # 清理临时目录
            import shutil
            shutil.rmtree(temp_dir)

    # 键盘按下事件处理
    def keyPressEvent(self, event):
        # 按下左箭头键
        if event.key() == QtCore.Qt.Key_Left:
            self.navigate_images(-1)
        # 按下右箭头键
        elif event.key() == QtCore.Qt.Key_Right:
            self.navigate_images(1)
        else:
            super().keyPressEvent(event)
    # 根据传入的方向参数来导航图片列表
    def navigate_images(self, direction: int):
        # 检查对象是否具有'image_files'属性并且不为空
        if hasattr(self, 'image_files') and self.image_files:
            # 计算新的图片索引
            new_index = self.current_image_index + direction
            # 如果新索引在有效范围内，则显示新图片并高亮对应卡片
            if 0 <= new_index < len(self.image_files):
                self.display_image(new_index)
                self.highlight_card(new_index)

    # 保存主页面的设置
    def save_main_page_settings(self):
        # 创建QSettings对象，指定组织名和应用程序名
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("main_page")  # 开始'main_page'组的设置
        
        # 保存源语言和目标语言设置为英文
        settings.setValue("source_language", self.lang_mapping[self.s_combo.currentText()])
        settings.setValue("target_language", self.lang_mapping[self.t_combo.currentText()])
        
        # 根据选择的单选按钮保存模式设置为手动或自动
        settings.setValue("mode", "manual" if self.manual_radio.isChecked() else "automatic")
        
        # 保存画笔和橡皮擦的大小设置
        settings.setValue("brush_size", self.brush_size_slider.value())
        settings.setValue("eraser_size", self.eraser_size_slider.value())
        
        settings.endGroup()  # 结束'main_page'组的设置

        # 保存窗口状态
        settings.beginGroup("MainWindow")  # 开始'MainWindow'组的设置
        settings.setValue("geometry", self.saveGeometry())  # 保存窗口的几何状态
        settings.setValue("state", self.saveState())  # 保存窗口的状态
        settings.endGroup()  # 结束'MainWindow'组的设置

    # 加载主页面的设置
    def load_main_page_settings(self):
        # 创建QSettings对象，指定组织名和应用程序名
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("main_page")  # 开始'main_page'组的设置

        # 加载源语言和目标语言设置，并转换为当前语言
        source_lang = settings.value("source_language", "Korean")
        target_lang = settings.value("target_language", "English")
        
        # 使用反向映射获取翻译后的语言名称，并设置当前文本框的文本
        self.s_combo.setCurrentText(self.reverse_lang_mapping.get(source_lang, self.tr("Korean")))
        self.t_combo.setCurrentText(self.reverse_lang_mapping.get(target_lang, self.tr("English")))

        # 加载模式设置，并根据加载的模式选择相应的操作
        mode = settings.value("mode", "manual")
        if mode == "manual":
            self.manual_radio.setChecked(True)
            self.manual_mode_selected()
        else:
            self.automatic_radio.setChecked(True)
            self.batch_mode_selected()
        
        # 加载画笔和橡皮擦大小设置
        brush_size = settings.value("brush_size", 10)  # 默认值为10
        eraser_size = settings.value("eraser_size", 20)  # 默认值为20
        self.brush_size_slider.setValue(int(brush_size))
        self.eraser_size_slider.setValue(int(eraser_size))
        
        settings.endGroup()  # 结束'main_page'组的设置

        # 加载窗口状态
        settings.beginGroup("MainWindow")  # 开始'MainWindow'组的设置
        geometry = settings.value("geometry")  # 加载窗口的几何状态
        state = settings.value("state")  # 加载窗口的状态
        if geometry is not None:
            self.restoreGeometry(geometry)  # 如果存在几何状态，则恢复窗口几何状态
        if state is not None:
            self.restoreState(state)  # 如果存在窗口状态，则恢复窗口状态
        settings.endGroup()  # 结束'MainWindow'组的设置
    # 在应用关闭时保存所有设置
    self.settings_page.save_settings()
    # 保存主页面的设置
    self.save_main_page_settings()
    
    # 删除临时存档文件夹
    for archive in self.file_handler.archive_info:
        # 逐个删除存档信息中的临时文件夹
        shutil.rmtree(archive['temp_dir'])

    # 调用父类的关闭事件处理方法
    super().closeEvent(event)
def get_system_language():
    locale = QLocale.system().name()  # 获取系统的区域设置名称，例如 "en_US" 或 "zh_CN"
    
    # 处理中文特殊情况
    if locale.startswith('zh_'):
        if locale in ['zh_CN', 'zh_SG']:
            return '简体中文'  # 如果是中国大陆或新加坡中文区域设置，则返回简体中文
        elif locale in ['zh_TW', 'zh_HK']:
            return '繁體中文'  # 如果是台湾或香港中文区域设置，则返回繁体中文
    
    # 对于其他语言，仍然使用区域设置的第一部分作为语言代码
    lang_code = locale.split('_')[0]
    
    # 将系统语言代码映射到应用程序的语言名称
    lang_map = {
        'en': 'English',
        'ko': '한국어',
        'fr': 'Français',
        'ja': '日本語',
        'ru': 'русский',
        'de': 'Deutsch',
        'nl': 'Nederlands',
        'es': 'Español',
        'it': 'Italiano',
        'tr': 'Türkçe'
    }
    
    return lang_map.get(lang_code, 'English')  # 如果未找到对应的语言名称，则默认返回英语

def load_translation(app, language: str):
    translator = QTranslator(app)
    lang_code = {
        'English': 'en',
        '한국어': 'ko',
        'Français': 'fr',
        '日本語': 'ja',
        '简体中文': 'zh_CN',
        '繁體中文': 'zh_TW',
        'русский': 'ru',
        'Deutsch': 'de',
        'Nederlands': 'nl',
        'Español': 'es',
        'Italiano': 'it',
        'Türkçe': 'tr'
    }.get(language, 'en')  # 根据传入的语言名称获取对应的语言代码，默认为英语代码 'en'

    # 加载翻译文件
    if translator.load(f":/translations/ct_{lang_code}.qm"):  # 尝试加载应用程序的翻译文件
        app.installTranslator(translator)  # 安装翻译器到应用程序
    else:
        print(f"Failed to load translation for {language}")  # 如果加载失败，打印加载失败信息

if __name__ == "__main__":

    import sys
    from PySide6.QtGui import QIcon
    from app.ui.dayu_widgets.qt import application
    from app.translations import ct_translations
    from app import icon_resource

    if sys.platform == "win32":
        # Windows 平台下的必要解决方案：设置任务栏图标
        import ctypes
        myappid = u'ComicLabs.ComicTranslate'  # 任意字符串
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    with application() as app:
        # 设置应用程序图标
        icon = QIcon(":/icons/window_icon.png")  
        app.setWindowIcon(icon)

        settings = QSettings("ComicLabs", "ComicTranslate")
        selected_language = settings.value('language', get_system_language())
        if selected_language != 'English':
            load_translation(app, selected_language)  # 加载用户选择的语言翻译

        test = ComicTranslate()  # 创建应用程序主窗口实例
        test.show()  # 显示主窗口
```