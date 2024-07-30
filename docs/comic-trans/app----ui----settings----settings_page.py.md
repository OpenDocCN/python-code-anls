# `.\comic-translate\app\ui\settings\settings_page.py`

```py
# 导入必要的模块
import os, shutil
from typing import List

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Signal, QSettings

# 导入自定义的设置页面UI类
from .settings_ui import SettingsPageUI

# 设置页面类，继承自QWidget
class SettingsPage(QtWidgets.QWidget):
    # 定义主题改变的信号
    theme_changed = Signal(str)

    # 初始化方法
    def __init__(self, parent=None):
        super(SettingsPage, self).__init__(parent)
        
        # 创建设置页面UI实例
        self.ui = SettingsPageUI(self)
        
        # 设置页面信号和槽的连接
        self._setup_connections()
        
        # 设置加载设置的标志为False
        self._loading_settings = False
        
        # 创建垂直布局并将UI添加到布局中
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.ui)
        self.setLayout(layout)

    # 设置信号和槽的连接方法
    def _setup_connections(self):
        # 当主题组合框的选择文本发生变化时连接到槽函数on_theme_changed
        self.ui.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        # 当语言组合框的选择文本发生变化时连接到槽函数on_language_changed
        self.ui.lang_combo.currentTextChanged.connect(self.on_language_changed)
        # 当字体浏览器的文件改变信号发生时连接到槽函数import_font
        self.ui.font_browser.sig_files_changed.connect(self.import_font)
        # 当颜色按钮被点击时连接到槽函数select_color
        self.ui.color_button.clicked.connect(self.select_color)

    # 主题改变时发射主题改变信号的槽函数
    def on_theme_changed(self, theme: str):
        self.theme_changed.emit(theme)

    # 获取当前语言选择的方法
    def get_language(self):
        return self.ui.lang_combo.currentText()
    
    # 获取当前主题选择的方法
    def get_theme(self):
        return self.ui.theme_combo.currentText()

    # 获取特定工具类型的选择
    def get_tool_selection(self, tool_type):
        tool_combos = {
            'translator': self.ui.translator_combo,
            'ocr': self.ui.ocr_combo,
            'inpainter': self.ui.inpainter_combo
        }
        return tool_combos[tool_type].currentText()

    # 检查GPU是否启用的方法
    def is_gpu_enabled(self):
        return self.ui.use_gpu_checkbox.isChecked()

    # 获取文本渲染设置的方法
    def get_text_rendering_settings(self):
        return {
            'alignment': self.ui.text_rendering_widgets['alignment'].currentText(),
            'font': self.ui.text_rendering_widgets['font'].currentText(),
            'color': self.ui.text_rendering_widgets['color_button'].property('selected_color'),
            'upper_case': self.ui.text_rendering_widgets['upper_case'].isChecked()
        }

    # 获取LLM模型设置的方法
    def get_llm_settings(self):
        return {
            'extra_context': self.ui.llm_widgets['extra_context'].toPlainText(),
            'image_input_enabled': self.ui.llm_widgets['image_input'].isChecked()
        }

    # 获取导出设置的方法
    def get_export_settings(self):
        settings = {
            'export_raw_text': self.ui.export_widgets['raw_text'].isChecked(),
            'export_translated_text': self.ui.export_widgets['translated_text'].isChecked(),
            'export_inpainted_image': self.ui.export_widgets['inpainted_image'].isChecked(),
            'save_as': {}
        }
        for file_type in ['.pdf', '.epub', '.cbr', '.cbz', '.cb7', '.cbt']:
            settings['save_as'][file_type] = self.ui.export_widgets[f'{file_type}_save_as'].currentText()
        return settings
    # 获取用户凭证信息，根据用户界面的复选框状态确定是否保存密钥
    def get_credentials(self, service: str = ""):
        # 检查保存密钥复选框的选中状态
        save_keys = self.ui.save_keys_checkbox.isChecked()
        
        # 如果指定了特定的服务名
        if service:
            # 如果服务名为 "Microsoft Azure"
            if service == "Microsoft Azure":
                # 返回 Microsoft Azure 相关的凭证信息字典
                return {
                    'api_key_ocr': self.ui.credential_widgets["Microsoft Azure_api_key_ocr"].text(),
                    'api_key_translator': self.ui.credential_widgets["Microsoft Azure_api_key_translator"].text(),
                    'region_translator': self.ui.credential_widgets["Microsoft Azure_region"].text(),
                    'save_key': save_keys,
                    'endpoint': self.ui.credential_widgets["Microsoft Azure_endpoint"].text()
                }
            else:
                # 返回其他服务的凭证信息字典
                return {
                    'api_key': self.ui.credential_widgets[f"{service}_api_key"].text(),
                    'save_key': save_keys
                }
        else:
            # 返回所有已配置服务的凭证信息字典
            return {s: self.get_credentials(s) for s in self.ui.credential_services}
        
    # 获取高清策略设置信息，包括选择的策略和相关的参数
    def get_hd_strategy_settings(self):
        # 获取选择的图像修复策略
        strategy = self.ui.inpaint_strategy_combo.currentText()
        # 构造策略设置字典
        settings = {
            'strategy': strategy
        }

        # 根据选择的策略添加相应的参数到设置字典中
        if strategy == self.ui.tr("Resize"):
            settings['resize_limit'] = self.ui.resize_spinbox.value()
        elif strategy == self.ui.tr("Crop"):
            settings['crop_margin'] = self.ui.crop_margin_spinbox.value()
            settings['crop_trigger_size'] = self.ui.crop_trigger_spinbox.value()

        # 返回完整的策略设置字典
        return settings

    # 获取所有应用设置的信息，包括语言、主题、工具配置、文本渲染等
    def get_all_settings(self):
        return {
            'language': self.get_language(),
            'theme': self.get_theme(),
            'tools': {
                'translator': self.get_tool_selection('translator'),
                'ocr': self.get_tool_selection('ocr'),
                'inpainter': self.get_tool_selection('inpainter'),
                'use_gpu': self.is_gpu_enabled(),
                'hd_strategy': self.get_hd_strategy_settings()
            },
            'text_rendering': self.get_text_rendering_settings(),
            'llm': self.get_llm_settings(),
            'export': self.get_export_settings(),
            'credentials': self.get_credentials(),
            'save_keys': self.ui.save_keys_checkbox.isChecked()
        }

    # 导入字体文件到应用中的字体文件夹，并更新界面中的字体选择器
    def import_font(self, file_paths: List[str]):
        # 获取导入的字体文件路径列表
        font_files = file_paths
        # 设置字体文件夹的路径
        font_folder_path = os.path.join(os.getcwd(), "fonts")
        # 如果字体文件夹不存在，则创建它
        if not os.path.exists(font_folder_path):
            os.makedirs(font_folder_path)

        # 如果有导入的字体文件
        if font_files:
            # 将每个字体文件复制到字体文件夹中
            for font_file in font_files:
                shutil.copy(font_file, font_folder_path)

            # 清空字体选择器中的选项，并添加字体文件夹中的字体文件名
            self.ui.font_combo.clear()
            font_files = [f for f in os.listdir(font_folder_path) if f.endswith((".ttf", ".ttc", ".otf", ".woff", ".woff2"))]
            self.ui.font_combo.addItems(font_files)
            # 设置字体选择器的当前文本为第一个字体文件的文件名
            filename = os.path.basename(font_files[0])
            self.ui.font_combo.setCurrentText(filename)
    # 定义选择颜色的方法，该方法属于一个类的实例方法（self 参数指向当前实例）
    def select_color(self):
        # 定义默认颜色为黑色
        default_color = QtGui.QColor('#000000')  
        
        # 创建一个颜色选择对话框实例
        color_dialog = QtWidgets.QColorDialog()
        
        # 设置颜色选择对话框的当前颜色为默认颜色
        color_dialog.setCurrentColor(default_color)
        
        # 如果颜色选择对话框执行后用户点击了确认按钮
        if color_dialog.exec() == QtWidgets.QDialog.Accepted:
            # 获取用户选择的颜色
            color = color_dialog.selectedColor()
            
            # 如果选择的颜色有效
            if color.isValid():
                # 根据用户选择的颜色设置按钮的样式，以显示选择的背景色
                self.ui.color_button.setStyleSheet(
                    f"background-color: {color.name()}; border: none; border-radius: 5px;"
                )
                
                # 将选择的颜色名称存储在按钮的属性中
                self.ui.color_button.setProperty('selected_color', color.name())

    # 通过映射，设置以英文值保存设置，并在选择的语言中加载
    # 保存应用程序设置到指定的配置文件中
    def save_settings(self):
        # 创建一个名为 "ComicTranslate" 的应用程序设置对象，归属于 "ComicLabs" 发布者
        settings = QSettings("ComicLabs", "ComicTranslate")
        # 获取所有设置项及其值的字典
        all_settings = self.get_all_settings()

        # 遍历所有设置项及其值
        for key, value in all_settings.items():
            # 如果值是字典类型，则开始一个新的设置组
            if isinstance(value, dict):
                settings.beginGroup(key)
                # 遍历字典内部的子设置项及其值
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        settings.beginGroup(sub_key)
                        # 遍历字典内部的子子设置项及其值，将值映射为英文（如果有映射）
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            mapped_value = self.ui.value_mappings.get(sub_sub_value, sub_sub_value)
                            settings.setValue(sub_sub_key, mapped_value)
                        settings.endGroup()
                    else:
                        # 将值映射为英文（如果有映射）
                        mapped_value = self.ui.value_mappings.get(sub_value, sub_value)
                        settings.setValue(sub_key, mapped_value)
                settings.endGroup()
            else:
                # 将值映射为英文（如果有映射）
                mapped_value = self.ui.value_mappings.get(value, value)
                settings.setValue(key, mapped_value)

        # 如果选中了保存密钥，则将凭据信息分别保存
        credentials = self.get_credentials()
        save_keys = self.ui.save_keys_checkbox.isChecked()
        settings.beginGroup('credentials')
        settings.setValue('save_keys', save_keys)
        if save_keys:
            # 遍历凭据信息，并根据服务类型进行保存（将服务名翻译为英文）
            for service, cred in credentials.items():
                translated_service = self.ui.value_mappings.get(service, service)
                if translated_service == "Microsoft Azure":
                    settings.setValue(f"{translated_service}_api_key_ocr", cred['api_key_ocr'])
                    settings.setValue(f"{translated_service}_api_key_translator", cred['api_key_translator'])
                    settings.setValue(f"{translated_service}_region_translator", cred['region_translator'])
                    settings.setValue(f"{translated_service}_endpoint", cred['endpoint'])
                else:
                    settings.setValue(f"{translated_service}_api_key", cred['api_key'])
        else:
            # 如果未选中保存密钥，则清除所有凭据信息
            settings.remove('credentials')
        settings.endGroup()

    # 处理语言改变事件的方法，显示重新启动对话框
    def on_language_changed(self, new_language):
        # 如果不是正在加载设置，则显示重新启动对话框
        if not self._loading_settings:
            self.show_restart_dialog()

    # 显示重新启动对话框的方法
    def show_restart_dialog(self):
        # 创建一个消息框对象，显示重新启动应用程序的消息
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle(self.tr("Restart Required"))
        msg_box.setText(self.tr("Please restart the application for the language changes to take effect."))
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec()
```