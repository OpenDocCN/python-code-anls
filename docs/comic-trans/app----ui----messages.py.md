# `.\comic-translate\app\ui\messages.py`

```py
# 导入自定义消息模块 MMessage
from .dayu_widgets.message import MMessage
# 导入 Qt 核心模块 QCoreApplication
from PySide6.QtCore import QCoreApplication

# 定义 Messages 类，处理消息相关的静态方法
class Messages:
    # 显示 API 密钥翻译器错误消息
    @staticmethod
    def show_api_key_translator_error(parent):
        MMessage.error(
            # 使用 QCoreApplication 进行国际化翻译，显示需要 API 密钥的翻译器错误信息
            text=QCoreApplication.translate("Messages", "An API Key is required for the selected translator.\nGo to Settings > Credentials to set one"),
            parent=parent,
            duration=None,
            closable=True
        )

    # 显示 API 密钥 OCR 错误消息
    @staticmethod
    def show_api_key_ocr_error(parent):
        MMessage.error(
            # 使用 QCoreApplication 进行国际化翻译，显示需要 API 密钥的 OCR 错误信息
            text=QCoreApplication.translate("Messages", "An API Key is required for the selected OCR.\nGo to Settings > Credentials to set one"),
            parent=parent,
            duration=None,
            closable=True
        )

    # 显示 API 密钥 OCR GPT-4v 错误消息
    @staticmethod
    def show_api_key_ocr_gpt4v_error(parent):
        MMessage.error(
            # 使用 QCoreApplication 进行国际化翻译，显示默认 OCR 为 GPT-4o 时需要 API 密钥的错误信息
            text=QCoreApplication.translate("Messages", "Default OCR for one of the selected Source Languages is GPT-4o\nwhich requires an API Key. Go to Settings > Credentials > GPT to set one"),
            parent=parent,
            duration=None,
            closable=True
        )

    # 显示端点 URL 错误消息
    @staticmethod
    def show_endpoint_url_error(parent):
        MMessage.error(
            # 使用 QCoreApplication 进行国际化翻译，显示需要端点 URL 的 Microsoft OCR 错误信息
            text=QCoreApplication.translate("Messages", "An Endpoint URL is required for Microsoft OCR.\nGo to Settings > Credentials > Microsoft to set one"),
            parent=parent,
            duration=None,
            closable=True
        )

    # 显示 DeepL 不支持繁体中文的错误消息
    @staticmethod
    def show_deepl_ch_error(parent):
        MMessage.error(
            # 使用 QCoreApplication 进行国际化翻译，显示 DeepL 不支持繁体中文的错误信息
            text=QCoreApplication.translate("Messages", "DeepL does not translate to Traditional Chinese"),
            parent=parent,
            duration=None,
            closable=True
        )

    # 显示 Google 翻译不支持巴西葡萄牙语的错误消息
    @staticmethod
    def show_googlet_ptbr_error(parent):
        MMessage.error(
            # 使用 QCoreApplication 进行国际化翻译，显示 Google 翻译不支持巴西葡萄牙语的错误信息
            text=QCoreApplication.translate("Messages", "Google Translate does not support Brazillian Portuguese"),
            parent=parent,
            duration=None,
            closable=True
        )

    # 显示翻译完成的成功消息
    @staticmethod
    def show_translation_complete(parent):
        MMessage.success(
            # 使用 QCoreApplication 进行国际化翻译，显示翻译完成的成功消息
            text=QCoreApplication.translate("Messages", "Comic has been Translated!"),
            parent=parent,
            duration=None,
            closable=True
        )

    # 显示未选择字体的错误消息
    @staticmethod
    def select_font_error(parent):
        MMessage.error(
            # 使用 QCoreApplication 进行国际化翻译，显示未选择字体的错误信息
            text=QCoreApplication.translate("Messages", "No Font selected.\nGo to Settings > Text Rendering > Font to select or import one "),
            parent=parent,
            duration=None,
            closable=True
        )
```