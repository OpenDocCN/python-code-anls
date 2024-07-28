# `.\comic-translate\modules\translator.py`

```py
import numpy as np
from typing import List
from .utils.textblock import TextBlock  # 导入TextBlock工具类
from .rendering.render import cv2_to_pil  # 导入cv2_to_pil函数
from .utils.translator_utils import encode_image_array, get_raw_text, set_texts_from_json, get_llm_client  # 导入相关翻译工具函数
from .utils.pipeline_utils import get_language_code  # 导入获取语言代码的工具函数
from deep_translator import GoogleTranslator, YandexTranslator, MicrosoftTranslator  # 导入深度翻译相关库
import deepl  # 导入DeepL翻译库


class Translator:
    def __init__(self, main_page, source_lang: str = "", target_lang: str = ""):
        self.main_page = main_page  # 初始化主页面对象引用
        self.settings = main_page.settings_page  # 初始化设置页面对象引用

        self.translator_key = self.get_translator_key(self.settings.get_tool_selection('translator'))  # 获取并存储翻译工具的键值

        self.source_lang = source_lang  # 存储源语言
        self.source_lang_en = self.get_english_lang(main_page, self.source_lang)  # 获取并存储源语言的英文名
        self.target_lang = target_lang  # 存储目标语言
        self.target_lang_en = self.get_english_lang(main_page, self.target_lang)  # 获取并存储目标语言的英文名

        self.api_key = self.get_api_key(self.translator_key)  # 获取并存储翻译API的密钥
        self.client = get_llm_client(self.translator_key, self.api_key)  # 获取并存储语言模型客户端对象

        self.img_as_llm_input = self.settings.get_llm_settings()['image_input_enabled']  # 获取并存储LLM设置中的图像输入选项的状态

    def get_translator_key(self, localized_translator: str) -> str:
        # 根据本地化翻译工具名称映射到相应的键值
        translator_map = {
            self.settings.ui.tr("GPT-4o"): "GPT-4o",
            self.settings.ui.tr("GPT-4o mini"): "GPT-4o mini",
            self.settings.ui.tr("Claude-3-Opus"): "Claude-3-Opus",
            self.settings.ui.tr("Claude-3.5-Sonnet"): "Claude-3.5-Sonnet",
            self.settings.ui.tr("Claude-3-Haiku"): "Claude-3-Haiku",
            self.settings.ui.tr("Gemini-1.5-Flash"): "Gemini-1.5-Flash",
            self.settings.ui.tr("Gemini-1.5-Pro"): "Gemini-1.5-Pro",
            self.settings.ui.tr("Google Translate"): "Google Translate",
            self.settings.ui.tr("Microsoft Translator"): "Microsoft Translator",
            self.settings.ui.tr("DeepL"): "DeepL",
            self.settings.ui.tr("Yandex"): "Yandex"
        }
        return translator_map.get(localized_translator, localized_translator)  # 返回映射得到的翻译工具键值

    def get_english_lang(self, main_page, translated_lang: str) -> str:
        return main_page.lang_mapping.get(translated_lang, translated_lang)  # 根据给定语言名称获取对应的英文语言名称

    def get_llm_model(self, translator_key: str):
        model_map = {
            "GPT-4o": "gpt-4o",
            "GPT-4o mini": "gpt-4o-mini",
            "Claude-3-Opus": "claude-3-opus-20240229",
            "Claude-3.5-Sonnet": "claude-3-5-sonnet-20240620",
            "Claude-3-Haiku": "claude-3-haiku-20240307",
            "Gemini-1.5-Flash": "gemini-1.5-flash-latest",
            "Gemini-1.5-Pro": "gemini-1.5-pro-latest"
        }
        return model_map.get(translator_key)  # 根据给定的翻译工具键值返回相应的语言模型名称
    # 返回一个描述系统提示语的字符串，包括翻译源语言和目标语言
    def get_system_prompt(self, source_lang: str, target_lang: str):
        return f"""You are an expert translator who translates {source_lang} to {target_lang}. You pay attention to style, formality, idioms, slang etc and try to convey it in the way a {target_lang} speaker would understand.
        BE MORE NATURAL. NEVER USE 당신, 그녀, 그 or its Japanese equivalents.
        Specifically, you will be translating text OCR'd from a comic. The OCR is not perfect and as such you may receive text with typos or other mistakes.
        To aid you and provide context, You may be given the image of the page and/or extra context about the comic. You will be given a json string of the detected text blocks and the text to translate. Return the json string with the texts translated. DO NOT translate the keys of the json. For each block:
        - If it's already in {target_lang} or looks like gibberish, OUTPUT IT AS IT IS instead
        - DO NOT give explanations
        Do Your Best! I'm really counting on you."""

    # 使用给定的用户提示、模型、系统提示和图像来获取GPT翻译
    def get_gpt_translation(self, user_prompt: str, model: str, system_prompt: str, image: np.ndarray):
        # 将图像编码成base64格式
        encoded_image = encode_image_array(image)

        # 根据需求构建消息
        if self.img_as_llm_input:
            message = [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}]}
                ]
        else:
            message = [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
                ]

        # 发送请求并获取响应
        response = self.client.chat.completions.create(
            model=model,
            messages=message,
            temperature=1,
            max_tokens=1000,
        )

        # 提取翻译结果
        translated = response.choices[0].message.content
        return translated
    
    # 使用给定的用户提示、模型、系统提示和图像来获取Claude翻译
    def get_claude_translation(self, user_prompt: str, model: str, system_prompt: str, image: np.ndarray):
        # 将图像编码成base64格式
        encoded_image = encode_image_array(image)
        media_type = "image/png"

        # 根据需求构建消息
        if self.img_as_llm_input:
            message = [
                {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded_image}}]}
            ]
        else:
            message = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]

        # 发送请求并获取响应
        response = self.client.messages.create(
            model=model,
            system=system_prompt,
            messages=message,
            temperature=1,
            max_tokens=1000,
        )

        # 提取翻译结果
        translated = response.content[0].text
        return translated
    # 定义一个方法，用于获取Gemini翻译服务的响应
    def get_gemini_translation(self, user_prompt: str, model: str, system_prompt: str, image):
        # 配置生成文本的参数，包括温度、top_p、top_k和最大输出token数
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 1000,
        }
        
        # 安全设置列表，定义了不同类型有害内容的阈值设定
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]

        # 创建一个生成模型实例，使用给定的模型名称、生成配置、系统提示和安全设置
        model_instance = self.client.GenerativeModel(model_name=model, generation_config=generation_config, system_instruction=system_prompt, safety_settings=safety_settings)
        
        # 开始与生成模型的对话
        chat = model_instance.start_chat(history=[])

        # 如果图像作为语言模型的输入
        if self.img_as_llm_input:
            # 发送包含图像和用户提示的消息到聊天模型
            chat.send_message([image, user_prompt])
        else:
            # 否则，只发送用户提示的消息
            chat.send_message([user_prompt])
        
        # 获取聊天模型的最后一条文本响应
        response = chat.last.text

        # 返回生成的响应文本
        return response
    # 定义一个方法用于翻译文本块列表中的文本，使用给定的图像和额外上下文信息
    def translate(self, blk_list: List[TextBlock], image: np.ndarray, extra_context: str):
        # 获取源语言的语言代码
        source_lang_code = get_language_code(self.source_lang_en)
        # 获取目标语言的语言代码
        target_lang_code = get_language_code(self.target_lang_en)

        # 非基于LLM的翻译
        if self.translator_key in ["Google Translate", "DeepL", "Yandex", "Microsoft Translator"]:
            # 遍历文本块列表
            for blk in blk_list:
                # 如果源语言代码中包含'zh'或者源语言代码是'ja'，则去除文本中的空格
                text = blk.text.replace(" ", "") if 'zh' in source_lang_code.lower() or source_lang_code.lower() == 'ja' else blk.text
                # 根据选择的翻译服务进行翻译
                if self.translator_key == "Google Translate":
                    translation = GoogleTranslator(source='auto', target=target_lang_code).translate(text)
                elif self.translator_key == "Yandex":
                    translation = YandexTranslator(source='auto', target=target_lang_code, api_key=self.api_key).translate(text)
                elif self.translator_key == "Microsoft Translator":
                    credentials = self.settings.get_credentials("Microsoft Azure")
                    region = credentials['region_translator']
                    translation = MicrosoftTranslator(source_lang_code, target_lang_code, self.api_key, region).translate(text)
                else:  # DeepL
                    trans = deepl.Translator(self.api_key)
                    # 根据目标语言的不同选择不同的翻译目标语言
                    if self.target_lang == self.main_page.tr("Simplified Chinese"):
                        result = trans.translate_text(text, source_lang=source_lang_code, target_lang="zh")
                    elif self.target_lang == self.main_page.tr("English"):
                        result = trans.translate_text(text, source_lang=source_lang_code, target_lang="EN-US")
                    else:
                        result = trans.translate_text(text, source_lang=source_lang_code, target_lang=target_lang_code)
                    translation = result.text

                # 如果翻译结果不为空，则将翻译结果赋给文本块的翻译属性
                if translation is not None:
                    blk.translation = translation
        
        # 处理基于LLM的翻译
        else:
            # 获取LLM模型
            model = self.get_llm_model(self.translator_key)
            # 获取整个文本块列表的原始文本
            entire_raw_text = get_raw_text(blk_list)
            # 获取系统提示
            system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
            # 构建用户提示，包含额外上下文信息和要翻译的文本
            user_prompt = f"{extra_context}\nMake the translation sound as natural as possible.\nTranslate this:\n{entire_raw_text}"

            # 根据选择的LLM类型选择相应的翻译方法
            if 'GPT' in self.translator_key:
                entire_translated_text = self.get_gpt_translation(user_prompt, model, system_prompt, image)
            elif 'Claude' in self.translator_key:
                entire_translated_text = self.get_claude_translation(user_prompt, model, system_prompt, image)
            elif 'Gemini' in self.translator_key:
                image = cv2_to_pil(image)
                entire_translated_text = self.get_gemini_translation(user_prompt, model, system_prompt, image)

            # 将翻译结果设置到文本块列表中
            set_texts_from_json(blk_list, entire_translated_text)

        # 返回翻译后的文本块列表
        return blk_list
    # 获取 API 密钥的方法，根据提供的翻译服务商关键词从凭据中获取对应的 API 密钥
    def get_api_key(self, translator_key: str):
        # 获取存储在系统设置中的凭据信息
        credentials = self.settings.get_credentials()

        # 初始化 API 密钥为空字符串
        api_key = ""

        # 根据提供的翻译服务商关键词选择对应的 API 密钥
        if 'GPT' in translator_key:
            api_key = credentials['Open AI GPT']['api_key']
        elif 'Claude' in translator_key:
            api_key = credentials['Anthropic Claude']['api_key']
        elif 'Gemini' in translator_key:
            api_key = credentials['Google Gemini']['api_key']
        else:
            # 如果提供的关键词不匹配上述任何一种情况，则从预定义的映射中获取 API 密钥
            map = {
                "Microsoft Translator": credentials['Microsoft Azure']['api_key_translator'],
                "DeepL": credentials['DeepL']['api_key'],
                "Yandex": credentials['Yandex']['api_key'],
            }
            api_key = map.get(translator_key)

        # 返回获取的 API 密钥
        return api_key
```