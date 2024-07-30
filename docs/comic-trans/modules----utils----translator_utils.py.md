# `.\comic-translate\modules\utils\translator_utils.py`

```py
import cv2  # 导入OpenCV图像处理库
import base64  # 导入base64编解码模块
import json  # 导入处理JSON格式数据的模块
import re  # 导入正则表达式模块
import stanza  # 导入stanza自然语言处理库
import numpy as np  # 导入处理数组和矩阵的数学库
from openai import OpenAI  # 从OpenAI库导入OpenAI类
import google.generativeai as genai  # 导入Google的生成人工智能模块
import anthropic  # 导入anthropic API
from .textblock import TextBlock  # 导入当前目录下的textblock模块中的TextBlock类
from typing import List  # 从typing模块导入List类型


def encode_image_array(img_array: np.ndarray):
    # 将图像数组编码为PNG格式的图像字节流
    _, img_bytes = cv2.imencode('.png', img_array)
    # 将图像字节流转换为base64编码的字符串并解码为UTF-8格式
    return base64.b64encode(img_bytes).decode('utf-8')


def get_llm_client(translator: str, api_key: str):
    # 根据translator参数选择不同的语言模型客户端
    if 'GPT' in translator:
        client = OpenAI(api_key=api_key)  # 使用OpenAI的API键创建客户端
    elif 'Claude' in translator:
        client = anthropic.Anthropic(api_key=api_key)  # 使用Anthropic的API键创建客户端
    elif 'Gemini' in translator:
        client = genai  # 使用Google的生成AI模块作为客户端
        client.configure(api_key=api_key)  # 配置生成AI客户端的API键
    else:
        client = None  # 如果无法识别translator参数，返回None
    
    return client  # 返回选择的语言模型客户端


def get_raw_text(blk_list: List[TextBlock]):
    rw_txts_dict = {}  # 创建空字典以存储文本块的原始文本
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"  # 创建每个文本块的键，如"block_0"、"block_1"等
        rw_txts_dict[block_key] = blk.text  # 将文本块的原始文本存储在字典中
    
    # 将原始文本字典转换为格式化的JSON字符串
    raw_texts_json = json.dumps(rw_txts_dict, ensure_ascii=False, indent=4)
    
    return raw_texts_json  # 返回格式化后的原始文本JSON字符串


def get_raw_translation(blk_list: List[TextBlock]):
    rw_translations_dict = {}  # 创建空字典以存储文本块的翻译文本
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"  # 创建每个文本块的键，如"block_0"、"block_1"等
        rw_translations_dict[block_key] = blk.translation  # 将文本块的翻译文本存储在字典中
    
    # 将翻译文本字典转换为格式化的JSON字符串
    raw_translations_json = json.dumps(rw_translations_dict, ensure_ascii=False, indent=4)
    
    return raw_translations_json  # 返回格式化后的翻译文本JSON字符串


def set_texts_from_json(blk_list: List[TextBlock], json_string: str):
    match = re.search(r"\{[\s\S]*\}", json_string)  # 在输入字符串中寻找JSON格式的数据
    if match:
        json_string = match.group(0)  # 提取匹配到的JSON字符串
        translation_dict = json.loads(json_string)  # 将JSON字符串解析为Python字典
        
        for idx, blk in enumerate(blk_list):
            block_key = f"block_{idx}"  # 创建每个文本块的键，如"block_0"、"block_1"等
            if block_key in translation_dict:
                blk.translation = translation_dict[block_key]  # 根据JSON中的键更新文本块的翻译文本
            else:
                print(f"Warning: {block_key} not found in JSON string.")  # 如果JSON中未找到对应的键，输出警告
    else:
        print("No JSON found in the input string.")  # 如果未找到JSON格式的数据，输出提示信息


def format_translations(blk_list: List[TextBlock], trg_lng_cd: str, upper_case: bool = True):
    # 此函数尚未完整，需要根据具体情况补充注释
    pass
    # 遍历块列表中的每一个块对象
    for blk in blk_list:
        # 获取当前块对象的翻译文本
        translation = blk.translation
        
        # 如果目标语言代码中包含 'zh' 或 'ja'
        if any(lang in trg_lng_cd.lower() for lang in ['zh', 'ja']):

            # 如果目标语言代码为 'zh-TW'，转换为 'zh-Hant'
            if trg_lng_cd == 'zh-TW':
                trg_lng_cd = 'zh-Hant'
            # 如果目标语言代码为 'zh-CN'，转换为 'zh-Hans'
            elif trg_lng_cd == 'zh-CN':
                trg_lng_cd = 'zh-Hans'
            # 否则保持目标语言代码不变
            else:
                trg_lng_cd = trg_lng_cd
            
            # 下载并初始化指定语言的 Stanza 管道，包括分词器
            stanza.download(trg_lng_cd, processors='tokenize')
            nlp = stanza.Pipeline(trg_lng_cd, processors='tokenize')
            # 对翻译文本进行处理，生成文档对象
            doc = nlp(translation)
            seg_result = []
            # 遍历文档中的句子和单词，生成分词结果列表
            for sentence in doc.sentences:
                for word in sentence.words:
                    seg_result.append(word.text)
            # 将分词结果拼接成字符串，处理标点符号前的空格
            translation = ''.join(word if word in ['.', ','] else f' {word}' for word in seg_result).lstrip()
            # 更新当前块对象的翻译文本为处理后的结果
            blk.translation = translation
        
        else:
            # 如果不是目标语言为中文或日文
            # 如果需要将翻译文本转为大写且当前文本不全为大写，则转为大写
            if upper_case and not translation.isupper():
                blk.translation = translation.upper()
            # 如果不需要将翻译文本转为大写且当前文本全为大写，则首字母大写
            elif not upper_case and translation.isupper():
                blk.translation = translation.capitalize()
            # 否则保持翻译文本不变
            else:
                blk.translation = translation
# 检查文本块列表中是否存在文本内容
def is_there_text(blk_list: List[TextBlock]) -> bool:
    # 使用 Python 的 any() 函数检查列表中是否有至少一个文本块的文本内容不为空
    return any(blk.text for blk in blk_list)
```