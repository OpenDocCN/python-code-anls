# `.\comic-translate\modules\ocr\ocr.py`

```py
import numpy as np
import base64, json
import easyocr
import cv2
import requests
from typing import List
from ..utils.translator_utils import get_llm_client
from ..utils.textblock import TextBlock, adjust_text_line_coordinates
from ..utils.pipeline_utils import lists_to_blk_list, ensure_within_bounds
from ..utils.download import get_models, manga_ocr_data, pororo_data
from ..ocr.manga_ocr.manga_ocr import MangaOcr
from ..ocr.pororo.main import PororoOcr
from ..utils.pipeline_utils import language_codes

# 定义全局变量，指定Manga OCR模型的路径
manga_ocr_path = 'models/ocr/manga-ocr-base'
        
class OCRProcessor:
    def __init__(self, main_page, source_lang: str):
        # 初始化OCR处理器实例
        self.main_page = main_page
        self.settings = main_page.settings_page
        self.source_lang = source_lang
        # 转换源语言为英语
        self.source_lang_english = self.get_english_lang(main_page, self.source_lang)
        # 获取当前选择的OCR模型
        self.ocr_model = self.settings.get_tool_selection('ocr')
        # 判断是否使用Microsoft OCR
        self.microsoft_ocr = True if self.ocr_model == self.settings.ui.tr("Microsoft OCR") else False
        # 判断是否使用Google Cloud Vision OCR
        self.google_ocr = True if self.ocr_model == self.settings.ui.tr("Google Cloud Vision") else False
        # 根据GPU是否启用选择计算设备
        self.device = 'cuda' if self.settings.is_gpu_enabled() else 'cpu'

        # 如果源语言为以下语言且OCR模型为默认模型，则启用GPT OCR
        if self.source_lang_english in ["French", "German", "Dutch", "Russian", "Spanish", "Italian"] and self.ocr_model == self.settings.ui.tr("Default"):
            self.gpt_ocr = True
        else:
            self.gpt_ocr = False

    def get_english_lang(self, main_page, translated_lang: str) -> str:
        # 根据主页面提供的语言映射，获取对应的英语语言名称
        return main_page.lang_mapping.get(translated_lang, translated_lang)

    def set_source_orientation(self, blk_list: List[TextBlock]):
        # 设置源漫画文本块列表的语言方向
        # 在代码的其他部分，源语言会设置文本的方向。稍后可能会添加专用的方向变量。

        # 根据源语言英语名称获取其对应的语言代码
        source_lang_code = language_codes[self.source_lang_english]
        # 遍历文本块列表，为每个文本块设置源语言代码
        for blk in blk_list:
            blk.source_lang = source_lang_code
    # 定义一个处理函数，用于对图像进行OCR识别，并根据条件选择合适的OCR引擎进行处理
    def process(self, img: np.ndarray, blk_list: List[TextBlock]):
        # 设置源图像的方向，根据文本块列表进行识别
        self.set_source_orientation(blk_list)
        
        # 如果源语言为中文，并且没有选择Microsoft OCR和Google OCR，则使用PaddleOCR进行识别
        if self.source_lang == self.settings.ui.tr('Chinese') and (not self.microsoft_ocr and not self.google_ocr):
            return self._ocr_paddle(img, blk_list)
        
        # 如果选择了Microsoft OCR，则从设置中获取Microsoft Azure的凭证，调用Microsoft OCR进行识别
        elif self.microsoft_ocr:
            credentials = self.settings.get_credentials(self.settings.ui.tr("Microsoft Azure"))
            api_key = credentials['api_key_ocr']
            endpoint = credentials['endpoint']
            return self._ocr_microsoft(img, blk_list, api_key=api_key, 
                                   endpoint=endpoint)
        
        # 如果选择了Google OCR，则从设置中获取Google Cloud的凭证，调用Google OCR进行识别
        elif self.google_ocr:
            credentials = self.settings.get_credentials(self.settings.ui.tr("Google Cloud"))
            api_key = credentials['api_key']
            return self._ocr_google(img, blk_list, api_key=api_key)

        # 如果选择了GPT OCR，则从设置中获取Open AI GPT的凭证，创建GPT客户端，并调用GPT OCR进行识别
        elif self.gpt_ocr:
            credentials = self.settings.get_credentials(self.settings.ui.tr("Open AI GPT"))
            api_key = credentials['api_key']
            gpt_client = get_llm_client('GPT', api_key)
            return self._ocr_gpt(img, blk_list, gpt_client)

        # 默认情况下，使用默认的OCR处理函数进行识别，传入源语言和设备参数
        else:
            return self._ocr_default(img, blk_list, self.source_lang, self.device)

    # 使用PaddleOCR进行中文文本识别的私有方法
    def _ocr_paddle(self, img: np.ndarray, blk_list: List[TextBlock]):
        # 导入PaddleOCR模块
        from paddleocr import PaddleOCR
        
        # 创建PaddleOCR实例，设置语言为中文
        ch_ocr = PaddleOCR(lang='ch')
        
        # 对图像进行OCR识别，获取结果
        result = ch_ocr.ocr(img)
        
        # 如果识别结果不为空，则处理识别结果，提取文本框坐标和文本内容
        result = result[0]  # 只使用第一个结果（通常情况下只有一个）
        
        # 提取每个文本框的坐标信息，并转换为简化的文本框边界框形式
        texts_bboxes = [tuple(coord for point in bbox for coord in point) for bbox, _ in result] if result else []
        condensed_texts_bboxes = [(x1, y1, x2, y2) for (x1, y1, x2, y1_, x2_, y2, x1_, y2_) in texts_bboxes]

        # 提取每个文本框的文本内容
        texts_string = [line[1][0] for line in result] if result else []

        # 将处理后的文本框信息和文本内容更新到输入的文本块列表中
        blk_list = lists_to_blk_list(blk_list, condensed_texts_bboxes, texts_string)

        # 返回更新后的文本块列表
        return blk_list
    # 使用 Microsoft Azure 的图像分析客户端来进行 OCR（光学字符识别），分析给定的图像并返回文本块列表
    def _ocr_microsoft(self, img: np.ndarray, blk_list: List[TextBlock], api_key: str, endpoint: str):
        # 初始化文本框和文本字符串列表
        texts_bboxes = []
        texts_string = []

        # 创建 Microsoft Azure 的图像分析客户端
        client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        # 将图像编码为 PNG 格式的字节流
        image_buffer = cv2.imencode('.png', img)[1].tobytes()
        # 使用 Azure 客户端分析图像，并指定需要读取文本的视觉特征
        result = client.analyze(image_data=image_buffer, visual_features=[VisualFeatures.READ])

        # 如果分析结果中存在可读取的文本
        if result.read is not None:
            # 遍历每一行文本块
            for line in result.read.blocks[0].lines:
                # 获取文本行的边界多边形顶点
                vertices = line.bounding_polygon
                
                # 确保所有顶点都有 'x' 和 'y' 坐标
                if all('x' in vertex and 'y' in vertex for vertex in vertices):
                    # 提取文本行的左上角和右下角坐标
                    x1 = vertices[0]['x']
                    y1 = vertices[0]['y']
                    x2 = vertices[2]['x']
                    y2 = vertices[2]['y']
                    
                    # 将文本行的边界框坐标和文本内容添加到对应列表中
                    texts_bboxes.append((x1, y1, x2, y2))
                    texts_string.append(line.text)

        # 将提取的文本块和文本字符串列表转换为文本块列表
        blk_list = lists_to_blk_list(blk_list, texts_bboxes, texts_string)

        # 返回更新后的文本块列表
        return blk_list

    # 使用 Google Cloud Vision API 进行 OCR，分析给定的图像并返回文本块列表
    def _ocr_google(self, img: np.ndarray, blk_list: List[TextBlock], api_key: str):
        # 初始化文本框和文本字符串列表
        texts_bboxes = []
        texts_string = []

        # 将图像编码为 PNG 格式的字节流
        cv2_to_google = cv2.imencode('.png', img)[1].tobytes()
        # 构建 Google Cloud Vision API 请求的 payload
        payload = {
            "requests": [
                {
                    "image": {
                        "content": base64.b64encode(cv2_to_google).decode('utf-8')
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }
        headers = {"Content-Type": "application/json"}
        # 发送 POST 请求到 Google Cloud Vision API
        response = requests.post("https://vision.googleapis.com/v1/images:annotate", headers=headers, params={"key": api_key}, data=json.dumps(payload))
        # 解析 API 返回的 JSON 结果
        result = response.json()
        # 获取文本注释列表
        texts = result['responses'][0]['textAnnotations']

        # 如果存在识别到的文本注释
        if texts is not None:
            # 遍历每个文本注释
            for index, text in enumerate(texts):
                # 获取文本注释的边界多边形顶点
                vertices = text['boundingPoly']['vertices']
                # 跳过第一个文本注释（通常是整个图像的注释）
                if index == 0:
                    continue

                # 确保所有顶点都有 'x' 和 'y' 坐标
                if all('x' in vertex and 'y' in vertex for vertex in vertices):
                    # 提取文本注释的左上角和右下角坐标
                    x1 = vertices[0]['x']
                    y1 = vertices[0]['y']
                    x2 = vertices[2]['x']
                    y2 = vertices[2]['y']
                    
                    # 提取文本注释的内容
                    string = text['description']
                    # 将文本注释的边界框坐标和文本内容添加到对应列表中
                    texts_bboxes.append((x1, y1, x2, y2))
                    texts_string.append(string)

        # 将提取的文本块和文本字符串列表转换为文本块列表
        blk_list = lists_to_blk_list(blk_list, texts_bboxes, texts_string)

        # 返回更新后的文本块列表
        return blk_list
    # 定义一个方法 `_ocr_gpt`，接受参数包括图片数组 `img`、文本块列表 `blk_list`、GPT 客户端 `client`，
    # 以及一个可选的扩展百分比参数 `expansion_percentage`。
    def _ocr_gpt(self, img: np.ndarray, blk_list: List[TextBlock], client, expansion_percentage: int = 0):
        # 获取图像的高度和宽度
        im_h, im_w = img.shape[:2]
        
        # 遍历文本块列表中的每个文本块 `blk`
        for blk in blk_list:
            # 如果文本块有气泡的坐标信息，则使用气泡的坐标
            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = blk.bubble_xyxy
            else:
                # 否则，根据文本行的坐标信息调整坐标，并根据扩展百分比确保坐标在图像边界内
                x1, y1, x2, y2 = adjust_text_line_coordinates(blk.xyxy, expansion_percentage, expansion_percentage)
                x1, y1, x2, y2 = ensure_within_bounds((x1, y1, x2, y2), im_w, im_h, expansion_percentage, expansion_percentage)

            # 检查坐标是否有效，并且边界框不会超出图像范围
            if x1 < x2 and y1 < y2:
                # 从原始图像中提取感兴趣区域（ROI），转换为 PNG 格式的图像数据
                cv2_to_gpt = cv2.imencode('.png', img[y1:y2, x1:x2])[1]
                # 将 PNG 格式的图像数据编码为 base64 字符串
                cv2_to_gpt = base64.b64encode(cv2_to_gpt).decode('utf-8')
                # 使用 GPT 模型进行 OCR，获取文本结果
                text = get_gpt_ocr(cv2_to_gpt, client)
                # 将识别的文本赋值给文本块 `blk` 的 `text` 属性
                blk.text = text

        # 返回更新后的文本块列表 `blk_list`
        return blk_list
    # 定义默认的OCR方法，用于识别图像中的文本区域
    def _ocr_default(self, img: np.ndarray, blk_list: List[TextBlock], source_language: str, device: str, expansion_percentage: int = 5):
        # 根据设备确定是否使用 GPU 加速
        gpu_state = False if device == 'cpu' else True

        # 获取图像的高度和宽度
        im_h, im_w = img.shape[:2]
        
        # 遍历文本块列表
        for blk in blk_list:
            # 如果文本块具有泡泡状边界框，则使用泡泡边界框坐标
            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = blk.bubble_xyxy
            else:
                # 否则，根据文本线坐标调整并确保在图像边界内
                x1, y1, x2, y2 = adjust_text_line_coordinates(blk.xyxy, expansion_percentage, expansion_percentage)
                x1, y1, x2, y2 = ensure_within_bounds((x1, y1, x2, y2), im_w, im_h, expansion_percentage, expansion_percentage)

            # 检查坐标是否有效且边界框未超出图像范围
            if x1 < x2 and y1 < y2:
                # 如果源语言是日语
                if source_language == self.main_page.tr('Japanese'):
                    # 获取日语OCR模型并进行文本识别
                    get_models(manga_ocr_data)
                    manga_ocr = MangaOcr(pretrained_model_name_or_path=manga_ocr_path, device=device)
                    blk.text = manga_ocr(img[y1:y2, x1:x2])

                # 如果源语言是英语
                elif source_language == self.main_page.tr('English'):
                    # 使用easyocr读取英语文本
                    reader = easyocr.Reader(['en'], gpu=gpu_state)
                    result = reader.readtext(img[y1:y2, x1:x2], paragraph=True)
                    texts = []
                    for r in result:
                        if r is None:
                            continue
                        texts.append(r[1])
                    text = ' '.join(texts)
                    blk.text = text
                
                # 如果源语言是韩语
                elif source_language == self.main_page.tr('Korean'):
                    # 获取韩语OCR模型并运行OCR
                    get_models(pororo_data)
                    kor_ocr = PororoOcr()
                    kor_ocr.run_ocr(img[y1:y2, x1:x2])
                    result = kor_ocr.get_ocr_result()
                    descriptions = result['description']
                    all_descriptions = ' '.join(descriptions)
                    blk.text = all_descriptions     

            else:
                # 若边界框无效，则打印错误信息并将文本块的文本置空
                print('Invalid textbbox to target img')
                blk.text = ['']

        # 返回处理后的文本块列表
        return blk_list
# 定义名为 get_gpt_ocr 的函数，用于获取 OCR（光学字符识别）结果
def get_gpt_ocr(base64_image: str, client):
    # 使用 client 对象调用 chat.completions.create 方法，发起请求获取 OCR 结果
    response = client.chat.completions.create(
        model="gpt-4o",  # 指定模型为 "gpt-4o"
        messages=[
            {
                "role": "user",  # 用户角色
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},  # 图像的 base64 编码 URL
                    {"type": "text", "text": """ Write out the text in this image. Do NOT Translate. Do not write anything else"""},  # 要识别的图像中的文本内容
                ]
            }
        ],
        max_tokens=1000,  # 最大生成的 token 数量
    )
    # 从响应中获取第一个选择项的消息内容
    text = response.choices[0].message.content
    # 如果文本中包含换行符，则替换为空格，否则保持原样
    text = text.replace('\n', ' ') if '\n' in text else text
    # 返回识别出的文本结果
    return text
```