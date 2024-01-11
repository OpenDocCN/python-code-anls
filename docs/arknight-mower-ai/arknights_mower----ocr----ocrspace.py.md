# `arknights-mower\arknights_mower\ocr\ocrspace.py`

```
# 导入必要的模块
import base64
import traceback
import cv2
import numpy
import requests
# 导入自定义的日志模块和识别错误模块
from ..utils.log import logger
from ..utils.recognize import RecognizeError
# 导入工具模块中的修复函数
from .utils import fix

# 定义语言类，包含各种语言的缩写常量
class Language:
    Arabic = 'ara'
    Bulgarian = 'bul'
    Chinese_Simplified = 'chs'
    Chinese_Traditional = 'cht'
    Croatian = 'hrv'
    Danish = 'dan'
    Dutch = 'dut'
    English = 'eng'
    Finnish = 'fin'
    French = 'fre'
    German = 'ger'
    Greek = 'gre'
    Hungarian = 'hun'
    Korean = 'kor'
    Italian = 'ita'
    Japanese = 'jpn'
    Norwegian = 'nor'
    Polish = 'pol'
    Portuguese = 'por'
    Russian = 'rus'
    Slovenian = 'slv'
    Spanish = 'spa'
    Swedish = 'swe'
    Turkish = 'tur'

# 定义 API 类
class API:
    def __init__(
        self,
        endpoint='https://api.ocr.space/parse/image',  # API 的默认端点
        api_key='helloworld',  # API 的默认密钥
        language=Language.Chinese_Simplified,  # 默认语言为简体中文
        **kwargs,  # 其他参数
    ):
        """
        :param endpoint: API endpoint to contact  # 联系的 API 端点
        :param api_key: API key string  # API 密钥字符串
        :param language: document language  # 文档语言
        :param **kwargs: other settings to API  # 其他 API 设置
        """
        self.timeout = (5, 10)  # 设置超时时间
        self.endpoint = endpoint  # 设置 API 端点
        self.payload = {
            'isOverlayRequired': True,  # 是否需要覆盖
            'apikey': api_key,  # API 密钥
            'language': language,  # 文档语言
            **kwargs  # 其他参数
        }
    # 解析 OCR 结果
    def _parse(self, raw):
        # 记录原始数据
        logger.debug(raw)
        # 如果原始数据类型为字符串，抛出识别错误异常
        if type(raw) == str:
            raise RecognizeError(raw)
        # 如果处理过程中出现错误，抛出识别错误异常
        if raw['IsErroredOnProcessing']:
            raise RecognizeError(raw['ErrorMessage'][0])
        # 如果解析结果中没有文本叠加信息，抛出识别错误异常
        if raw['ParsedResults'][0].get('TextOverlay') is None:
            raise RecognizeError('No Result')
        # 从解析结果中提取文本信息和位置坐标
        ret = [x['LineText']
               for x in raw['ParsedResults'][0]['TextOverlay']['Lines']]
        return ret

    # 从本地路径处理图像
    def ocr_file(self, fp):
        """
        Process image from a local path.
        :param fp: A path or pointer to your file
        :return: Result in JSON format
        """
        # 打开文件流
        with (open(fp, 'rb') if type(fp) == str else fp) as f:
            # 发送 POST 请求进行 OCR 识别
            r = requests.post(
                self.endpoint,
                files={'filename': f},
                data=self.payload,
                timeout=self.timeout,
            )
        # 解析 OCR 结果
        return self._parse(r.json())

    # 从 URL 处理图像
    def ocr_url(self, url):
        """
        Process an image at a given URL.
        :param url: Image url
        :return: Result in JSON format.
        """
        # 构造请求数据
        data = self.payload
        data['url'] = url
        # 发送 POST 请求进行 OCR 识别
        r = requests.post(
            self.endpoint,
            data=data,
            timeout=self.timeout,
        )
        # 解析 OCR 结果
        return self._parse(r.json())
    # 以 base64 格式处理给定的图像
    def ocr_base64(self, base64image):
        # 获取初始数据
        data = self.payload
        # 将图像以 base64 格式添加到数据中
        data['base64Image'] = base64image
        # 发送 POST 请求到指定的端点，设置超时时间
        r = requests.post(
            self.endpoint,
            data=data,
            timeout=self.timeout,
        )
        # 解析返回的 JSON 格式结果
        return self._parse(r.json())

    # 处理给定的图像
    def ocr_image(self, image: numpy.ndarray):
        # 获取初始数据
        data = self.payload
        # 将图像编码为 base64 格式，并添加到数据中
        data['base64Image'] = 'data:image/jpg;base64,' + \
            base64.b64encode(cv2.imencode('.jpg', image)[1].tobytes()).decode()

        # 设置重试次数
        retry_times = 1
        # 循环直到成功发送请求或达到重试次数上限
        while True:
            try:
                # 发送 POST 请求到指定的端点，设置超时时间
                r = requests.post(
                    self.endpoint,
                    data=data,
                    timeout=self.timeout,
                )
                # 如果成功发送请求，则跳出循环
                break
            except Exception as e:
                # 记录警告日志
                logger.warning(e)
                # 记录调试信息
                logger.debug(traceback.format_exc())
                # 减少重试次数
                retry_times -= 1
                # 如果还有重试次数，则继续重试
                if retry_times > 0:
                    logger.warning('重试中……')
                else:
                    # 如果重试次数用尽，则记录警告日志并返回空列表
                    logger.warning('无网络或网络故障，无法连接到 OCR Space')
                    return []
        try:
            # 解析返回的 JSON 格式结果
            return self._parse(r.json())
        except Exception as e:
            # 记录调试信息
            logger.debug(e)
            # 返回空列表
            return []

    # 对给定图像进行预测
    def predict(self, image, scope):
        # 裁剪图像并进行 OCR 处理
        ret = self.ocr_image(
            image[scope[0][1]:scope[2][1], scope[0][0]:scope[2][0]])
        # 如果返回结果为空列表，则返回 None
        if len(ret) == 0:
            return None
        # 对返回结果进行修正并返回
        return fix(ret[0])
```