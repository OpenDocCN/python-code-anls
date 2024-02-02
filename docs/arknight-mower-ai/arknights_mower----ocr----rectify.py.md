# `arknights-mower\arknights_mower\ocr\rectify.py`

```py
# 从上级目录的 data 模块中导入 ocr_error 对象
from ..data import ocr_error
# 从上级目录的 utils 模块中导入 config 对象
from ..utils import config
# 从上级目录的 utils 模块中导入 log 模块中的 logger 对象
from ..utils.log import logger
# 从当前目录的 ocrspace 模块中导入 API 和 Language 对象
from .ocrspace import API, Language

# 定义函数 ocr_rectify，用于调用在线 OCR 校正本地 OCR 得到的错误结果，并返回校正后的识别结果
# 若在线 OCR 依旧无法正确识别则返回 None
def ocr_rectify(img, pre, valid, text=''):
    # 记录警告日志，指示正在调用在线 OCR 处理异常结果
    logger.warning(f'{text}识别异常：正在调用在线 OCR 处理异常结果……')

    # 声明 ocronline 为全局变量
    global ocronline
    # 打印 config 对象的内容
    print(config)
    # 初始化 ocronline 对象，使用 config 中的 OCR_APIKEY 和 Language.Chinese_Simplified
    ocronline = API(api_key=config.OCR_APIKEY, language=Language.Chinese_Simplified)
    # 获取本地 OCR 得到的错误结果的范围
    pre_res = pre[1]
    # 使用 ocronline 对象对图像进行识别，得到识别结果
    res = ocronline.predict(img, pre[2])
    # 如果识别结果为 None 或与本地 OCR 得到的结果相同，则记录警告日志
    if res is None or res == pre_res:
        logger.warning(f'{text}识别异常：{pre_res} 为不存在的数据')
    # 如果识别结果不在期望的识别结果列表中，则记录警告日志
    elif res not in valid:
        logger.warning(f'{text}识别异常：{pre_res} 和 {res} 均为不存在的数据')
    # 否则记录警告日志，更新 ocr_error 对象，并返回识别结果
    else:
        logger.warning(f'{text}识别异常：{pre_res} 应为 {res}')
        ocr_error[pre_res] = res
        pre_res = res
    return pre_res
```