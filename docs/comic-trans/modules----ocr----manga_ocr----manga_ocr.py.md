# `.\comic-translate\modules\ocr\manga_ocr\manga_ocr.py`

```py
# modified from https://github.com/kha-white/manga-ocr/blob/master/manga_ocr/ocr.py

# 导入必要的库
import re                  # 导入正则表达式模块
import jaconv              # 导入日语字符转换模块
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel  # 导入transformers库中的类
import numpy as np         # 导入NumPy库
import torch               # 导入PyTorch库

# 定义MANGA_OCR_PATH常量，指向预训练模型的路径
MANGA_OCR_PATH = r'data/models/manga-ocr-base'

# 定义MangaOcr类
class MangaOcr:
    def __init__(self, pretrained_model_name_or_path=MANGA_OCR_PATH, device='cpu'):
        # 初始化特征提取器、分词器和视觉编码解码模型
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)
        self.to(device)  # 将模型移动到指定的设备上
        
    def to(self, device):
        self.model.to(device)  # 将模型移动到指定的设备上

    @torch.no_grad()
    def __call__(self, img: np.ndarray):
        # 使用特征提取器将图像转换为PyTorch张量，然后压缩为一维张量
        x = self.feature_extractor(img, return_tensors="pt").pixel_values.squeeze()
        # 使用模型生成解码后的输出序列，并移动到CPU上
        x = self.model.generate(x[None].to(self.model.device))[0].cpu()
        # 使用分词器解码序列，去除特殊标记
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        # 对解码后的文本进行后处理
        x = post_process(x)
        return x

    # 未实现的方法，用于批量OCR处理
    def ocr_batch(self, im_batch: torch.Tensor):
        raise NotImplementedError

# 定义文本后处理函数
def post_process(text):
    text = ''.join(text.split())  # 去除空格
    text = text.replace('…', '...')  # 替换省略号
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)  # 使用点号替换多个连续的点或中文句号
    text = jaconv.h2z(text, ascii=True, digit=True)  # 将半角转换为全角
    return text

# 主程序入口
if __name__ == '__main__':
    import cv2  # 导入OpenCV库

    img_path = r'data/testpacks/textline/ballontranslator.png'  # 图像文件路径
    manga_ocr = MangaOcr(pretrained_model_name_or_path=MANGA_OCR_PATH, device='cuda')  # 创建MangaOcr实例，指定使用CUDA加速

    img = cv2.imread(img_path)  # 读取图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从BGR格式转换为RGB格式

    dummy = np.zeros((1024, 1024, 3), np.uint8)  # 创建虚拟图像
    manga_ocr(dummy)  # 对虚拟图像进行OCR处理
    # preprocessed = manga_ocr(img_path)  # 对真实图像进行OCR处理，未实现

    # im_batch =
    # img = (torch.from_numpy(img[np.newaxis, ...]).float() - 127.5) / 127.5
    # img = einops.rearrange(img, 'N H W C -> N C H W')
    import time  # 导入时间模块

    # 多次运行OCR并计时
    for ii in range(10):
        t0 = time.time()  # 记录起始时间
        out = manga_ocr(dummy)  # 对虚拟图像进行OCR处理
        print(out, time.time() - t0)  # 输出OCR结果和处理时间
```