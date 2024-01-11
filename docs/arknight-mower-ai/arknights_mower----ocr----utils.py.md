# `arknights-mower\arknights_mower\ocr\utils.py`

```
# 导入正则表达式模块
import re

# 导入 numpy 模块，并重命名为 np
import numpy as np
# 从 PIL 库中导入 Image 模块
from PIL import Image
# 从上级目录中的 data 模块中导入 ocr_error
from ..data import ocr_error
# 从上级目录中的 utils.log 模块中导入 logger
from ..utils.log import logger

# 定义 resizeNormalize 类
class resizeNormalize(object):

    # 初始化方法，接受尺寸和插值方式作为参数
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    # 调用对象时执行的方法，接受图像作为参数
    def __call__(self, img):
        size = self.size
        imgW, imgH = size
        scale = img.size[1] * 1.0 / imgH
        w = img.size[0] / scale
        w = int(w)
        img = img.resize((w, imgH), self.interpolation)
        w, h = img.size
        if w <= imgW:
            newImage = np.zeros((imgH, imgW), dtype='uint8')
            newImage[:] = 255
            newImage[:, :w] = np.array(img)
            img = Image.fromarray(newImage)
        else:
            img = img.resize((imgW, imgH), self.interpolation)
        img = np.array(img, dtype=np.float32)
        img -= 127.5
        img /= 127.5
        img = img.reshape([*img.shape, 1])
        return img

# 定义 strLabelConverter 类
class strLabelConverter(object):

    # 初始化方法，接受字母表作为参数
    def __init__(self, alphabet):
        self.alphabet = alphabet + 'ç'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    # 解码方法，接受 t、length 和 raw 作为参数
    def decode(self, t, length, raw=False):
        t = t[:length]
        if raw:
            return ''.join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return ''.join(char_list)

# 定义 fix 函数，接受字符串 s 作为参数
def fix(s):
    """
    对识别结果进行简单处理，并查询是否在 ocr_error 中有过记录
    """
    # 使用正则表达式替换特定字符
    s = re.sub(r'[。？！，、；：“”‘’（）《》〈〉【】『』「」﹃﹄〔〕…～﹏￥－＿]', '', s)
    s = re.sub(r'[\'\"\,\.\(\)]', '', s)
    # 如果 s 在 ocr_error 中，则替换为对应的值
    if s in ocr_error.keys():
        logger.debug(f'fix with ocr_error: {s} -> {ocr_error[s]}')
        s = ocr_error[s]
    return s
```