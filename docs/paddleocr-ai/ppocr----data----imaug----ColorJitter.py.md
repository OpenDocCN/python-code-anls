# `.\PaddleOCR\ppocr\data\imaug\ColorJitter.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制
from paddle.vision.transforms import ColorJitter as pp_ColorJitter

# 导出 ColorJitter 类
__all__  = ['ColorJitter']

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0,**kwargs):
        # 创建一个 pp_ColorJitter 实例，传入亮度、对比度、饱和度和色调参数
        self.aug = pp_ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, data):
        # 获取数据中的图像
        image = data['image']
        # 对图像进行颜色增强
        image = self.aug(image)
        # 更新数据中的图像
        data['image'] = image
        # 返回更新后的数据
        return data
```