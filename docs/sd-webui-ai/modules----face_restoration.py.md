# `stable-diffusion-webui\modules\face_restoration.py`

```py
# 从 modules 模块中导入 shared 对象
from modules import shared

# 定义人脸修复类
class FaceRestoration:
    # 返回类的名称
    def name(self):
        return "None"

    # 对输入的 numpy 图像进行修复，返回修复后的 numpy 图像
    def restore(self, np_image):
        return np_image

# 修复人脸的函数，接收一个 numpy 图像作为输入
def restore_faces(np_image):
    # 从 shared 模块中获取人脸修复器列表，筛选出名称匹配的修复器
    face_restorers = [x for x in shared.face_restorers if x.name() == shared.opts.face_restoration_model or shared.opts.face_restoration_model is None]
    # 如果没有匹配的修复器，则直接返回原始图像
    if len(face_restorers) == 0:
        return np_image

    # 选择第一个匹配的修复器
    face_restorer = face_restorers[0]

    # 调用选定的修复器对输入图像进行修复，并返回修复后的图像
    return face_restorer.restore(np_image)
```