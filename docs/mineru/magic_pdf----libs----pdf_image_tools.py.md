# `.\MinerU\magic_pdf\libs\pdf_image_tools.py`

```
# 从 magic_pdf 库中导入必要的模块和类
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.libs.commons import fitz
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.hash_utils import compute_sha256


# 定义函数 cut_image，接受裁剪框、页码、页面对象、返回路径和图像写入器
def cut_image(bbox: tuple, page_num: int, page: fitz.Page, return_path, imageWriter: AbsReaderWriter):
    """
    从第page_num页的page中，根据bbox进行裁剪出一张jpg图片，返回图片路径
    save_path：需要同时支持s3和本地, 图片存放在save_path下，文件名是: {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , bbox内数字取整。
    """
    # 拼接文件名，格式为：页码_左_上_右_下
    filename = f"{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"

    # 如果提供了返回路径，则拼接完整路径；否则设为 None
    img_path = join_path(return_path, filename) if return_path is not None else None

    # 计算文件的 SHA-256 哈希值，并生成新的图片文件名
    img_hash256_path = f"{compute_sha256(img_path)}.jpg"

    # 将裁剪框的坐标转换为 fitz.Rect 对象
    rect = fitz.Rect(*bbox)
    # 设置缩放矩阵为 3 倍
    zoom = fitz.Matrix(3, 3)
    # 从页面中根据裁剪框获取图像
    pix = page.get_pixmap(clip=rect, matrix=zoom)

    # 将获取的图像转换为 JPEG 格式的字节数据，质量设为 95
    byte_data = pix.tobytes(output='jpeg', jpg_quality=95)

    # 使用图像写入器将字节数据写入指定路径
    imageWriter.write(byte_data, img_hash256_path, AbsReaderWriter.MODE_BIN)

    # 返回生成的图像路径
    return img_hash256_path
```