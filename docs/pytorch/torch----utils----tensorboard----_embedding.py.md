# `.\pytorch\torch\utils\tensorboard\_embedding.py`

```py
# mypy: allow-untyped-defs
# 导入数学库和第三方库
import math
import numpy as np
# 导入本地模块
from ._convert_np import make_np
from ._utils import make_grid
# 导入 TensorFlow 库和相关模块
from tensorboard.compat import tf
from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo

# 检查 tf.io.gfile 模块是否具有 join 方法
_HAS_GFILE_JOIN = hasattr(tf.io.gfile, "join")


def _gfile_join(a, b):
    # 根据不同情况使用 tf.io.gfile 的 join 方法
    # 参考：https://github.com/tensorflow/tensorboard/issues/6080
    if _HAS_GFILE_JOIN:
        return tf.io.gfile.join(a, b)
    else:
        # 如果没有 join 方法，则获取文件系统并使用文件系统的 join 方法
        fs = tf.io.gfile.get_filesystem(a)
        return fs.join(a, b)


def make_tsv(metadata, save_path, metadata_header=None):
    # 根据是否有 metadata_header 决定如何处理 metadata
    if not metadata_header:
        # 将 metadata 转换为字符串列表
        metadata = [str(x) for x in metadata]
    else:
        # 确保 metadata_header 的长度与 metadata 的列数相同
        assert len(metadata_header) == len(
            metadata[0]
        ), "len of header must be equal to the number of columns in metadata"
        # 将 metadata 转换为带有制表符的字符串列表，用于写入文件
        metadata = ["\t".join(str(e) for e in l) for l in [metadata_header] + metadata]

    # 将 metadata 转换为字节流，并写入到指定路径的文件中
    metadata_bytes = tf.compat.as_bytes("\n".join(metadata) + "\n")
    with tf.io.gfile.GFile(_gfile_join(save_path, "metadata.tsv"), "wb") as f:
        f.write(metadata_bytes)


# https://github.com/tensorflow/tensorboard/issues/44 image label will be squared
def make_sprite(label_img, save_path):
    # 导入必要的图像处理模块
    from PIL import Image
    from io import BytesIO

    # 确保生成的 sprite 图像具有正确的维度
    nrow = int(math.ceil((label_img.size(0)) ** 0.5))
    arranged_img_CHW = make_grid(make_np(label_img), ncols=nrow)

    # 扩展图像以确保图像数量达到 nrow*nrow
    arranged_augment_square_HWC = np.zeros(
        (arranged_img_CHW.shape[2], arranged_img_CHW.shape[2], 3)
    )
    arranged_img_HWC = arranged_img_CHW.transpose(1, 2, 0)  # chw -> hwc
    arranged_augment_square_HWC[: arranged_img_HWC.shape[0], :, :] = arranged_img_HWC
    im = Image.fromarray(np.uint8((arranged_augment_square_HWC * 255).clip(0, 255)))

    # 将生成的 sprite 图像保存为 PNG 格式，并转换为字节流
    with BytesIO() as buf:
        im.save(buf, format="PNG")
        im_bytes = buf.getvalue()

    # 将生成的 sprite 图像字节流写入到指定路径的文件中
    with tf.io.gfile.GFile(_gfile_join(save_path, "sprite.png"), "wb") as f:
        f.write(im_bytes)


def get_embedding_info(metadata, label_img, subdir, global_step, tag):
    # 创建 EmbeddingInfo 对象，并设置相关属性
    info = EmbeddingInfo()
    info.tensor_name = f"{tag}:{str(global_step).zfill(5)}"
    info.tensor_path = _gfile_join(subdir, "tensors.tsv")
    if metadata is not None:
        info.metadata_path = _gfile_join(subdir, "metadata.tsv")
    if label_img is not None:
        # 设置 sprite 图像的路径及单个图像的尺寸
        info.sprite.image_path = _gfile_join(subdir, "sprite.png")
        info.sprite.single_image_dim.extend([label_img.size(3), label_img.size(2)])
    return info


def write_pbtxt(save_path, contents):
    # 将内容写入到指定路径的 .pbtxt 文件中
    config_path = _gfile_join(save_path, "projector_config.pbtxt")
    with tf.io.gfile.GFile(config_path, "wb") as f:
        f.write(tf.compat.as_bytes(contents))
# 定义一个函数 make_mat，接收 matlist 和 save_path 两个参数，用于生成一个 TSV 格式的文件
def make_mat(matlist, save_path):
    # 使用 tf.io.gfile.GFile 打开一个二进制写入文件流，文件名为 save_path 下的 "tensors.tsv"
    with tf.io.gfile.GFile(_gfile_join(save_path, "tensors.tsv"), "wb") as f:
        # 遍历 matlist 中的每一个元素 x
        for x in matlist:
            # 将 x 中的每个元素 i 转换为字符串并存入列表，转换为一行的 TSV 格式
            x = [str(i.item()) for i in x]
            # 将拼接好的 TSV 行数据转换为字节并写入文件流 f
            f.write(tf.compat.as_bytes("\t".join(x) + "\n"))
```