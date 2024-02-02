# `arknights-mower\packaging\image.py`

```py
# 版权声明
# 本文件包含图像预处理的一些常见接口。
# 许多用户对图像布局感到困惑。我们介绍图像布局如下。

# - CHW 布局
#   - 缩写：C=通道，H=高度，W=宽度
#   - cv2 或 PIL 打开的图像的默认布局是 HWC。
#     PaddlePaddle 只支持 CHW 布局。而 CHW 只是 HWC 的转置。必须对输入图像进行转置。

# - 色彩格式：RGB 或 BGR
#   OpenCV 使用 BGR 色彩格式。PIL 使用 RGB 色彩格式。两种格式都可以用于训练。请注意，在训练和推断期间，格式应保持一致。
# 尝试导入 cv2 模块，如果导入失败则将 cv2 设为 None
try:
    import cv2
except ImportError:
    cv2 = None
# 导入 os 模块
import os
# 导入 tarfile 模块
import tarfile
# 导入 six.moves.cPickle 模块并重命名为 pickle
import six.moves.cPickle as pickle

# 定义一个空列表
__all__ = []

# 检查是否成功导入了 cv2 模块
def _check_cv2():
    # 如果 cv2 为 None，则输出警告信息
    if cv2 is None:
        import sys
        sys.stderr.write(
            '''Warning with paddle image module: opencv-python should be imported,
         or paddle image module could NOT work; please install opencv-python first.'''
        )
        return False
    else:
        return True

# 从 tar 文件中批量读取图像并将它们分批存储到文件中
def batch_images_from_tar(data_file,
                          dataset_name,
                          img2label,
                          num_per_batch=1024):
    """
    Read images from tar file and batch them into batch file.

    :param data_file: path of image tar file
    :type data_file: string
    :param dataset_name: 'train','test' or 'valid'
    :type dataset_name: string
    :param img2label: a dic with image file name as key
                    and image's label as value
    :type img2label: dic
    :param num_per_batch: image number per batch file
    :type num_per_batch: int
    :return: path of list file containing paths of batch file
    :rtype: string
    """
    # 设置批处理文件夹路径
    batch_dir = data_file + "_batch"
    # 设置输出路径
    out_path = "%s/%s_%s" % (batch_dir, dataset_name, os.getpid())
    # 设置元数据文件路径
    meta_file = "%s/%s_%s.txt" % (batch_dir, dataset_name, os.getpid())

    # 如果输出路径已经存在，则返回元数据文件路径
    if os.path.exists(out_path):
        return meta_file
    else:
        # 否则创建输出路径
        os.makedirs(out_path)

    # 打开 tar 文件
    tf = tarfile.open(data_file)
    # 获取 tar 文件中的所有成员
    mems = tf.getmembers()
    # 定义空列表
    data = []
    labels = []
    # 文件编号初始化为 0
    file_id = 0
    # 遍历mems列表中的每个成员
    for mem in mems:
        # 如果成员的名称在img2label字典中
        if mem.name in img2label:
            # 读取成员的数据并添加到data列表中
            data.append(tf.extractfile(mem).read())
            # 将成员的标签添加到labels列表中
            labels.append(img2label[mem.name])
            # 如果data列表的长度达到了num_per_batch
            if len(data) == num_per_batch:
                # 创建一个空字典output
                output = {}
                # 将labels列表添加到output字典中的'label'键下
                output['label'] = labels
                # 将data列表添加到output字典中的'data'键下
                output['data'] = data
                # 将output字典以二进制形式写入文件
                pickle.dump(output,
                            open('%s/batch_%d' % (out_path, file_id), 'wb'),
                            protocol=2)
                # 增加file_id的值
                file_id += 1
                # 重置data和labels列表
                data = []
                labels = []
    # 如果data列表中还有剩余数据
    if len(data) > 0:
        # 创建一个空字典output
        output = {}
        # 将labels列表添加到output字典中的'label'键下
        output['label'] = labels
        # 将data列表添加到output字典中的'data'键下
        output['data'] = data
        # 将output字典以二进制形式写入文件
        pickle.dump(output,
                    open('%s/batch_%d' % (out_path, file_id), 'wb'),
                    protocol=2)

    # 以追加模式打开meta_file文件
    with open(meta_file, 'a') as meta:
        # 遍历out_path目录下的文件
        for file in os.listdir(out_path):
            # 将文件的绝对路径写入meta_file文件中
            meta.write(os.path.abspath("%s/%s" % (out_path, file)) + "\n")
    # 返回meta_file文件的路径
    return meta_file
# 从字节数组加载彩色或灰度图像
def load_image_bytes(bytes, is_color=True):
    # 检查是否安装了 OpenCV
    assert _check_cv2() is True

    # 如果 is_color 为 True，则设置标志为 1，否则为 0
    flag = 1 if is_color else 0
    # 将字节数组转换为无符号 8 位整数数组
    file_bytes = np.asarray(bytearray(bytes), dtype=np.uint8)
    # 使用 OpenCV 解码字节数组，根据标志返回彩色或灰度图像
    img = cv2.imdecode(file_bytes, flag)
    return img


# 从文件路径加载彩色或灰度图像
def load_image(file, is_color=True):
    # 检查是否安装了 OpenCV
    assert _check_cv2() is True

    # 如果 is_color 为 True，则设置标志为 1，否则为 0
    flag = 1 if is_color else 0
    # 使用 OpenCV 读取文件，根据标志返回彩色或灰度图像
    im = cv2.imread(file, flag)
    return im


# 调整图像大小，使较短边的长度为指定大小
def resize_short(im, size):
    # 检查是否安装了 OpenCV
    assert _check_cv2() is True

    # 获取图像的高度和宽度
    h, w = im.shape[:2]
    # 设置新的高度和宽度为指定大小
    h_new, w_new = size, size
    # 如果图片的高度大于宽度，则按照比例调整高度
    if h > w:
        h_new = size * h // w
    # 否则按照比例调整宽度
    else:
        w_new = size * w // h
    # 使用OpenCV的resize函数按照新的宽度和高度调整图片大小，使用立方插值法
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
    # 返回调整大小后的图片
    return im
# 将输入图像的通道顺序进行转置，从HWC格式转换为CHW格式
def to_chw(im, order=(2, 0, 1)):
    # 检查输入图像的维度和转置顺序的长度是否一致
    assert len(im.shape) == len(order)
    # 根据给定的顺序进行转置
    im = im.transpose(order)
    # 返回转置后的图像
    return im


# 对图像进行中心裁剪，保留图像中心部分
def center_crop(im, size, is_color=True):
    # 获取输入图像的高度和宽度
    h, w = im.shape[:2]
    # 计算裁剪的起始位置
    h_start = (h - size) // 2
    w_start = (w - size) // 2
    # 计算裁剪的结束位置
    h_end, w_end = h_start + size, w_start + size
    # 根据是否为彩色图像进行裁剪
    if is_color:
        im = im[h_start:h_end, w_start:w_end, :]
    else:
        im = im[h_start:h_end, w_start:w_end]
    # 返回裁剪后的图像
    return im


# 对图像进行随机裁剪，随机选择图像的一部分进行裁剪
def random_crop(im, size, is_color=True):
    # 获取输入图像的高度和宽度
    h, w = im.shape[:2]
    # 随机选择裁剪的起始位置
    h_start = np.random.randint(0, h - size + 1)
    w_start = np.random.randint(0, w - size + 1)
    # 计算裁剪的结束位置
    h_end, w_end = h_start + size, w_start + size
    # 根据是否为彩色图像进行裁剪
    if is_color:
        im = im[h_start:h_end, w_start:w_end, :]
    else:
        im = im[h_start:h_end, w_start:w_end]
    # 返回裁剪后的图像
    return im


# 对图像进行左右翻转
def left_right_flip(im, is_color=True):
    # 略
    # 水平翻转图像
    # 返回翻转后的图像
    
    # 示例用法:
    # im = left_right_flip(im)
    
    # 参数 im: 输入图像，HWC布局或灰度图像的HW布局
    # 参数类型 im: ndarray
    # 参数 is_color: 输入图像是否为彩色图像
    # 参数类型 is_color: bool
    """
    如果图像的维度为3且是彩色图像，则返回水平翻转后的图像
    否则，返回水平翻转后的图像
    """
    if len(im.shape) == 3 and is_color:
        return im[:, ::-1, :]
    else:
        return im[:, ::-1]
# 对训练数据进行简单的数据增强，包括调整大小、裁剪和翻转

# 从输入图像中调整大小，使其短边长度为指定值
im = resize_short(im, resize_size)

# 如果是训练数据
if is_train:
    # 从图像中随机裁剪出指定大小的区域
    im = random_crop(im, crop_size, is_color=is_color)
    # 以50%的概率对图像进行左右翻转
    if np.random.randint(2) == 0:
        im = left_right_flip(im, is_color)

# 如果不是训练数据
else:
    # 从图像中心裁剪出指定大小的区域
    im = center_crop(im, crop_size, is_color=is_color)

# 如果图像是三通道的，将其转换为通道-宽-高的格式
if len(im.shape) == 3:
    im = to_chw(im)

# 将图像数据类型转换为 float32
im = im.astype('float32')

# 如果给定了均值
if mean is not None:
    # 将均值转换为 float32 类型的数组
    mean = np.array(mean, dtype=np.float32)
    # 如果均值是一维数组且图像是彩色的
    if mean.ndim == 1 and is_color:
        mean = mean[:, np.newaxis, np.newaxis]
    # 如果均值是一维数组
    elif mean.ndim == 1:
        mean = mean
    # 否则，均值是元素级的均值
    else:
        assert len(mean.shape) == len(im)
    # 对图像数据减去均值
    im -= mean

# 返回处理后的图像
return im
    # 数据增强函数，参考 simple_transform 接口进行转换操作
    # 示例用法
    # 加载并转换图像
    # im = load_and_transform('cat.jpg', 256, 224, True)
    
    # 输入图像文件名
    # filename: 输入图像文件名
    # 类型: 字符串
    # resize_size: 调整后图像的较短边长度
    # 类型: 整数
    # crop_size: 裁剪尺寸
    # 类型: 整数
    # is_train: 是否为训练
    # 类型: 布尔值
    # is_color: 图像是否为彩色
    # 类型: 布尔值
    # mean: 均值，可以是逐元素均值或通道均值
    # 类型: numpy 数组 | 列表
    def load_and_transform(filename, resize_size, crop_size, is_train, is_color, mean):
        # 加载图像
        im = load_image(filename, is_color)
        # 对图像进行简单转换
        im = simple_transform(im, resize_size, crop_size, is_train, is_color, mean)
        # 返回转换后的图像
        return im
```