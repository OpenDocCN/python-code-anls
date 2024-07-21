# `.\pytorch\torch\utils\tensorboard\_utils.py`

```
# mypy: allow-untyped-defs
import numpy as np

# Functions for converting
# 将 matplotlib 图形转换为 numpy 格式的图像
def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Args:
        figures (matplotlib.pyplot.figure or list of figures): figure or a list of figures
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as plt_backend_agg

    def render_to_rgb(figure):
        # 使用 Agg 后端将图形渲染到画布
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        # 从画布缓冲区获取 RGBA 数据并转换为 numpy 数组
        data: np.ndarray = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        # 将 RGBA 数据转换为 RGB 数据
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        # 调整通道顺序为 [CHW]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_chw

    if isinstance(figures, list):
        # 如果输入是图形列表，则对每个图形进行转换并堆叠成一个 numpy 数组
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        # 如果输入是单个图形，则直接转换为 numpy 数组
        image = render_to_rgb(figures)
        return image


def _prepare_video(V):
    """
    Convert a 5D tensor into 4D tensor.

    Convesrion is done from [batchsize, time(frame), channel(color), height, width]  (5D tensor)
    to [time(frame), new_width, new_height, channel] (4D tensor).

    A batch of images are spreaded to a grid, which forms a frame.
    e.g. Video with batchsize 16 will have a 4x4 grid.
    """
    b, t, c, h, w = V.shape

    if V.dtype == np.uint8:
        # 将 uint8 类型的数据转换为 float32，并归一化到 [0, 1]
        V = np.float32(V) / 255.0

    def is_power2(num):
        # 判断一个数是否为 2 的幂次方
        return num != 0 and ((num & (num - 1)) == 0)

    # 将 batchsize 补齐到最接近的 2 的幂次方
    if not is_power2(V.shape[0]):
        len_addition = int(2 ** V.shape[0].bit_length() - V.shape[0])
        V = np.concatenate((V, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)

    n_rows = 2 ** ((b.bit_length() - 1) // 2)
    n_cols = V.shape[0] // n_rows

    # 重新组织数据，形成时间帧 x 新宽度 x 新高度 x 通道的张量
    V = np.reshape(V, newshape=(n_rows, n_cols, t, c, h, w))
    V = np.transpose(V, axes=(2, 0, 4, 1, 5, 3))
    V = np.reshape(V, newshape=(t, n_rows * h, n_cols * w, c))

    return V


def make_grid(I, ncols=8):
    # I: N1HW or N3HW
    # 断言输入是 numpy 数组，并且通道数为 3
    assert isinstance(I, np.ndarray), "plugin error, should pass numpy array here"
    if I.shape[1] == 1:
        # 如果通道数为 1，则将其复制成 3 通道的图像
        I = np.concatenate([I, I, I], 1)
    assert I.ndim == 4 and I.shape[1] == 3
    nimg = I.shape[0]
    H = I.shape[2]
    W = I.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    # 创建一个画布，用于将多个图像拼接成网格
    canvas = np.zeros((3, H * nrows, W * ncols), dtype=I.dtype)
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            # 将每个图像放置到画布的相应位置
            canvas[:, y * H : (y + 1) * H, x * W : (x + 1) * W] = I[i]
            i = i + 1
    return canvas

    # if modality == 'IMG':
    #     if x.dtype == np.uint8:
    # 将变量 x 转换为 numpy 数组，并将其类型转换为 np.float32，然后将所有元素除以 255.0。
    # 这行代码假设 x 是一个包含像素值的数组（通常是图像数据），并将像素值标准化到 [0, 1] 的范围内。
    x = x.astype(np.float32) / 255.0
# 将输入的张量(tensor)转换为指定的 HWC（Height, Width, Channel）格式
def convert_to_HWC(tensor, input_format):  # tensor: numpy array
    # 确保输入格式(input_format)中没有重复的维度缩写
    assert len(set(input_format)) == len(
        input_format
    ), f"You can not use the same dimension shordhand twice. input_format: {input_format}"
    # 确保输入张量(tensor)的维度与输入格式(input_format)的长度相同
    assert len(tensor.shape) == len(
        input_format
    ), f"size of input tensor and input format are different. tensor shape: {tensor.shape}, input_format: {input_format}"
    # 将输入格式(input_format)转换为大写形式
    input_format = input_format.upper()

    # 如果输入格式(input_format)包含四个维度缩写，则转换为 NCHW 格式
    if len(input_format) == 4:
        # 获取 "NCHW" 中每个维度在输入格式(input_format)中的索引
        index = [input_format.find(c) for c in "NCHW"]
        # 根据索引重新排列张量(tensor)的维度顺序，得到 NCHW 格式的张量
        tensor_NCHW = tensor.transpose(index)
        # 将 NCHW 格式的张量(tensor_NCHW)转换为 HWC 格式的张量(tensor_CHW)
        tensor_CHW = make_grid(tensor_NCHW)
        # 将 HWC 格式的张量(tensor_CHW)再次转置，得到最终的 HWC 格式的张量
        return tensor_CHW.transpose(1, 2, 0)

    # 如果输入格式(input_format)包含三个维度缩写，则转换为 HWC 格式
    if len(input_format) == 3:
        # 获取 "HWC" 中每个维度在输入格式(input_format)中的索引
        index = [input_format.find(c) for c in "HWC"]
        # 根据索引重新排列张量(tensor)的维度顺序，得到 HWC 格式的张量
        tensor_HWC = tensor.transpose(index)
        # 如果 HWC 格式的张量(tensor_HWC)的通道维度为 1，则复制该通道以填充为三通道
        if tensor_HWC.shape[2] == 1:
            tensor_HWC = np.concatenate([tensor_HWC, tensor_HWC, tensor_HWC], 2)
        # 返回最终的 HWC 格式的张量(tensor_HWC)
        return tensor_HWC

    # 如果输入格式(input_format)包含两个维度缩写，则转换为 HW 格式
    if len(input_format) == 2:
        # 获取 "HW" 中每个维度在输入格式(input_format)中的索引
        index = [input_format.find(c) for c in "HW"]
        # 根据索引重新排列张量(tensor)的维度顺序，得到 HW 格式的张量
        tensor = tensor.transpose(index)
        # 复制张量(tensor)的最后一个维度以填充为三通道
        tensor = np.stack([tensor, tensor, tensor], 2)
        # 返回最终的 HW 格式的张量(tensor)
        return tensor
```