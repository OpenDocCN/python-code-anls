# `.\transformers\models\owlvit\image_processing_owlvit.py`

```
# 编码设置为 UTF-8
# 定义 OwlViT 图像处理器类，继承自 BaseImageProcessor
class OwlViTImageProcessor(BaseImageProcessor):
    # 构造函数
    r"""
    Constructs an OWL-ViT image processor.

    This image processor inherits from [`ImageProcessingMixin`] which contains most of the main methods. Users should
    """
    
    # 定义一些函数
    def _upcast(t):
        """
        Protects from numerical overflows in multiplications by upcasting to the equivalent higher type.
        
        Args:
            t (`torch.Tensor`):
                The input tensor.
        
        Returns:
            `torch.Tensor`: The upcast tensor.
        """
        # 如果输入为浮点型，则返回 float32 或 float64 类型的张量
        if t.is_floating_point():
            return t if t.dtype in (torch.float32, torch.float64) else t.float()
        # 否则返回 int32 或 int64 类型的张量    
        else:
            return t if t.dtype in (torch.int32, torch.int64) else t.int()
    
    def box_area(boxes):
        """
        Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.
        
        Args:
            boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
                Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
                < x2` and `0 <= y1 < y2`.
        Returns:
            `torch.FloatTensor`: a tensor containing the area for each box.
        """
        # 将输入张量转换为可以支持数值计算的更高类型
        boxes = _upcast(boxes)
        # 计算每个边界框的面积并返回
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    def box_iou(boxes1, boxes2):
        """
        Computes the Intersection over Union (IoU) of two set of boxes.
        
        Args:
            boxes1 (`torch.FloatTensor` of shape `(number_of_boxes1, 4)`):
                boxes in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
            boxes2 (`torch.FloatTensor` of shape `(number_of_boxes2, 4)`):
                boxes in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        Returns:
            `torch.FloatTensor`: the Intersection over Union (IoU) of boxes1 and boxes2. Shape will be `(number_of_boxes1, number_of_boxes2)`.
        """
        # 计算两组边界框的面积
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        # 计算两组边界框的交集区域
        left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
        inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

        # 计算两组边界框的并集面积
        union = area1[:, None] + area2 - inter

        # 计算交并比
        iou = inter / union

        return iou, union
    """
    此类用于处理图像的预处理，包括调整大小、裁剪、缩放、归一化等操作。查阅超类以获取有关这些方法的更多信息。

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整输入图像的较短边到特定的 `size`。
        size (`Dict[str, int]`, *optional*, defaults to {"height": 768, "width": 768}):
            用于调整图像大小的尺寸。仅当 `do_resize` 设置为 `True` 时才有效。如果 `size` 是类似 (h, w) 的序列，输出大小将与之匹配。如果 `size` 是整数，则图像将调整为 (size, size)。
        resample (`int`, *optional*, defaults to `Resampling.BICUBIC`):
            可选的重采样滤波器。可以是 `PIL.Image.Resampling.NEAREST`、`PIL.Image.Resampling.BOX`、`PIL.Image.Resampling.BILINEAR`、`PIL.Image.Resampling.HAMMING`、`PIL.Image.Resampling.BICUBIC` 或 `PIL.Image.Resampling.LANCZOS` 中的一个。仅当 `do_resize` 设置为 `True` 时才有效。
        do_center_crop (`bool`, *optional*, defaults to `False`):
            是否在中心位置裁剪输入图像。如果输入大小沿任何边缘小于 `crop_size`，则图像将使用 0 进行填充，然后在中心位置裁剪。
        crop_size (`int`, *optional*, defaults to {"height": 768, "width": 768}):
            用于中心裁剪图像的尺寸。仅当 `do_center_crop` 设置为 `True` 时才有效。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按一定因子对输入进行缩放。
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            用于缩放图像的因子。仅当 `do_rescale` 设置为 `True` 时才有效。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对输入进行归一化，使用 `image_mean` 和 `image_std`。仅当 `do_center_crop` 设置为 `True` 时才有效。
        image_mean (`List[int]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            每个通道的均值序列，用于归一化图像时使用。
        image_std (`List[int]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            每个通道的标准差序列，用于归一化图像时使用。
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=None,
        resample=PILImageResampling.BICUBIC,
        do_center_crop=False,
        crop_size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs,
    ):
    # 这是一个 ImageProcessor 类的初始化方法
    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Union[int, Tuple[int, int], Dict[str, int]]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Optional[Union[int, Tuple[int, int], Dict[str, int]]] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[Sequence[float], float]] = None,
        image_std: Optional[Union[Sequence[float], float]] = None,
        **kwargs,
    ):
        # 如果没有指定尺寸大小，则设置默认值为 {"height": 768, "width": 768}
        size = size if size is not None else {"height": 768, "width": 768}
        # 确保 size 参数是一个字典，且包含 height 和 width 键
        size = get_size_dict(size, default_to_square=True)
    
        # 如果没有指定裁剪大小，则设置默认值为 {"height": 768, "width": 768}
        crop_size = crop_size if crop_size is not None else {"height": 768, "width": 768}
        # 确保 crop_size 参数是一个字典，且包含 height 和 width 键
        crop_size = get_size_dict(crop_size, default_to_square=True)
    
        # 处理 "rescale" 参数的向后兼容性
        if "rescale" in kwargs:
            rescale_val = kwargs.pop("rescale")
            kwargs["do_rescale"] = rescale_val
    
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置类属性
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
    
    # 这是一个用于调整图像大小的方法
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to a certain size.
    
        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                The size to resize the image to. Must contain height and width keys.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                The resampling filter to use when resizing the input.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 确保 size 参数是一个字典，且包含 height 和 width 键
        size = get_size_dict(size, default_to_square=True)
        if "height" not in size or "width" not in size:
            raise ValueError("size dictionary must contain height and width keys")
    
        # 调用 resize 函数调整图像大小
        return resize(
            image,
            (size["height"], size["width"]),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    # 定义一个方法，用于将图片居中裁剪到特定尺寸
    def center_crop(
        self,
        image: np.ndarray,
        crop_size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Center crop an image to a certain size.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            crop_size (`Dict[str, int]`):
                The size to center crop the image to. Must contain height and width keys.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取裁剪尺寸字典，并将其调整为正方形
        crop_size = get_size_dict(crop_size, default_to_square=True)
        if "height" not in crop_size or "width" not in crop_size:
            raise ValueError("crop_size dictionary must contain height and width keys")

        # 返回居中裁剪后的图片
        return center_crop(
            image,
            (crop_size["height"], crop_size["width"]),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # 从transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale中复制而来
    # 定义一个方法，用于将图片按指定因子进行重新缩放
    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Rescale the image by the given factor. image = image * rescale_factor.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            rescale_factor (`float`):
                The value to use for rescaling.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. If unset, is inferred from the input image. Can be
                one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        # 调用rescale函数对图像进行缩放处理
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
        ):
    # 将[`OwlViTForObjectDetection`]的原始输出转换为最终的边界框格式，即(top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    def post_process(self, outputs, target_sizes):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation). For visualization, this should be the image size after data
                augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # 发出警告，提示用户`post_process`即将被废弃，在Transformers版本5中将被移除，建议使用`post_process_object_detection`替代，使用`threshold=0.`来获得等效的结果
        warnings.warn(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
            FutureWarning,
        )

        # 获得logits和boxes变量
        logits, boxes = outputs.logits, outputs.pred_boxes

        # 检查logits和target_sizes的维度是否匹配，不匹配则抛出数值错误
        if len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        # 检查target_sizes的shape是否为(batch_size, 2)，不匹配则抛出数值错误
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        # 对logits进行softmax操作，用于计算scores
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # 将边框格式转换为[x0, y0, x1, y1]格式
        boxes = center_to_corners_format(boxes)

        # 将相对坐标[0, 1]转换为绝对坐标[0, height]
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        # 创建字典列表，每个字典包含模型预测的每个图像的scores，labels和boxes
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results

    # 用于对象检测的后处理函数，允许设置阈值和目标尺寸
    def post_process_object_detection(
        self, outputs, threshold: float = 0.1, target_sizes: Union[TensorType, List[Tuple]] = None
    # 该函数将 OwlViTForObjectDetection 模型的原始输出转换为最终的边界框，格式为(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    def postprocess_object_detection(
        outputs,
        threshold=None,
        target_sizes=None,
    ):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.
    
        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # 从输出中获取预测的分类logits和边界框
        logits, boxes = outputs.logits, outputs.pred_boxes
    
        # 如果提供了目标尺寸，确保其个数与输出batch维度一致
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
    
        # 从logits中获取最大值，即分类概率
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices
    
        # 将边界框从中心点格式转换为左上右下格式
        boxes = center_to_corners_format(boxes)
    
        # 如果提供了目标尺寸，将相对坐标转换为绝对坐标
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)
    
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]
    
        # 根据阈值筛选结果
        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})
    
        return results
    
    # TODO: (Amy) Make compatible with other frameworks
```