# `.\yolov8\ultralytics\models\sam\predict.py`

```py
# 导入必要的库和模块
import numpy as np
import torch
import torch.nn.functional as F

# 导入 Ultralytics 自定义的数据增强模块 LetterBox
from ultralytics.data.augment import LetterBox
# 导入 Ultralytics 自定义的预测器基类 BasePredictor
from ultralytics.engine.predictor import BasePredictor
# 导入 Ultralytics 自定义的结果处理模块 Results
from ultralytics.engine.results import Results
# 导入 Ultralytics 的一些实用函数和操作
from ultralytics.utils import DEFAULT_CFG, ops
# 导入选择设备的函数
from ultralytics.utils.torch_utils import select_device

# 导入局部模块中的函数和类
from .amg import (
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    remove_small_regions,
    uncrop_boxes_xyxy,
    uncrop_masks,
)
# 导入局部模块中的构建 SAM 模型的函数
from .build import build_sam

# 定义 SAM 模型的预测器类，继承自 BasePredictor 类
class Predictor(BasePredictor):
    """
    SAM 模型的预测器类，继承自 BasePredictor 类。

    该类提供了用于图像分割任务的模型推断接口。
    具有高级架构和可提示分割功能，支持灵活和实时的掩模生成。
    该类能够处理多种类型的提示，如边界框、点和低分辨率掩模。

    Attributes:
        cfg (dict): 模型和任务相关参数的配置字典。
        overrides (dict): 包含覆盖默认配置的值的字典。
        _callbacks (dict): 用户定义的回调函数字典，用于增强行为。
        args (namespace): 保存命令行参数或其他操作变量的命名空间。
        im (torch.Tensor): 预处理后的输入图像张量。
        features (torch.Tensor): 用于推断的提取图像特征。
        prompts (dict): 包含各种提示类型的集合，如边界框和点。
        segment_all (bool): 控制是否对图像中的所有对象进行分割或仅对指定对象进行分割的标志。
    """
    # 初始化 Predictor 对象，使用默认配置 cfg，如果提供了 overrides，则将其合并到配置中
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the Predictor with configuration, overrides, and callbacks.

        The method sets up the Predictor object and applies any configuration overrides or callbacks provided. It
        initializes task-specific settings for SAM, such as retina_masks being set to True for optimal results.

        Args:
            cfg (dict): Configuration dictionary.
            overrides (dict, optional): Dictionary of values to override default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to customize behavior.
        """
        # 如果 overrides 为 None，则初始化为空字典
        if overrides is None:
            overrides = {}
        # 更新 overrides 字典，设置任务为 "segment"，模式为 "predict"，图像大小为 1024
        overrides.update(dict(task="segment", mode="predict", imgsz=1024))
        # 调用父类的初始化方法，传入 cfg、overrides 和 _callbacks
        super().__init__(cfg, overrides, _callbacks)
        # 设置 self.args.retina_masks 为 True，针对 SAM 模型的特定设置
        self.args.retina_masks = True
        # 初始化 self.im 为 None，用于存储输入图像
        self.im = None
        # 初始化 self.features 为 None，用于存储特征
        self.features = None
        # 初始化 self.prompts 为空字典，用于存储提示信息
        self.prompts = {}
        # 初始化 self.segment_all 为 False，用于控制是否对所有数据进行分割
        self.segment_all = False

    # 对输入图像进行预处理，以供模型推断使用
    def preprocess(self, im):
        """
        Preprocess the input image for model inference.

        The method prepares the input image by applying transformations and normalization.
        It supports both torch.Tensor and list of np.ndarray as input formats.

        Args:
            im (torch.Tensor | List[np.ndarray]): BCHW tensor format or list of HWC numpy arrays.

        Returns:
            (torch.Tensor): The preprocessed image tensor.
        """
        # 如果 self.im 不为 None，直接返回已经处理好的图像
        if self.im is not None:
            return self.im
        # 判断 im 是否为 torch.Tensor 类型之外的类型
        not_tensor = not isinstance(im, torch.Tensor)
        # 如果 im 不是 torch.Tensor 类型，则进行以下转换和预处理步骤
        if not_tensor:
            # 将输入的 HWC 格式的 numpy 数组堆叠成 BCHW 格式的 numpy 数组
            im = np.stack(self.pre_transform(im))
            # 对图像进行颜色通道反转（RGB to BGR）和维度转置，以符合模型输入要求
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            # 转换为连续存储的数组
            im = np.ascontiguousarray(im)
            # 将 numpy 数组转换为 torch.Tensor
            im = torch.from_numpy(im)

        # 将处理好的图像数据移到设备（GPU 或 CPU）上
        im = im.to(self.device)
        # 如果模型使用 FP16 运算，则将图像数据类型转换为半精度（half），否则转换为单精度（float）
        im = im.half() if self.model.fp16 else im.float()
        # 如果输入图像不是 torch.Tensor 类型，则进行均值和标准差归一化处理
        if not_tensor:
            im = (im - self.mean) / self.std
        # 返回预处理后的图像 tensor
        return im

    # 对输入图像执行初始转换，以进行进一步的预处理
    def pre_transform(self, im):
        """
        Perform initial transformations on the input image for preprocessing.

        The method applies transformations such as resizing to prepare the image for further preprocessing.
        Currently, batched inference is not supported; hence the list length should be 1.

        Args:
            im (List[np.ndarray]): List containing images in HWC numpy array format.

        Returns:
            (List[np.ndarray]): List of transformed images.
        """
        # 断言输入的图像列表长度为 1，因为 SAM 模型不支持批量推断
        assert len(im) == 1, "SAM model does not currently support batched inference"
        # 创建 LetterBox 转换器，用于将输入图像调整为模型需要的大小
        letterbox = LetterBox(self.args.imgsz, auto=False, center=False)
        # 对输入图像列表中的每张图像应用 LetterBox 转换
        return [letterbox(image=x) for x in im]
    def inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs):
        """
        Perform image segmentation inference based on the given input cues, using the currently loaded image. This
        method leverages SAM's (Segment Anything Model) architecture consisting of image encoder, prompt encoder, and
        mask decoder for real-time and promptable segmentation tasks.

        Args:
            im (torch.Tensor): The preprocessed input image in tensor format, with shape (N, C, H, W).
            bboxes (np.ndarray | List, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | List, optional): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | List, optional): Labels for point prompts, shape (N, ). 1 = foreground, 0 = background.
            masks (np.ndarray, optional): Low-resolution masks from previous predictions shape (N,H,W). For SAM H=W=256.
            multimask_output (bool, optional): Flag to return multiple masks. Helpful for ambiguous prompts.

        Returns:
            (tuple): Contains the following three elements.
                - np.ndarray: The output masks in shape CxHxW, where C is the number of generated masks.
                - np.ndarray: An array of length C containing quality scores predicted by the model for each mask.
                - np.ndarray: Low-resolution logits of shape CxHxW for subsequent inference, where H=W=256.
        """
        # Override prompts if any stored in self.prompts
        # 从 self.prompts 中取出存储的提示信息（如果有），覆盖函数参数中的对应项
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)

        # 如果所有的提示信息都是 None，则调用 generate 方法生成输出
        if all(i is None for i in [bboxes, points, masks]):
            return self.generate(im, *args, **kwargs)

        # 否则，调用 prompt_inference 方法进行推断
        return self.prompt_inference(im, bboxes, points, labels, masks, multimask_output)

    def generate(
        self,
        im,
        crop_n_layers=0,
        crop_overlap_ratio=512 / 1500,
        crop_downscale_factor=1,
        point_grids=None,
        points_stride=32,
        points_batch_size=64,
        conf_thres=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=0.95,
        crop_nms_thresh=0.7,
    ):
        """
        Generate segmentation masks based on the input image and various parameters.

        Args:
            im (torch.Tensor): The preprocessed input image in tensor format, with shape (N, C, H, W).
            crop_n_layers (int, optional): Number of layers to crop.
            crop_overlap_ratio (float, optional): Ratio of overlap in cropping.
            crop_downscale_factor (int, optional): Factor by which to downscale crops.
            point_grids (np.ndarray, optional): Grids of points for segmentation.
            points_stride (int, optional): Stride for points.
            points_batch_size (int, optional): Batch size for points processing.
            conf_thres (float, optional): Confidence threshold.
            stability_score_thresh (float, optional): Stability score threshold.
            stability_score_offset (float, optional): Stability score offset.
            crop_nms_thresh (float, optional): NMS threshold for cropping.

        Returns:
            None
        """
        # 实现生成分割 mask 的具体逻辑，根据参数设置生成相应的输出
        pass
    def setup_model(self, model, verbose=True):
        """
        Initializes the Segment Anything Model (SAM) for inference.

        This method sets up the SAM model by allocating it to the appropriate device and initializing the necessary
        parameters for image normalization and other Ultralytics compatibility settings.

        Args:
            model (torch.nn.Module): A pre-trained SAM model. If None, a model will be built based on configuration.
            verbose (bool): If True, prints selected device information.

        Attributes:
            model (torch.nn.Module): The SAM model allocated to the chosen device for inference.
            device (torch.device): The device to which the model and tensors are allocated.
            mean (torch.Tensor): The mean values for image normalization.
            std (torch.Tensor): The standard deviation values for image normalization.
        """
        # 选择设备并打印设备信息（如果 verbose=True）
        device = select_device(self.args.device, verbose=verbose)
        
        # 如果未提供预训练的模型，则根据配置构建 SAM 模型
        if model is None:
            model = build_sam(self.args.model)
        
        # 将模型设置为评估模式（不进行梯度更新）
        model.eval()
        
        # 将模型移动到指定的设备上
        self.model = model.to(device)
        self.device = device
        
        # 设置图像归一化所需的均值和标准差，并移动到指定的设备上
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)

        # 设置 Ultralytics 兼容性选项
        self.model.pt = False
        self.model.triton = False
        self.model.stride = 32
        self.model.fp16 = False
        
        # 标记初始化已完成
        self.done_warmup = True
    # 对 SAM 模型推断输出进行后处理，生成目标检测的掩码和边界框
    
    def postprocess(self, preds, img, orig_imgs):
        """
        Post-processes SAM's inference outputs to generate object detection masks and bounding boxes.
    
        The method scales masks and boxes to the original image size and applies a threshold to the mask predictions.
        The SAM model uses advanced architecture and promptable segmentation tasks to achieve real-time performance.
    
        Args:
            preds (tuple): The output from SAM model inference, containing masks, scores, and optional bounding boxes.
            img (torch.Tensor): The processed input image tensor.
            orig_imgs (list | torch.Tensor): The original, unprocessed images.
    
        Returns:
            (list): List of Results objects containing detection masks, bounding boxes, and other metadata.
        """
    
        # 获取预测的掩码和得分
        pred_masks, pred_scores = preds[:2]
        # 如果需要分割所有类别，则获取预测的边界框
        pred_bboxes = preds[2] if self.segment_all else None
        # 生成掩码名称字典
        names = dict(enumerate(str(i) for i in range(len(pred_masks))))
    
        # 如果原始图像不是列表而是张量，则转换为 numpy 数组的批处理
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    
        results = []
        # 遍历每个预测的掩码、原始图像和图像路径
        for masks, orig_img, img_path in zip([pred_masks], orig_imgs, self.batch[0]):
            # 如果存在预测的边界框，则调整边界框大小
            if pred_bboxes is not None:
                pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(), orig_img.shape, padding=False)
                cls = torch.arange(len(pred_masks), dtype=torch.int32, device=pred_masks.device)
                pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)
    
            # 调整预测的掩码大小
            masks = ops.scale_masks(masks[None].float(), orig_img.shape[:2], padding=False)[0]
            # 应用掩码阈值，转换为布尔值
            masks = masks > self.model.mask_threshold
            # 添加处理后的结果到结果列表
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=pred_bboxes))
    
        # 重置分割所有类别的模式标志
        self.segment_all = False
        # 返回结果列表
        return results
    
    
    def setup_source(self, source):
        """
        Sets up the data source for inference.
    
        This method configures the data source from which images will be fetched for inference. The source could be a
        directory, a video file, or other types of image data sources.
    
        Args:
            source (str | Path): The path to the image data source for inference.
        """
        
        # 如果源路径不为 None，则调用父类方法设置数据源
        if source is not None:
            super().setup_source(source)
    def set_image(self, image):
        """
        Preprocesses and sets a single image for inference.

        This function sets up the model if not already initialized, configures the data source to the specified image,
        and preprocesses the image for feature extraction. Only one image can be set at a time.

        Args:
            image (str | np.ndarray): Image file path as a string, or a np.ndarray image read by cv2.

        Raises:
            AssertionError: If more than one image is set.
        """
        # 如果模型尚未初始化，则根据给定的模型参数构建 SAM 模型
        if self.model is None:
            model = build_sam(self.args.model)
            # 初始化模型
            self.setup_model(model)
        
        # 配置数据源为指定的图像
        self.setup_source(image)
        
        # 检查数据集中是否只有一个图像，否则引发断言错误
        assert len(self.dataset) == 1, "`set_image` only supports setting one image!"
        
        # 遍历数据集，预处理图像并提取特征，仅处理第一个 batch
        for batch in self.dataset:
            im = self.preprocess(batch[1])  # 对图像进行预处理
            self.features = self.model.image_encoder(im)  # 提取图像特征
            self.im = im  # 保存原始图像
            break

    def set_prompts(self, prompts):
        """Set prompts in advance."""
        # 设置预定义的提示语句
        self.prompts = prompts

    def reset_image(self):
        """Resets the image and its features to None."""
        # 重置图像和特征为 None
        self.im = None
        self.features = None

    @staticmethod
    def remove_small_regions(masks, min_area=0, nms_thresh=0.7):
        """
        Perform post-processing on segmentation masks generated by the Segment Anything Model (SAM). Specifically, this
        function removes small disconnected regions and holes from the input masks, and then performs Non-Maximum
        Suppression (NMS) to eliminate any newly created duplicate boxes.

        Args:
            masks (torch.Tensor): A tensor containing the masks to be processed. Shape should be (N, H, W), where N is
                                  the number of masks, H is height, and W is width.
            min_area (int): The minimum area below which disconnected regions and holes will be removed. Defaults to 0.
            nms_thresh (float): The IoU threshold for the NMS algorithm. Defaults to 0.7.

        Returns:
            (tuple([torch.Tensor, List[int]])):
                - new_masks (torch.Tensor): The processed masks with small regions removed. Shape is (N, H, W).
                - keep (List[int]): The indices of the remaining masks post-NMS, which can be used to filter the boxes.
        """
        import torchvision  # import statement needed for using torchvision

        if len(masks) == 0:
            return masks  # Return the input masks if empty

        # Filter small disconnected regions and holes
        new_masks = []  # Initialize an empty list for storing processed masks
        scores = []  # Initialize an empty list for storing scores of masks

        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)  # Convert mask tensor to numpy array of type uint8
            mask, changed = remove_small_regions(mask, min_area, mode="holes")  # Remove small holes
            unchanged = not changed  # Check if changes occurred in holes removal
            mask, changed = remove_small_regions(mask, min_area, mode="islands")  # Remove small islands
            unchanged = unchanged and not changed  # Check if changes occurred in islands removal

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))  # Convert processed mask back to tensor and append
            scores.append(float(unchanged))  # Append the score (0 or 1) indicating if mask was unchanged

        # Recalculate boxes and remove any new duplicates
        new_masks = torch.cat(new_masks, dim=0)  # Concatenate all masks into a single tensor
        boxes = batched_mask_to_box(new_masks)  # Convert masks to bounding boxes
        keep = torchvision.ops.nms(boxes.float(), torch.as_tensor(scores), nms_thresh)  # Perform NMS using scores

        return new_masks[keep].to(device=masks.device, dtype=masks.dtype), keep  # Return filtered masks and indices
```