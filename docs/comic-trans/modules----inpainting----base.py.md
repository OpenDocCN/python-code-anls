# `.\comic-translate\modules\inpainting\base.py`

```py
import abc  # 导入抽象基类模块
from typing import Optional  # 导入类型提示模块

import cv2  # 导入 OpenCV 模块
import torch  # 导入 PyTorch 模块
import numpy as np  # 导入 NumPy 模块
from loguru import logger  # 从 loguru 模块导入 logger

from ..utils.inpainting import (  # 导入自定义模块中的函数
    boxes_from_mask,
    resize_max_size,
    pad_img_to_modulo,
    # switch_mps_device,
)
from .schema import Config, HDStrategy  # 从当前包中导入 Config 和 HDStrategy 类


class InpaintModel:
    name = "base"  # 类属性 name 设置为 "base"
    min_size: Optional[int] = None  # 可选的最小尺寸属性，默认为 None
    pad_mod = 8  # 填充模数为 8
    pad_to_square = False  # 不强制填充成正方形

    def __init__(self, device, **kwargs):
        """

        Args:
            device: 设备名称
            **kwargs: 其他关键字参数
        """
        # device = switch_mps_device(self.name, device)
        self.device = device  # 设置实例的设备属性
        self.init_model(device, **kwargs)  # 调用初始化模型的方法

    @abc.abstractmethod
    def init_model(self, device, **kwargs):
        ...

    @staticmethod
    @abc.abstractmethod
    def is_downloaded() -> bool:
        ...

    @abc.abstractmethod
    def forward(self, image, mask, config: Config):
        """输入和输出图像具有相同的尺寸
        images: [H, W, C] RGB
        masks: [H, W, 1] 255 为 masks 区域
        return: BGR IMAGE
        """
        ...

    def _pad_forward(self, image, mask, config: Config):
        origin_height, origin_width = image.shape[:2]  # 获取原始图像的高度和宽度
        pad_image = pad_img_to_modulo(
            image, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )  # 对图像进行填充，使其能被模数整除

        pad_mask = pad_img_to_modulo(
            mask, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )  # 对掩膜进行填充，使其能被模数整除

        logger.info(f"final forward pad size: {pad_image.shape}")  # 记录填充后的图像尺寸信息

        result = self.forward(pad_image, pad_mask, config)  # 调用前向传播方法

        result = result[0:origin_height, 0:origin_width, :]  # 裁剪结果到原始图像的尺寸

        result, image, mask = self.forward_post_process(result, image, mask, config)  # 调用前向传播后处理方法

        mask = mask[:, :, np.newaxis]  # 给掩膜增加一个通道维度
        result = result * (mask / 255) + image[:, :, ::-1] * (1 - (mask / 255))  # 合成最终输出结果

        return result  # 返回最终结果

    def forward_post_process(self, result, image, mask, config):
        return result, image, mask  # 默认的前向传播后处理方法

    @torch.no_grad()  # 禁用 PyTorch 的梯度计算
    def __call__(self, image, mask, config: Config):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        # 初始化 inpaint_result 为 None
        inpaint_result = None
        # 记录配置中的 hd_strategy 信息
        logger.info(f"hd_strategy: {config.hd_strategy}")

        # 如果配置中指定使用 HDStrategy.CROP 策略
        if config.hd_strategy == HDStrategy.CROP:
            # 如果图像的最大维度大于 hd_strategy_crop_trigger_size
            if max(image.shape) > config.hd_strategy_crop_trigger_size:
                # 记录运行裁剪策略的日志
                logger.info(f"Run crop strategy")
                # 从 mask 中获取 boxes
                boxes = boxes_from_mask(mask)
                crop_result = []
                # 遍历每个 box，执行裁剪操作
                for box in boxes:
                    crop_image, crop_box = self._run_box(image, mask, box, config)
                    crop_result.append((crop_image, crop_box))

                # 创建一个 BGR 形式的 inpaint_result
                inpaint_result = image[:, :, ::-1]
                # 将裁剪得到的图像填充回 inpaint_result 中的相应位置
                for crop_image, crop_box in crop_result:
                    x1, y1, x2, y2 = crop_box
                    inpaint_result[y1:y2, x1:x2, :] = crop_image

        # 如果配置中指定使用 HDStrategy.RESIZE 策略
        elif config.hd_strategy == HDStrategy.RESIZE:
            # 如果图像的最大维度大于 hd_strategy_resize_limit
            if max(image.shape) > config.hd_strategy_resize_limit:
                # 记录原始图像尺寸
                origin_size = image.shape[:2]
                # 缩小图像和 mask 到指定大小
                downsize_image = resize_max_size(
                    image, size_limit=config.hd_strategy_resize_limit
                )
                downsize_mask = resize_max_size(
                    mask, size_limit=config.hd_strategy_resize_limit
                )

                # 记录运行缩放策略的日志和前后的尺寸
                logger.info(
                    f"Run resize strategy, origin size: {image.shape} forward size: {downsize_image.shape}"
                )

                # 执行前向填充并生成 inpaint_result
                inpaint_result = self._pad_forward(
                    downsize_image, downsize_mask, config
                )

                # 将 inpaint_result 缩放回原始尺寸，仅粘贴 mask 区域的结果
                inpaint_result = cv2.resize(
                    inpaint_result,
                    (origin_size[1], origin_size[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                # 获取原始 mask 中像素值小于 127 的索引
                original_pixel_indices = mask < 127
                # 将原始图像中对应位置的像素值复制到 inpaint_result 中
                inpaint_result[original_pixel_indices] = image[:, :, ::-1][
                    original_pixel_indices
                ]

        # 如果 inpaint_result 仍为 None，则执行默认的前向填充
        if inpaint_result is None:
            inpaint_result = self._pad_forward(image, mask, config)

        # 返回最终的 inpaint_result
        return inpaint_result
    def _crop_box(self, image, mask, box, config: Config):
        """
        Args:
            image: [H, W, C] RGB，输入的原始图像
            mask: [H, W, 1]，掩码图像，用于标记感兴趣区域
            box: [left,top,right,bottom]，指定的感兴趣区域边界框

        Returns:
            BGR IMAGE, (l, r, r, b)，裁剪后的图像和更新后的边界框坐标
        """
        # 计算感兴趣区域的高度和宽度
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        
        # 计算感兴趣区域的中心点坐标
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        
        # 获取原始图像的高度和宽度
        img_h, img_w = image.shape[:2]
        
        # 根据配置参数计算裁剪框的宽度和高度
        w = box_w + config.hd_strategy_crop_margin * 2
        h = box_h + config.hd_strategy_crop_margin * 2
        
        # 计算裁剪框的左右上下边界
        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2
        
        # 调整裁剪框的边界，确保不超出图像范围
        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)
        
        # 如果裁剪框超出图像边界，尝试扩展图像边界以获取更多上下文
        if _l < 0:
            r += abs(_l)
        if _r > img_w:
            l -= _r - img_w
        if _t < 0:
            b += abs(_t)
        if _b > img_h:
            t -= _b - img_h
        
        # 再次确保裁剪框在图像范围内
        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)
        
        # 根据计算得到的边界框，裁剪原始图像和掩码图像
        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]
        
        # 记录裁剪后的图像尺寸信息
        logger.info(f"box size: ({box_h},{box_w}) crop size: {crop_img.shape}")
        
        # 返回裁剪后的图像、掩码和更新后的边界框坐标
        return crop_img, crop_mask, [l, t, r, b]

    def _calculate_cdf(self, histogram):
        """
        计算累积分布函数（CDF）。

        Args:
            histogram: 直方图数据

        Returns:
            normalized_cdf: 归一化后的累积分布函数
        """
        cdf = histogram.cumsum()
        normalized_cdf = cdf / float(cdf.max())
        return normalized_cdf

    def _calculate_lookup(self, source_cdf, reference_cdf):
        """
        计算匹配的查找表。

        Args:
            source_cdf: 源图像的累积分布函数
            reference_cdf: 参考图像的累积分布函数

        Returns:
            lookup_table: 查找表，用于映射源图像到参考图像的像素值
        """
        lookup_table = np.zeros(256)
        lookup_val = 0
        for source_index, source_val in enumerate(source_cdf):
            for reference_index, reference_val in enumerate(reference_cdf):
                if reference_val >= source_val:
                    lookup_val = reference_index
                    break
            lookup_table[source_index] = lookup_val
        return lookup_table

    def _match_histograms(self, source, reference, mask):
        """
        将源图像的直方图匹配到参考图像的直方图。

        Args:
            source: 源图像
            reference: 参考图像
            mask: 掩码图像，用于指定需要处理的区域

        Returns:
            result: 匹配后的结果图像
        """
        transformed_channels = []
        for channel in range(source.shape[-1]):
            source_channel = source[:, :, channel]
            reference_channel = reference[:, :, channel]

            # 仅计算非掩码部分的直方图
            source_histogram, _ = np.histogram(source_channel[mask == 0], 256, [0, 256])
            reference_histogram, _ = np.histogram(reference_channel[mask == 0], 256, [0, 256])

            # 计算源图像和参考图像的累积分布函数
            source_cdf = self._calculate_cdf(source_histogram)
            reference_cdf = self._calculate_cdf(reference_histogram)

            # 计算匹配的查找表
            lookup = self._calculate_lookup(source_cdf, reference_cdf)

            # 应用查找表，对源图像进行直方图匹配
            transformed_channels.append(cv2.LUT(source_channel, lookup))

        # 合并处理后的通道并转换为无符号8位整数格式
        result = cv2.merge(transformed_channels)
        result = cv2.convertScaleAbs(result)

        # 返回匹配后的结果图像
        return result
    # 对图像和掩模应用裁剪器，返回裁剪后的图像、掩模和裁剪框坐标
    def _apply_cropper(self, image, mask, config: Config):
        # 获取图像的高度和宽度
        img_h, img_w = image.shape[:2]
        # 从配置中获取裁剪框的左边界、上边界、宽度和高度
        l, t, w, h = (
            config.croper_x,
            config.croper_y,
            config.croper_width,
            config.croper_height,
        )
        # 计算裁剪框的右边界和底边界
        r = l + w
        b = t + h

        # 确保裁剪框不超出图像边界
        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        # 根据裁剪框坐标裁剪图像和掩模
        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]
        # 返回裁剪后的图像、掩模和裁剪框坐标
        return crop_img, crop_mask, (l, t, r, b)

    # 运行框操作，包括裁剪、填充和返回裁剪框坐标
    def _run_box(self, image, mask, box, config: Config):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE
        """
        # 使用给定的框（box）裁剪图像和掩模，并获取裁剪框坐标
        crop_img, crop_mask, [l, t, r, b] = self._crop_box(image, mask, box, config)

        # 对裁剪后的图像和掩模进行前向填充，得到BGR格式的图像，并返回裁剪框坐标
        return self._pad_forward(crop_img, crop_mask, config), [l, t, r, b]
class DiffusionInpaintModel(InpaintModel):
    @torch.no_grad()
    def __call__(self, image, mask, config: Config):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        # 如果配置中使用了croper，则使用裁剪器处理图像和掩码
        if config.use_croper:
            # 调用裁剪器方法，获取裁剪后的图像、掩码以及裁剪框的位置信息
            crop_img, crop_mask, (l, t, r, b) = self._apply_cropper(image, mask, config)
            # 对裁剪后的图像和掩码进行进一步的缩放填充处理
            crop_image = self._scaled_pad_forward(crop_img, crop_mask, config)
            # 创建一个新的图像，将原始图像的BGR颜色通道复制到新图像中对应位置
            inpaint_result = image[:, :, ::-1]
            # 将裁剪后的图像粘贴回原始图像的指定区域
            inpaint_result[t:b, l:r, :] = crop_image
        else:
            # 如果未使用裁剪器，则直接进行缩放填充处理
            inpaint_result = self._scaled_pad_forward(image, mask, config)

        return inpaint_result

    def _scaled_pad_forward(self, image, mask, config: Config):
        # 计算长边的长度，根据配置的尺度因子进行调整
        longer_side_length = int(config.sd_scale * max(image.shape[:2]))
        # 记录原始图像的尺寸
        origin_size = image.shape[:2]
        # 将图像和掩码按照最大尺寸进行缩放
        downsize_image = resize_max_size(image, size_limit=longer_side_length)
        downsize_mask = resize_max_size(mask, size_limit=longer_side_length)
        # 如果尺度因子不为1，记录调整后的图像尺寸信息
        if config.sd_scale != 1:
            logger.info(
                f"Resize image to do sd inpainting: {image.shape} -> {downsize_image.shape}"
            )
        # 使用填充方法进行图像修复
        inpaint_result = self._pad_forward(downsize_image, downsize_mask, config)
        # 将修复结果缩放回原始图像的尺寸
        inpaint_result = cv2.resize(
            inpaint_result,
            (origin_size[1], origin_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        # 仅将掩码区域内的修复结果粘贴回原始图像
        original_pixel_indices = mask < 127
        inpaint_result[original_pixel_indices] = image[:, :, ::-1][
            original_pixel_indices
        ]
        return inpaint_result
```