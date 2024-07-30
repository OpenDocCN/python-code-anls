# `.\comic-translate\modules\inpainting\schema.py`

```py
# 导入必要的库：Optional用于可选类型提示，Enum用于定义枚举类型
from typing import Optional
from enum import Enum

# 从pydantic库导入BaseModel，用于定义配置类
from pydantic import BaseModel

# 定义高清处理策略的枚举类型，继承自str
class HDStrategy(str, Enum):
    # 使用原始图像尺寸
    ORIGINAL = "Original"
    # 将图像较长边缩放到特定尺寸（hd_strategy_resize_limit），然后在缩放后的图像上进行修复，最后将修复结果调整回原始尺寸
    # 蒙版外的区域不会失去质量
    RESIZE = "Resize"
    # 从原始图像中裁剪掩膜区域（由hd_strategy_crop_margin控制裁剪边缘）进行修复
    CROP = "Crop"

# 定义配置类Config，继承自BaseModel
class Config(BaseModel):
    class Config:
        # 允许任意类型的配置
        arbitrary_types_allowed = True

    # 配置项用于zits模型
    zits_wireframe: bool = True

    # 高清处理策略的配置项（用于预处理图像的不同方式）
    hd_strategy: str = HDStrategy.ORIGINAL  # 查看HDStrategy枚举类型
    hd_strategy_crop_margin: int = 512
    # 如果图像的较长边大于此值，使用裁剪策略
    hd_strategy_crop_trigger_size: int = 512
    hd_strategy_resize_limit: int = 512

    # 下面是一些被注释掉的配置项，不参与当前代码的功能实现
    # ldm模型的配置项
    # ldm_steps: int = 2
    # ldm_sampler: str = LDMSampler.plms

    # stable diffusion 1.5的配置项
    # prompt: str = ""
    # negative_prompt: str = ""
    # use_croper: bool = False
    # croper_x: int = None
    # croper_y: int = None
    # croper_height: int = None
    # croper_width: int = None
    # sd_scale: float = 1.0
    # sd_mask_blur: int = 0
    # sd_strength: float = 0.75
    # sd_steps: int = 50
    # sd_guidance_scale: float = 7.5
    # sd_sampler: str = SDSampler.uni_pc
    # sd_seed: int = 42
    # sd_match_histograms: bool = False

    # opencv修复的配置项
    # OpenCV文档中的链接，指向图像修复功能的具体说明页面
    # cv2_flag: 字符串变量，指定修复算法类型为“INPAINT_NS”
    # cv2_radius: 整数变量，指定修复算法的半径参数为4
    
    # 使用“Paint by Example”方法进行图像修复
    # paint_by_example_steps: 整数变量，指定修复步骤数为50
    # paint_by_example_guidance_scale: 浮点数变量，指定引导尺度为7.5
    # paint_by_example_mask_blur: 整数变量，指定蒙版模糊度为0
    # paint_by_example_seed: 整数变量，指定随机种子为42
    # paint_by_example_match_histograms: 布尔变量，指定是否匹配直方图为False
    # paint_by_example_example_image: 可选的图像变量，指定用于示例的图像，可以为空
    
    # 使用“InstructPix2Pix”方法进行图像修复
    # p2p_steps: 整数变量，指定修复步骤数为50
    # p2p_image_guidance_scale: 浮点数变量，指定图像引导尺度为1.5
    # p2p_guidance_scale: 浮点数变量，指定引导尺度为7.5
    
    # 使用“ControlNet”方法进行图像修复
    # controlnet_conditioning_scale: 浮点数变量，指定条件尺度为0.4
    # controlnet_method: 字符串变量，指定控制方法为“control_v11p_sd15_canny”
```