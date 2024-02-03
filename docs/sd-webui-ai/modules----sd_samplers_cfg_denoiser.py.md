# `stable-diffusion-webui\modules\sd_samplers_cfg_denoiser.py`

```py
# 导入 torch 库
import torch
# 从 modules 模块中导入 prompt_parser, devices, sd_samplers_common
from modules import prompt_parser, devices, sd_samplers_common

# 从 modules.shared 模块中导入 opts, state
from modules.shared import opts, state
# 将 modules.shared 重命名为 shared
import modules.shared as shared
# 从 modules.script_callbacks 模块中导入 CFGDenoiserParams, cfg_denoiser_callback
from modules.script_callbacks import CFGDenoiserParams, cfg_denoiser_callback
# 从 modules.script_callbacks 模块中导入 CFGDenoisedParams, cfg_denoised_callback
from modules.script_callbacks import CFGDenoisedParams, cfg_denoised_callback
# 从 modules.script_callbacks 模块中导入 AfterCFGCallbackParams, cfg_after_cfg_callback

# 定义函数 catenate_conds，用于拼接条件
def catenate_conds(conds):
    # 如果 conds 的第一个元素不是字典，则直接拼接
    if not isinstance(conds[0], dict):
        return torch.cat(conds)

    # 如果 conds 的第一个元素是字典，则按键拼接值
    return {key: torch.cat([x[key] for x in conds]) for key in conds[0].keys()}

# 定义函数 subscript_cond，用于对条件进行切片
def subscript_cond(cond, a, b):
    # 如果 cond 不是字典，则直接切片
    if not isinstance(cond, dict):
        return cond[a:b]

    # 如果 cond 是字典，则按键切片值
    return {key: vec[a:b] for key, vec in cond.items()}

# 定义函数 pad_cond，用于对条件进行填充
def pad_cond(tensor, repeats, empty):
    # 如果 tensor 不是字典，则直接填充
    if not isinstance(tensor, dict):
        return torch.cat([tensor, empty.repeat((tensor.shape[0], repeats, 1))], axis=1)

    # 如果 tensor 是字典，则对 'crossattn' 键的值进行填充
    tensor['crossattn'] = pad_cond(tensor['crossattn'], repeats, empty)
    return tensor

# 定义类 CFGDenoiser，继承自 torch.nn.Module
class CFGDenoiser(torch.nn.Module):
    """
    Classifier free guidance denoiser. A wrapper for stable diffusion model (specifically for unet)
    that can take a noisy picture and produce a noise-free picture using two guidances (prompts)
    instead of one. Originally, the second prompt is just an empty string, but we use non-empty
    negative prompt.
    """

    # 初始化方法
    def __init__(self, sampler):
        super().__init__()
        self.model_wrap = None
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.steps = None
        """number of steps as specified by user in UI"""

        self.total_steps = None
        """expected number of calls to denoiser calculated from self.steps and specifics of the selected sampler"""

        self.step = 0
        self.image_cfg_scale = None
        self.padded_cond_uncond = False
        self.sampler = sampler
        self.model_wrap = None
        self.p = None
        self.mask_before_denoising = False

    @property
    # 定义一个抽象方法，子类需要实现该方法
    def inner_model(self):
        raise NotImplementedError()

    # 将去噪后的输出数据与条件列表结合，返回结合后的数据
    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        # 获取无条件数据
        denoised_uncond = x_out[-uncond.shape[0]:]
        # 克隆无条件数据
        denoised = torch.clone(denoised_uncond)

        # 遍历条件列表
        for i, conds in enumerate(conds_list):
            # 遍历每个条件及其权重
            for cond_index, weight in conds:
                # 根据条件和权重更新去噪后的数据
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)

        return denoised

    # 为编辑模型结合去噪后的输出数据，返回结合后的数据
    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        # 将输出数据分成条件数据、图像条件数据和无条件数据
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        # 结合无条件数据、条件数据和图像条件数据，返回结合后的数据
        denoised = out_uncond + cond_scale * (out_cond - out_img_cond) + self.image_cfg_scale * (out_img_cond - out_uncond)

        return denoised

    # 获取预测的 x0 数据
    def get_pred_x0(self, x_in, x_out, sigma):
        return x_out

    # 更新内部模型
    def update_inner_model(self):
        # 将模型包装对象置为 None
        self.model_wrap = None

        # 获取条件数据和无条件数据
        c, uc = self.p.get_conds()
        # 更新采样器的额外参数中的条件和无条件数据
        self.sampler.sampler_extra_args['cond'] = c
        self.sampler.sampler_extra_args['uncond'] = uc
```