# `stable-diffusion-webui\modules\sd_hijack.py`

```py
# 导入 torch 库
import torch
# 从 torch.nn.functional 中导入 silu 函数
from torch.nn.functional import silu
# 从 types 模块中导入 MethodType 类
from types import MethodType

# 从 modules 模块中导入 devices, sd_hijack_optimizations, shared, script_callbacks, errors, sd_unet, patches
from modules import devices, sd_hijack_optimizations, shared, script_callbacks, errors, sd_unet, patches
# 从 modules.hypernetworks 模块中导入 hypernetwork
from modules.hypernetworks import hypernetwork
# 从 modules.shared 模块中导入 cmd_opts
from modules.shared import cmd_opts
# 从 modules 中导入 sd_hijack_clip, sd_hijack_open_clip, sd_hijack_unet, sd_hijack_xlmr, xlmr, xlmr_m18
from modules import sd_hijack_clip, sd_hijack_open_clip, sd_hijack_unet, sd_hijack_xlmr, xlmr, xlmr_m18

# 从 ldm.modules.attention 模块中导入 CrossAttention_forward 函数
import ldm.modules.attention
# 从 ldm.modules.diffusionmodules.model 模块中导入 nonlinearity 函数
import ldm.modules.diffusionmodules.model
# 从 ldm.modules.diffusionmodules.openaimodel 模块中导入 openaimodel
import ldm.modules.diffusionmodules.openaimodel
# 从 ldm.models.diffusion 模块中导入 ddpm, ddim, plms
import ldm.models.diffusion.ddpm
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms
# 从 ldm.modules.encoders.modules 模块中导入 modules

# 从 sgm.modules.attention 模块中导入 attention
import sgm.modules.attention
# 从 sgm.modules.diffusionmodules.model 模块中导入 model
import sgm.modules.diffusionmodules.model
# 从 sgm.modules.diffusionmodules.openaimodel 模块中导入 openaimodel
import sgm.modules.diffusionmodules.openaimodel
# 从 sgm.modules.encoders.modules 模块中导入 modules

# 将 ldm.modules.attention.CrossAttention 中的 forward 函数赋值给 attention_CrossAttention_forward
attention_CrossAttention_forward = ldm.modules.attention.CrossAttention.forward
# 将 ldm.modules.diffusionmodules.model 中的 nonlinearity 函数赋值给 diffusionmodules_model_nonlinearity
diffusionmodules_model_nonlinearity = ldm.modules.diffusionmodules.model.nonlinearity
# 将 ldm.modules.diffusionmodules.model.AttnBlock 中的 forward 函数赋值给 diffusionmodules_model_AttnBlock_forward

# 禁用 SD2.0 的内存高效交叉注意力
ldm.modules.attention.MemoryEfficientCrossAttention = ldm.modules.attention.CrossAttention
ldm.modules.attention.BasicTransformerBlock.ATTENTION_MODES["softmax-xformers"] = ldm.modules.attention.CrossAttention

# 禁止 SD2 的新控制台输出
ldm.modules.attention.print = shared.ldm_print
ldm.modules.diffusionmodules.model.print = shared.ldm_print
ldm.util.print = shared.ldm_print
ldm.models.diffusion.ddpm.print = shared.ldm_print

# 初始化优化器列表
optimizers = []
# 初始化当前优化器
current_optimizer: sd_hijack_optimizations.SdOptimization = None

# 创建 ldm_patched_forward 函数，用于创建 SD UNet 模型的前向传播
ldm_patched_forward = sd_unet.create_unet_forward(ldm.modules.diffusionmodules.openaimodel.UNetModel.forward)
# 对指定文件进行补丁操作，将原始模型的 forward 方法替换为自定义的 patched_forward 方法
ldm_original_forward = patches.patch(__file__, ldm.modules.diffusionmodules.openaimodel.UNetModel, "forward", ldm_patched_forward)

# 创建一个新的 forward 方法，用于替换 sgm 模块中 UNetModel 的 forward 方法
sgm_patched_forward = sd_unet.create_unet_forward(sgm.modules.diffusionmodules.openaimodel.UNetModel.forward)
# 对 sgm 模块中 UNetModel 的 forward 方法进行补丁操作，替换为新的 patched_forward 方法
sgm_original_forward = patches.patch(__file__, sgm.modules.diffusionmodules.openaimodel.UNetModel, "forward", sgm_patched_forward)

# 列出所有可用的优化器
def list_optimizers():
    # 调用回调函数获取新的优化器列表
    new_optimizers = script_callbacks.list_optimizers_callback()
    
    # 过滤出可用的优化器
    new_optimizers = [x for x in new_optimizers if x.is_available()]
    
    # 根据优化器的优先级降序排序
    new_optimizers = sorted(new_optimizers, key=lambda x: x.priority, reverse=True)
    
    # 清空原有的优化器列表，将新的优化器列表添加进去
    optimizers.clear()
    optimizers.extend(new_optimizers)

# 应用优化操作
def apply_optimizations(option=None):
    global current_optimizer
    
    # 撤销之前的优化操作
    undo_optimizations()
    
    # 如果优化器列表为空，则返回空字符串
    if len(optimizers) == 0:
        # 一个脚本可能会在很早的阶段访问模型，此时优化器可能还未填充
        current_optimizer = None
        return ''
    
    # 设置 ldm 和 sgm 模块中的非线性函数为 silu
    ldm.modules.diffusionmodules.model.nonlinearity = silu
    ldm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th
    
    sgm.modules.diffusionmodules.model.nonlinearity = silu
    sgm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th
    
    # 如果当前优化器不为空，则撤销当前优化操作
    if current_optimizer is not None:
        current_optimizer.undo()
        current_optimizer = None
    
    # 根据选项选择匹配的优化器
    selection = option or shared.opts.cross_attention_optimization
    if selection == "Automatic" and len(optimizers) > 0:
        matching_optimizer = next(iter([x for x in optimizers if x.cmd_opt and getattr(shared.cmd_opts, x.cmd_opt, False)]), optimizers[0])
    else:
        matching_optimizer = next(iter([x for x in optimizers if x.title() == selection]), None)
    
    # 如果选项为 "None"，则匹配的优化器为 None
    if selection == "None":
        matching_optimizer = None
    # 如果选项为 "Automatic" 且禁用了 split attention 优化，则匹配的优化器为 None
    elif selection == "Automatic" and shared.cmd_opts.disable_opt_split_attention:
        matching_optimizer = None
    # 如果没有匹配到选项对应的优化器，则选择第一个优化器
    elif matching_optimizer is None:
        matching_optimizer = optimizers[0]
    # 如果匹配优化器不为空
    if matching_optimizer is not None:
        # 打印应用注意力优化器的信息
        print(f"Applying attention optimization: {matching_optimizer.name}... ", end='')
        # 应用匹配的优化器
        matching_optimizer.apply()
        # 打印完成信息
        print("done.")
        # 将当前优化器设置为匹配的优化器
        current_optimizer = matching_optimizer
        # 返回当前优化器的名称
        return current_optimizer.name
    # 如果匹配优化器为空
    else:
        # 打印禁用注意力优化器的信息
        print("Disabling attention optimization")
        # 返回空字符串
        return ''
def undo_optimizations():
    # 恢复模块的优化设置，将非线性模块、交叉注意力模块和注意力块模块的 forward 方法设置为原始值
    ldm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward

    sgm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    sgm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    sgm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward


def fix_checkpoint():
    """checkpoints are now added and removed in embedding/hypernet code, since torch doesn't want
    checkpoints to be added when not training (there's a warning)"""
    # 修复检查点，将检查点的添加和移除放在嵌入/超网络代码中，因为 torch 不希望在非训练时添加检查点（会有警告）
    pass


def weighted_loss(sd_model, pred, target, mean=True):
    # 计算加权损失，但忽略均值
    loss = sd_model._old_get_loss(pred, target, mean=False)

    # 检查是否有可用的权重
    weight = getattr(sd_model, '_custom_loss_weight', None)
    if weight is not None:
        loss *= weight

    # 返回损失，如果指定了均值则返回均值
    return loss.mean() if mean else loss

def weighted_forward(sd_model, x, c, w, *args, **kwargs):
    try:
        # 临时将权重附加到在损失计算期间可访问的位置
        sd_model._custom_loss_weight = w

        # 用支持权重的方法替换 'get_loss'。否则需要完全重新实现 'forward'
        # 保留 'get_loss'，但如果已经设置了旧的 old_get_loss，则不覆盖它
        if not hasattr(sd_model, '_old_get_loss'):
            sd_model._old_get_loss = sd_model.get_loss
        sd_model.get_loss = MethodType(weighted_loss, sd_model)

        # 运行标准的 forward 函数，但使用修补过的 'get_loss'
        return sd_model.forward(x, c, *args, **kwargs)
    finally:
        try:
            # 如果存在自定义的损失权重，删除临时权重
            del sd_model._custom_loss_weight
        except AttributeError:
            pass

        # 如果存在旧的损失函数，将损失函数重置为原始的损失函数
        if hasattr(sd_model, '_old_get_loss'):
            sd_model.get_loss = sd_model._old_get_loss
            del sd_model._old_get_loss
# 为给定的 SD 模型添加一个新的方法 'weighted_forward'，用于计算加权损失
def apply_weighted_forward(sd_model):
    sd_model.weighted_forward = MethodType(weighted_forward, sd_model)

# 删除 SD 模型中的 'weighted_forward' 方法
def undo_weighted_forward(sd_model):
    try:
        del sd_model.weighted_forward
    except AttributeError:
        pass

# 定义一个稳定扩散模型劫持类
class StableDiffusionModelHijack:
    # 初始化类属性
    fixes = None
    layers = None
    circular_enabled = False
    clip = None
    optimization_method = None

    # 初始化方法
    def __init__(self):
        # 导入模块
        import modules.textual_inversion.textual_inversion

        # 初始化实例属性
        self.extra_generation_params = {}
        self.comments = []

        # 创建嵌入数据库对象
        self.embedding_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase()
        self.embedding_db.add_embedding_dir(cmd_opts.embeddings_dir)

    # 应用优化方法
    def apply_optimizations(self, option=None):
        try:
            # 调用 apply_optimizations 方法
            self.optimization_method = apply_optimizations(option)
        except Exception as e:
            # 显示错误信息
            errors.display(e, "applying cross attention optimization")
            # 撤销优化
            undo_optimizations()

    # 将 SDXL 模型转换为 Segmind 稳定扩散模型
    def convert_sdxl_to_ssd(self, m):
        """Converts an SDXL model to a Segmind Stable Diffusion model (see https://huggingface.co/segmind/SSD-1B)"""

        # 删除指定属性
        delattr(m.model.diffusion_model.middle_block, '1')
        delattr(m.model.diffusion_model.middle_block, '2')
        for i in ['9', '8', '7', '6', '5', '4']:
            delattr(m.model.diffusion_model.input_blocks[7][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.input_blocks[8][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.output_blocks[0][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.output_blocks[1][1].transformer_blocks, i)
        delattr(m.model.diffusion_model.output_blocks[4][1].transformer_blocks, '1')
        delattr(m.model.diffusion_model.output_blocks[5][1].transformer_blocks, '1')
        # 执行 torch 垃圾回收
        devices.torch_gc()
    # 撤销 hijack 操作，将被 hijack 的模型还原为原始状态
    def undo_hijack(self, m):
        # 获取模型中的 conditioner 属性，如果不存在则赋值为 None
        conditioner = getattr(m, 'conditioner', None)
        # 如果 conditioner 存在
        if conditioner:
            # 遍历 conditioner 中的 embedders
            for i in range(len(conditioner.embedders)):
                # 获取 embedder
                embedder = conditioner.embedders[i]
                # 如果 embedder 是指定类型的实例
                if isinstance(embedder, (sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords, sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords)):
                    # 将 embedder 中的 token_embedding 进行解封装
                    embedder.wrapped.model.token_embedding = embedder.wrapped.model.token_embedding.wrapped
                    # 更新 conditioner 中的 embedder
                    conditioner.embedders[i] = embedder.wrapped
                # 如果 embedder 是指定类型的实例
                if isinstance(embedder, sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords):
                    # 将 embedder 中的 token_embedding 进行解封装
                    embedder.wrapped.transformer.text_model.embeddings.token_embedding = embedder.wrapped.transformer.text_model.embeddings.token_embedding.wrapped
                    # 更新 conditioner 中的 embedder
                    conditioner.embedders[i] = embedder.wrapped

            # 如果模型中存在 cond_stage_model 属性
            if hasattr(m, 'cond_stage_model'):
                # 删除 cond_stage_model 属性

        # 如果 cond_stage_model 的类型是指定类型的实例
        elif type(m.cond_stage_model) == sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords:
            # 将 cond_stage_model 进行解封装
            m.cond_stage_model = m.cond_stage_model.wrapped

        # 如果 cond_stage_model 的类型是指定类型的实例
        elif type(m.cond_stage_model) == sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords:
            # 将 cond_stage_model 进行解封装
            m.cond_stage_model = m.cond_stage_model.wrapped

            # 获取模型中的 embeddings
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            # 如果 embeddings 中的 token_embedding 的类型是指定类型的实例
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                # 将 token_embedding 进行解封装
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
        # 如果 cond_stage_model 的类型是指定类型的实例
        elif type(m.cond_stage_model) == sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords:
            # 将 cond_stage_model 中的 token_embedding 进行解封装
            m.cond_stage_model.wrapped.model.token_embedding = m.cond_stage_model.wrapped.model.token_embedding.wrapped
            # 更新 cond_stage_model

        # 撤销优化操作
        undo_optimizations()
        # 撤销前向传播权重
        undo_weighted_forward(m)

        # 应用循环操作
        self.apply_circular(False)
        # 重置 layers 属性为 None
        self.layers = None
        # 重置 clip 属性为 None
        self.clip = None
    # 根据传入的参数 enable 来设置是否启用循环填充
    def apply_circular(self, enable):
        # 如果当前的 circular_enabled 状态与传入参数相同，则直接返回
        if self.circular_enabled == enable:
            return

        # 更新 circular_enabled 状态为传入参数的值
        self.circular_enabled = enable

        # 遍历所有的层，找到类型为 torch.nn.Conv2d 的层，设置填充模式为循环或零填充
        for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    # 清空评论和额外生成参数
    def clear_comments(self):
        # 清空评论列表
        self.comments = []
        # 清空额外生成参数字典
        self.extra_generation_params = {}

    # 获取文本的提示长度
    def get_prompt_lengths(self, text):
        # 如果 clip 为 None，则返回 "-"，"-"
        if self.clip is None:
            return "-", "-"

        # 处理文本，获取 token 数量
        _, token_count = self.clip.process_texts([text])

        # 返回 token 数量和目标提示 token 数量
        return token_count, self.clip.get_target_prompt_token_count(token_count)

    # 重新执行 hijack 操作
    def redo_hijack(self, m):
        # 先撤销 hijack 操作
        self.undo_hijack(m)
        # 再执行 hijack 操作
        self.hijack(m)
# 定义一个继承自 torch.nn.Module 的类 EmbeddingsWithFixes
class EmbeddingsWithFixes(torch.nn.Module):
    # 初始化方法，接受 wrapped、embeddings 和 textual_inversion_key 三个参数
    def __init__(self, wrapped, embeddings, textual_inversion_key='clip_l'):
        super().__init__()
        # 将传入的 wrapped 赋值给 self.wrapped
        self.wrapped = wrapped
        # 将传入的 embeddings 赋值给 self.embeddings
        self.embeddings = embeddings
        # 将传入的 textual_inversion_key 赋值给 self.textual_inversion_key

    # 前向传播方法，接受 input_ids 作为输入
    def forward(self, input_ids):
        # 获取 embeddings 对象的 fixes 属性，赋值给 batch_fixes
        batch_fixes = self.embeddings.fixes
        # 将 embeddings 对象的 fixes 属性设置为 None
        self.embeddings.fixes = None

        # 调用 wrapped 对象的前向传播方法，传入 input_ids，获取结果
        inputs_embeds = self.wrapped(input_ids)

        # 如果 batch_fixes 为 None 或者长度为 0，或者最大长度为 0
        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            # 返回 inputs_embeds
            return inputs_embeds

        # 初始化一个空列表 vecs
        vecs = []
        # 遍历 batch_fixes 和 inputs_embeds
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            # 遍历 fixes 中的 offset 和 embedding
            for offset, embedding in fixes:
                # 如果 embedding 的 vec 是字典，则取出 key 为 textual_inversion_key 的值，否则取出 vec
                vec = embedding.vec[self.textual_inversion_key] if isinstance(embedding.vec, dict) else embedding.vec
                # 对 vec 进行类型转换
                emb = devices.cond_cast_unet(vec)
                # 计算 emb 的长度
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                # 在 tensor 的 offset 处插入 emb
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])

            # 将处理后的 tensor 添加到 vecs 中
            vecs.append(tensor)

        # 将 vecs 中的 tensor 拼接成一个张量返回
        return torch.stack(vecs)


# 定义一个函数，为 torch.nn.Conv2d 类添加 circular 选项
def add_circular_option_to_conv_2d():
    # 获取 torch.nn.Conv2d 的初始化方法
    conv2d_constructor = torch.nn.Conv2d.__init__

    # 定义一个新的初始化方法，添加 padding_mode='circular' 参数
    def conv2d_constructor_circular(self, *args, **kwargs):
        return conv2d_constructor(self, *args, padding_mode='circular', **kwargs)

    # 将新的初始化方法赋值给 torch.nn.Conv2d 的初始化方法
    torch.nn.Conv2d.__init__ = conv2d_constructor_circular


# 创建一个 StableDiffusionModelHijack 对象
model_hijack = StableDiffusionModelHijack()


# 定义一个 register_buffer 函数，修复 Mac OS 的 register buffer bug
def register_buffer(self, name, attr):
    """
    Fix register buffer bug for Mac OS.
    """

    # 如果 attr 的类型是 torch.Tensor
    if type(attr) == torch.Tensor:
        # 如果 attr 的设备不是 devices.device
        if attr.device != devices.device:
            # 将 attr 转移到 devices.device，并设置数据类型
            attr = attr.to(device=devices.device, dtype=(torch.float32 if devices.device.type == 'mps' else None))

    # 将 attr 设置为 self 的属性名为 name
    setattr(self, name, attr)


# 将 register_buffer 函数赋值给 DDIMSampler 类的 register_buffer 方法
ldm.models.diffusion.ddim.DDIMSampler.register_buffer = register_buffer
# 将 register_buffer 函数赋值给 PLMSSampler 类的 register_buffer 方法
ldm.models.diffusion.plms.PLMSSampler.register_buffer = register_buffer
```