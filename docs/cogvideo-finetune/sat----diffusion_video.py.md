# `.\cogvideo-finetune\sat\diffusion_video.py`

```py
# 导入随机数生成模块
import random

# 导入数学模块
import math
# 导入类型提示相关模块
from typing import Any, Dict, List, Tuple, Union
# 导入OmegaConf中的ListConfig类
from omegaconf import ListConfig
# 导入PyTorch中的功能模块
import torch.nn.functional as F

# 从sat.helpers模块导入print_rank0函数
from sat.helpers import print_rank0
# 导入PyTorch库
import torch
# 从PyTorch中导入nn模块
from torch import nn

# 从sgm.modules导入UNCONDITIONAL_CONFIG
from sgm.modules import UNCONDITIONAL_CONFIG
# 从sgm.modules.autoencoding.temporal_ae导入VideoDecoder类
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
# 从sgm.modules.diffusionmodules.wrappers导入OPENAIUNETWRAPPER
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
# 从sgm.util导入多个实用函数
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)
# 导入垃圾回收模块
import gc
# 从sat导入mpu模块
from sat import mpu


# 定义SATVideoDiffusionEngine类，继承自nn.Module
class SATVideoDiffusionEngine(nn.Module):
    # 禁用不可训练参数的方法
    def disable_untrainable_params(self):
        # 初始化可训练参数的总数
        total_trainable = 0
        # 遍历模型的所有参数
        for n, p in self.named_parameters():
            # 如果参数不可训练，跳过
            if p.requires_grad == False:
                continue
            # 初始化标志
            flag = False
            # 检查参数名是否以不可训练前缀开头
            for prefix in self.not_trainable_prefixes:
                if n.startswith(prefix) or prefix == "all":
                    flag = True
                    break

            # 定义LoRA前缀列表
            lora_prefix = ["matrix_A", "matrix_B"]
            # 检查参数名中是否包含LoRA前缀
            for prefix in lora_prefix:
                if prefix in n:
                    flag = False
                    break

            # 如果标志为真，禁用参数训练
            if flag:
                p.requires_grad_(False)
            else:
                # 统计可训练参数的数量
                total_trainable += p.numel()

        # 打印可训练参数的总数
        print_rank0("***** Total trainable parameters: " + str(total_trainable) + " *****")

    # 重初始化方法
    def reinit(self, parent_model=None):
        # 重新加载之前训练模块的初始参数
        # 可以通过parent_model.get_mixin()访问其他混合模型
        pass

    # 初始化第一个阶段的方法
    def _init_first_stage(self, config):
        # 根据配置实例化模型并设置为评估模式
        model = instantiate_from_config(config).eval()
        # 禁用训练模式
        model.train = disabled_train
        # 禁用模型参数的训练
        for param in model.parameters():
            param.requires_grad = False
        # 设置第一个阶段模型
        self.first_stage_model = model

    # 前向传播方法
    def forward(self, x, batch):
        # 计算损失
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        # 计算损失的平均值
        loss_mean = loss.mean()
        # 创建损失字典
        loss_dict = {"loss": loss_mean}
        # 返回平均损失和损失字典
        return loss_mean, loss_dict

    # 向第一帧添加噪声的方法
    def add_noise_to_first_frame(self, image):
        # 生成噪声标准差
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(self.device)
        # 将标准差转化为指数形式
        sigma = torch.exp(sigma).to(image.dtype)
        # 生成与图像同样形状的随机噪声
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        # 将噪声添加到图像中
        image = image + image_noise
        # 返回添加噪声后的图像
        return image
    # 处理共享步骤，接收一个批次的输入数据，返回损失值及其字典
        def shared_step(self, batch: Dict) -> Any:
            # 获取输入数据
            x = self.get_input(batch)
            # 如果学习率缩放因子不为空
            if self.lr_scale is not None:
                # 对输入进行下采样
                lr_x = F.interpolate(x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False)
                # 还原到原始大小
                lr_x = F.interpolate(lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False)
                # 编码下采样后的输入
                lr_z = self.encode_first_stage(lr_x, batch)
                # 将编码结果存入批次字典
                batch["lr_input"] = lr_z
    
            # 调整维度以便后续处理
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            # 如果使用带噪声的图像输入
            if self.noised_image_input:
                # 取出第一个帧作为图像
                image = x[:, :, 0:1]
                # 对图像添加噪声
                image = self.add_noise_to_first_frame(image)
                # 编码添加噪声的图像
                image = self.encode_first_stage(image, batch)
    
            # 编码输入数据
            x = self.encode_first_stage(x, batch)
            # 调整维度以便后续处理
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            # 如果使用带噪声的图像输入
            if self.noised_image_input:
                # 调整噪声图像的维度
                image = image.permute(0, 2, 1, 3, 4).contiguous()
                # 如果所有噪声图像需要拼接
                if self.noised_image_all_concat:
                    # 重复图像以匹配输入
                    image = image.repeat(1, x.shape[1], 1, 1, 1)
                else:
                    # 拼接零填充的张量
                    image = torch.concat([image, torch.zeros_like(x[:, 1:])], dim=1)
                # 根据概率决定是否丢弃图像
                if random.random() < self.noised_image_dropout:
                    image = torch.zeros_like(image)
                # 将拼接的图像存入批次字典
                batch["concat_images"] = image
    
            # 收集垃圾
            gc.collect()
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
            # 计算损失及其字典
            loss, loss_dict = self(x, batch)
            # 返回损失及字典
            return loss, loss_dict
    
        # 从批次中获取输入，转化为指定类型
        def get_input(self, batch):
            return batch[self.input_key].to(self.dtype)
    
        # 无梯度上下文中解码第一阶段
        @torch.no_grad()
        def decode_first_stage(self, z):
            # 对潜在变量进行缩放
            z = 1.0 / self.scale_factor * z
            # 计算每次解码的样本数
            n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
            # 计算轮数
            n_rounds = math.ceil(z.shape[0] / n_samples)
            all_out = []
            # 使用自动混合精度进行解码
            with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
                # 循环解码每个批次
                for n in range(n_rounds):
                    # 如果解码器是视频解码器
                    if isinstance(self.first_stage_model.decoder, VideoDecoder):
                        kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                    else:
                        kwargs = {}
                    # 解码当前批次
                    out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples], **kwargs)
                    # 将解码结果添加到输出列表
                    all_out.append(out)
            # 合并所有解码结果
            out = torch.cat(all_out, dim=0)
            # 返回解码输出
            return out
    
        # 无梯度上下文中编码第一阶段
        @torch.no_grad()
        def encode_first_stage(self, x, batch):
            # 获取帧数
            frame = x.shape[2]
    
            # 如果帧数大于1且输入为潜在变量
            if frame > 1 and self.latent_input:
                # 调整维度
                x = x.permute(0, 2, 1, 3, 4).contiguous()
                # 返回已编码的输入
                return x * self.scale_factor  # already encoded
    
            # 计算每次编码的样本数
            n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
            # 计算轮数
            n_rounds = math.ceil(x.shape[0] / n_samples)
            all_out = []
            # 使用自动混合精度进行编码
            with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
                # 循环编码每个批次
                for n in range(n_rounds):
                    # 编码当前批次
                    out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples])
                    # 将编码结果添加到输出列表
                    all_out.append(out)
            # 合并所有编码结果
            z = torch.cat(all_out, dim=0)
            # 对编码结果进行缩放
            z = self.scale_factor * z
            # 返回编码输出
            return z
    
        # 无梯度上下文中
    # 定义一个样本生成函数，接受条件、超参数和其他可选参数
    def sample(
        self,
        cond: Dict,  # 输入条件的字典
        uc: Union[Dict, None] = None,  # 可选的无条件输入，默认为 None
        batch_size: int = 16,  # 每次生成的样本数量，默认为 16
        shape: Union[None, Tuple, List] = None,  # 样本形状，默认为 None
        prefix=None,  # 可选的前缀，用于生成的样本
        concat_images=None,  # 用于连接的图像，默认为 None
        **kwargs,  # 其他关键字参数
    ):
        # 生成一个随机的高斯噪声张量，形状为 (batch_size, *shape)，并转为 float32 类型
        randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)
        # 如果对象有已设置的噪声，则用该噪声处理 randn
        if hasattr(self, "seeded_noise"):
            randn = self.seeded_noise(randn)

        # 如果前缀不为 None，将前缀与 randn 进行拼接
        if prefix is not None:
            randn = torch.cat([prefix, randn[:, prefix.shape[1] :]], dim=1)

        # 获取模型并行的世界大小，用于广播噪声
        mp_size = mpu.get_model_parallel_world_size()
        # 如果模型并行的世界大小大于 1
        if mp_size > 1:
            # 计算当前全局 rank 和源节点
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            # 广播 randn 到模型并行组的所有节点
            torch.distributed.broadcast(randn, src=src, group=mpu.get_model_parallel_group())

        scale = None  # 初始化缩放因子为 None
        scale_emb = None  # 初始化缩放嵌入为 None

        # 定义去噪器的 lambda 函数，使用模型进行去噪
        denoiser = lambda input, sigma, c, **addtional_model_inputs: self.denoiser(
            self.model, input, sigma, c, concat_images=concat_images, **addtional_model_inputs
        )

        # 调用采样器生成样本，传入去噪器、随机噪声和条件等参数
        samples = self.sampler(denoiser, randn, cond, uc=uc, scale=scale, scale_emb=scale_emb)
        # 将生成的样本转换为指定的数据类型
        samples = samples.to(self.dtype)
        # 返回生成的样本
        return samples

    # 使用 torch.no_grad 装饰器，禁止梯度计算以节省内存
    @torch.no_grad()
    # 定义日志记录函数，用于记录不同的条件
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        定义记录不同条件的启发式方法。
        这些可以是字符串列表（文本到图像）、张量、整数等。
        """
        # 获取输入图像的高度和宽度
        image_h, image_w = batch[self.input_key].shape[3:]
        log = dict()  # 初始化日志字典

        # 遍历条件嵌入器进行记录
        for embedder in self.conditioner.embedders:
            # 检查是否需要记录该嵌入器的条件
            if ((self.log_keys is None) or (embedder.input_key in self.log_keys)) and not self.no_cond_log:
                # 选取批次中的前 n 个样本
                x = batch[embedder.input_key][:n]
                # 如果 x 是张量
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # 如果是类条件，转换整数为字符串
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        # 将文本转换为图像进行记录
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # 如果是二维张量，处理条件等
                        x = ["x".join([str(xx) for xx in x[i].tolist()]) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        # 如果维度不支持，抛出未实现错误
                        raise NotImplementedError()
                # 如果 x 是列表或列表配置
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # 如果是字符串列表，转换为图像记录
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        # 否则抛出未实现错误
                        raise NotImplementedError()
                else:
                    # 如果类型不支持，抛出未实现错误
                    raise NotImplementedError()
                # 将记录的内容加入日志字典
                log[embedder.input_key] = xc
        # 返回记录的日志字典
        return log

    # 使用 torch.no_grad 装饰器，禁止梯度计算以节省内存
    @torch.no_grad()
    # 定义一个日志记录视频的函数
        def log_video(
            self,  # 该方法的调用对象
            batch: Dict,  # 输入参数，表示一批数据，类型为字典
            N: int = 8,  # 可选参数，表示要处理的视频数量，默认为 8
            ucg_keys: List[str] = None,  # 可选参数，表示用户生成内容的关键字列表，默认为 None
            only_log_video_latents=False,  # 可选参数，布尔值，表示是否仅记录视频潜在变量，默认为 False
            **kwargs,  # 可变参数，允许传入额外的关键字参数
    ) -> Dict:
        # 从 conditioner 的 embedders 中提取输入键
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        # 如果定义了 ucg_keys
        if ucg_keys:
            # 断言所有 ucg_keys 都在 conditioner_input_keys 中
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        # 如果没有定义 ucg_keys，则使用 conditioner_input_keys
        else:
            ucg_keys = conditioner_input_keys
        # 初始化日志字典
        log = dict()

        # 获取输入数据
        x = self.get_input(batch)

        # 获取无条件的条件
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            # 如果有 embedders，则传入 ucg_keys，否则为空列表
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
        )

        # 初始化采样参数字典
        sampling_kwargs = {}

        # 获取输入数据的最小批大小
        N = min(x.shape[0], N)
        # 将输入数据转移到指定设备，并限制为前 N 个
        x = x.to(self.device)[:N]
        # 如果不是潜在输入，则将输入转为浮点32
        if not self.latent_input:
            log["inputs"] = x.to(torch.float32)
        # 调整输入数据的维度
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # 编码第一阶段的输入数据
        z = self.encode_first_stage(x, batch)
        # 如果不是只记录视频潜在
        if not only_log_video_latents:
            # 解码潜在 z 并转为浮点32
            log["reconstructions"] = self.decode_first_stage(z).to(torch.float32)
            # 调整重构数据的维度
            log["reconstructions"] = log["reconstructions"].permute(0, 2, 1, 3, 4).contiguous()
        # 调整潜在 z 的维度
        z = z.permute(0, 2, 1, 3, 4).contiguous()

        # 更新日志字典，记录条件
        log.update(self.log_conditionings(batch, N))

        # 遍历条件 c 的每个键
        for k in c:
            # 如果条件是张量类型
            if isinstance(c[k], torch.Tensor):
                # 从条件中获取前 N 个数据并转移到设备
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        # 如果有噪声图像输入
        if self.noised_image_input:
            # 选择输入数据的第一帧
            image = x[:, :, 0:1]
            # 向第一帧添加噪声
            image = self.add_noise_to_first_frame(image)
            # 编码第一帧
            image = self.encode_first_stage(image, batch)
            # 调整图像的维度
            image = image.permute(0, 2, 1, 3, 4).contiguous()
            # 将图像与潜在 z 的后续帧拼接
            image = torch.concat([image, torch.zeros_like(z[:, 1:])], dim=1)
            # 将拼接图像添加到条件字典
            c["concat"] = image
            uc["concat"] = image
            # 进行采样，生成样本
            samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w
            # 调整样本的维度
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            # 如果只记录视频潜在
            if only_log_video_latents:
                # 计算潜在变量并记录
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                # 解码样本并转为浮点32
                samples = self.decode_first_stage(samples).to(torch.float32)
                # 调整样本的维度
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                # 将样本添加到日志中
                log["samples"] = samples
        else:
            # 进行采样，生成样本
            samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w
            # 调整样本的维度
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            # 如果只记录视频潜在
            if only_log_video_latents:
                # 计算潜在变量并记录
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                # 解码样本并转为浮点32
                samples = self.decode_first_stage(samples).to(torch.float32)
                # 调整样本的维度
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                # 将样本添加到日志中
                log["samples"] = samples
        # 返回日志字典
        return log
```