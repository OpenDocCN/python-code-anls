# `so-vits-svc\diffusion\unit2mel.py`

```py
# 导入所需的库
import os
import numpy as np
import torch
import torch.nn as nn
import yaml
from .diffusion import GaussianDiffusion
from .vocoder import Vocoder
from .wavenet import WaveNet

# 定义一个继承自字典的类，用于实现属性访问
class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   
    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

# 加载声码器模型
def load_model_vocoder(
        model_path,
        device='cpu',
        config_path = None
        ):
    # 如果未指定配置文件路径，则使用模型路径中的配置文件
    if config_path is None:
        config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    else:
        config_file = config_path

    # 读取配置文件
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # 加载声码器
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # 加载模型
    model = Unit2Mel(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                args.model.timesteps,
                args.model.k_step_max
                )
    
    # 打印加载模型的信息
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded diffusion model, sampler is {args.infer.method}, speedup: {args.infer.speedup} ')
    return model, vocoder, args

# 定义一个继承自 nn.Module 的类
class Unit2Mel(nn.Module):
    # 初始化方法，接受输入通道数、说话人数、是否使用音高增强、输出维度、层数、通道数、隐藏层维度、时间步长和最大步长参数
    def __init__(
            self,
            input_channel,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=20, 
            n_chans=384, 
            n_hidden=256,
            timesteps=1000,
            k_step_max=1000
            ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化输入单元的线性变换层
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        # 初始化音高的线性变换层
        self.f0_embed = nn.Linear(1, n_hidden)
        # 初始化音量的线性变换层
        self.volume_embed = nn.Linear(1, n_hidden)
        # 如果使用音高增强，则初始化音高增强的线性变换层，否则为None
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        # 初始化说话人数
        self.n_spk = n_spk
        # 如果说话人数不为None且大于1，则初始化说话人嵌入层
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        
        # 初始化时间步长，如果为None则默认为1000
        self.timesteps = timesteps if timesteps is not None else 1000
        # 初始化最大步长，如果为None或小于等于0或大于等于时间步长，则默认为时间步长
        self.k_step_max = k_step_max if k_step_max is not None and k_step_max>0 and k_step_max<self.timesteps else self.timesteps

        # 初始化隐藏层维度
        self.n_hidden = n_hidden
        # 初始化解码器，使用高斯扩散和WaveNet模型
        self.decoder = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden),timesteps=self.timesteps,k_step=self.k_step_max, out_dims=out_dims)
        # 初始化输入通道数
        self.input_channel = input_channel
    # 初始化说话人嵌入
    def init_spkembed(self, units, f0, volume, spk_id = None, spk_mix_dict = None, aug_shift = None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        # 计算单位嵌入、基频嵌入和音量嵌入的和
        x = self.unit_embed(units) + self.f0_embed((1+ f0 / 700).log()) + self.volume_embed(volume)
        # 如果存在多个说话人并且有混合字典
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                # 初始化混合说话人嵌入
                spk_embed_mix = torch.zeros((1,1,self.hidden_size))
                # 遍历混合字典，计算混合说话人嵌入
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    spk_embeddd = self.spk_embed(spk_id_torch)
                    self.speaker_map[k] = spk_embeddd
                    spk_embed_mix = spk_embed_mix + v * spk_embeddd
                x = x + spk_embed_mix
            else:
                # 添加单个说话人嵌入
                x = x + self.spk_embed(spk_id - 1)
        # 改变说话人嵌入的维度
        self.speaker_map = self.speaker_map.unsqueeze(0)
        # 分离说话人嵌入
        self.speaker_map = self.speaker_map.detach()
        # 转置张量
        return x.transpose(1, 2)

    # 初始化说话人混合
    def init_spkmix(self, n_spk):
        # 初始化说话人嵌入映射
        self.speaker_map = torch.zeros((n_spk,1,1,self.n_hidden))
        # 初始化 Hubert 隐藏层大小
        hubert_hidden_size = self.input_channel
        n_frames = 10
        # 随机初始化 Hubert 张量
        hubert = torch.randn((1, n_frames, hubert_hidden_size))
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spks = {}
        # 初始化混合字典
        for i in range(n_spk):
            spks.update({i:1.0/float(self.n_spk)})
        # 调用初始化说话人嵌入函数
        self.init_spkembed(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)
    # 定义一个前向传播函数，接受多个输入参数，并返回一个字典
    def forward(self, units, f0, volume, spk_id = None, spk_mix_dict = None, aug_shift = None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        
        # 如果模型处于推断状态且传入了gt_spec参数，且k_step大于最大扩散步数，则抛出异常
        if not self.training and gt_spec is not None and k_step>self.k_step_max:
            raise Exception("The shallow diffusion k_step is greater than the maximum diffusion k_step(k_step_max)!")
        
        # 如果模型处于推断状态且gt_spec参数为None且k_step_max不等于timesteps，则抛出异常
        if not self.training and gt_spec is None and self.k_step_max!=self.timesteps:
            raise Exception("This model can only be used for shallow diffusion and can not infer alone!")
        
        # 计算输入特征的嵌入表示
        x = self.unit_embed(units) + self.f0_embed((1+ f0 / 700).log()) + self.volume_embed(volume)
        
        # 如果说话人数量大于1，则进行处理
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                # 根据spk_mix_dict中的值对x进行加权处理
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch)
            else:
                if spk_id.shape[1] > 1:
                    # 对spk_id进行形状变换和计算
                    g = spk_id.reshape((spk_id.shape[0], spk_id.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
                    g = g * self.speaker_map  # [N, S, B, 1, H]
                    g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
                    g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
                    x = x + g
                else:
                    x = x + self.spk_embed(spk_id)
        
        # 如果存在aug_shift_embed和aug_shift参数，则进行处理
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5) 
        
        # 调用解码器进行解码
        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
    
        # 返回解码结果
        return x
```