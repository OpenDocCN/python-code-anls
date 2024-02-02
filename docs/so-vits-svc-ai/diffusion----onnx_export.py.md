# `so-vits-svc\diffusion\onnx_export.py`

```py
# 导入所需的库
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from diffusion_onnx import GaussianDiffusion

# 定义一个字典类，使其支持使用点号访问键值
class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   
    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

# 加载声码器模型
def load_model_vocoder(
        model_path,
        device='cpu'):
    # 从模型路径中获取配置文件路径
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    # 读取配置文件
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    # 将配置文件内容转换为DotDict对象
    args = DotDict(args)
    
    # 加载模型
    model = Unit2Mel(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                128,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                args.model.timesteps,
                args.model.k_step_max)
    
    # 打印加载模型的信息
    print(' [Loading] ' + model_path)
    # 加载模型的参数
    ckpt = torch.load(model_path, map_location=torch.device(device))
    # 将模型移动到指定设备
    model.to(device)
    # 加载模型的状态字典
    model.load_state_dict(ckpt['model'])
    # 设置模型为评估模式
    model.eval()
    # 返回加载的模型和配置参数
    return model, args

# 定义一个继承自nn.Module的类
class Unit2Mel(nn.Module):
    # 初始化函数，设置模型的输入通道数、说话人数量、是否使用音高增强、输出维度、层数、通道数、隐藏层维度、时间步长和最大步长
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
            k_step_max=1000):
        # 调用父类的初始化函数
        super().__init__()

        # 设置输入通道到隐藏层的线性映射
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        # 设置音高到隐藏层的线性映射
        self.f0_embed = nn.Linear(1, n_hidden)
        # 设置音量到隐藏层的线性映射
        self.volume_embed = nn.Linear(1, n_hidden)
        # 如果使用音高增强，则设置增强偏移到隐藏层的线性映射，否则为None
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        # 设置说话人数量
        self.n_spk = n_spk
        # 如果说话人数量不为None且大于1，则设置说话人嵌入
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)

        # 设置时间步长，如果为None则默认为1000
        self.timesteps = timesteps if timesteps is not None else 1000
        # 设置最大步长，如果为None或小于等于0或大于等于时间步长，则默认为时间步长
        self.k_step_max = k_step_max if k_step_max is not None and k_step_max>0 and k_step_max<self.timesteps else self.timesteps

        # 初始化高斯扩散解码器
        self.decoder = GaussianDiffusion(out_dims, n_layers, n_chans, n_hidden,self.timesteps,self.k_step_max)
        # 设置隐藏层维度
        self.hidden_size = n_hidden
        # 初始化说话人映射
        self.speaker_map = torch.zeros((self.n_spk,1,1,n_hidden))
    # 定义一个方法，用于进行前向传播计算
    def forward(self, units, mel2ph, f0, volume, g = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        # 在 units 的第二维度上进行填充，上方填充1个0，下方填充0个0
        decoder_inp = F.pad(units, [0, 0, 1, 0])
        # 在第二维度上复制 mel2ph，并扩展一个维度，使其与 units 的维度相同
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, units.shape[-1]])
        # 使用 mel2ph_ 从 decoder_inp 中进行索引，得到 units
        units = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        # 对 units、f0 和 volume 进行嵌入操作，并相加
        x = self.unit_embed(units) + self.f0_embed((1 + f0.unsqueeze(-1) / 700).log()) + self.volume_embed(volume.unsqueeze(-1))

        # 如果说说话者数量不为空且大于1
        if self.n_spk is not None and self.n_spk > 1:   # [N, S]  *  [S, B, 1, H]
            # 重新调整 g 的形状
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            # g 与 speaker_map 相乘
            g = g * self.speaker_map  # [N, S, B, 1, H]
            # 对 g 进行求和
            g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
            # 对 g 进行维度转换
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
            # 对 x 进行维度转换，并与 g 相加
            x = x.transpose(1, 2) + g
            # 返回 x
            return x
        else:
            # 对 x 进行维度转换
            return x.transpose(1, 2)
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
        # 如果存在多个说话人并且说话人混合字典不为空
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                # 初始化混合说话人嵌入
                spk_embed_mix = torch.zeros((1,1,self.hidden_size))
                # 遍历说话人混合字典，计算混合说话人嵌入
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    spk_embeddd = self.spk_embed(spk_id_torch)
                    self.speaker_map[k] = spk_embeddd
                    spk_embed_mix = spk_embed_mix + v * spk_embeddd
                # 更新输入特征
                x = x + spk_embed_mix
            else:
                # 更新输入特征
                x = x + self.spk_embed(spk_id - 1)
        # 扩展说话人映射
        self.speaker_map = self.speaker_map.unsqueeze(0)
        # 分离说话人映射
        self.speaker_map = self.speaker_map.detach()
        # 转置输入特征
        return x.transpose(1, 2)
    # 定义一个方法，用于导出模型到ONNX格式
    def OnnxExport(self, project_name=None, init_noise=None, export_encoder=True, export_denoise=True, export_pred=True, export_after=True):
        # 定义模型的隐藏层大小和帧数
        hubert_hidden_size = 768
        n_frames = 100
        # 生成随机的hubert输入
        hubert = torch.randn((1, n_frames, hubert_hidden_size))
        # 生成mel2ph张量
        mel2ph = torch.arange(end=n_frames).unsqueeze(0).long()
        # 生成随机的f0、volume张量
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spk_mix = []  # 初始化spk_mix列表
        spks = {}  # 初始化spks字典
        # 如果存在多个说话人并且说话人数量大于1
        if self.n_spk is not None and self.n_spk > 1:
            # 遍历说话人数量，计算spk_mix和更新spks字典
            for i in range(self.n_spk):
                spk_mix.append(1.0/float(self.n_spk))
                spks.update({i:1.0/float(self.n_spk)})
        # 将spk_mix列表转换为张量，并重复n_frames次
        spk_mix = torch.tensor(spk_mix)
        spk_mix = spk_mix.repeat(n_frames, 1)
        # 初始化说话人嵌入
        self.init_spkembed(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)
        # 调用模型的前向传播方法
        self.forward(hubert, mel2ph, f0, volume, spk_mix)
        # 如果需要导出编码器
        if export_encoder:
            # 使用torch.onnx.export方法导出编码器模型到ONNX格式
            torch.onnx.export(
                self,
                (hubert, mel2ph, f0, volume, spk_mix),
                f"{project_name}_encoder.onnx",
                input_names=["hubert", "mel2ph", "f0", "volume", "spk_mix"],
                output_names=["mel_pred"],
                dynamic_axes={
                    "hubert": [1],
                    "f0": [1],
                    "volume": [1],
                    "mel2ph": [1],
                    "spk_mix": [0],
                },
                opset_version=16
            )
        # 调用解码器的OnnxExport方法
        self.decoder.OnnxExport(project_name, init_noise=init_noise, export_denoise=export_denoise, export_pred=export_pred, export_after=export_after)
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 设置项目名称为"dddsp"
    project_name = "dddsp"
    # 设置模型路径为"dddsp/model_500000.pt"
    model_path = f'{project_name}/model_500000.pt'

    # 调用load_model_vocoder函数加载模型，将返回的模型和空对象赋值给model
    model, _ = load_model_vocoder(model_path)

    # 调用model对象的OnnxExport方法，导出Diffusion的各个部分（需要使用MoeSS/MoeVoiceStudio或者自己编写Pndm/Dpm采样）
    model.OnnxExport(project_name, export_encoder=True, export_denoise=True, export_pred=True, export_after=True)

    # 调用model对象的ExportOnnx方法，合并Diffusion导出（Encoder和Diffusion分开，直接将Encoder的结果和初始噪声输入Diffusion即可）
    # model.ExportOnnx(project_name)
```