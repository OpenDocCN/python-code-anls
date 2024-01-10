# `Bert-VITS2\losses.py`

```
# 导入 torch 库
import torch
# 导入 torchaudio 库
import torchaudio
# 从 transformers 库中导入 AutoModel 类
from transformers import AutoModel

# 定义特征损失函数，计算两个特征图之间的损失
def feature_loss(fmap_r, fmap_g):
    # 初始化损失值
    loss = 0
    # 遍历真实特征图和生成特征图
    for dr, dg in zip(fmap_r, fmap_g):
        # 遍历每个特征图的像素值
        for rl, gl in zip(dr, dg):
            # 将真实特征图转换为浮点数，并且断开梯度
            rl = rl.float().detach()
            # 将生成特征图转换为浮点数
            gl = gl.float()
            # 计算绝对值误差，并累加到损失值上
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2

# 定义判别器损失函数，计算真实输出和生成输出之间的损失
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    # 初始化损失值
    loss = 0
    # 初始化真实损失列表和生成损失列表
    r_losses = []
    g_losses = []
    # 遍历真实输出和生成输出
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # 将真实输出和生成输出转换为浮点数
        dr = dr.float()
        dg = dg.float()
        # 计算真实损失和生成损失，并累加到损失值上
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        # 将真实损失和生成损失添加到对应的列表中
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

# 定义生成器损失函数，计算生成器输出的损失
def generator_loss(disc_outputs):
    # 初始化损失值
    loss = 0
    # 初始化生成损失列表
    gen_losses = []
    # 遍历生成器输出
    for dg in disc_outputs:
        # 将生成器输出转换为浮点数
        dg = dg.float()
        # 计算生成器损失，并累加到损失值上
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

# 定义 KL 散度损失函数，计算两个分布之间的 KL 散度
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    # 将输入转换为浮点数
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    # 计算 KL 散度
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l

# 定义 WavLMLoss 类，用于计算语音模型的损失
class WavLMLoss(torch.nn.Module):
    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        # 加载预训练的语音模型
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        # 对模型的采样率进行重采样
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        # 将语音模型设置为评估模式，并且冻结参数
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False
    # 定义一个前向传播函数，接受输入的音频数据和重构的音频数据
    def forward(self, wav, y_rec):
        # 禁止梯度计算
        with torch.no_grad():
            # 对输入的音频数据进行重采样
            wav_16 = self.resample(wav)
            # 获取输入音频数据的嵌入表示
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
        # 对重构的音频数据进行重采样
        y_rec_16 = self.resample(y_rec)
        # 获取重构音频数据的嵌入表示
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16.squeeze(), output_hidden_states=True
        ).hidden_states

        # 初始化频谱损失
        floss = 0
        # 计算输入音频数据和重构音频数据的频谱损失
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        # 返回频谱损失的均值
        return floss.mean()

    # 定义一个生成器函数，接受重构的音频数据
    def generator(self, y_rec):
        # 对重构的音频数据进行重采样
        y_rec_16 = self.resample(y_rec)
        # 获取重构音频数据的嵌入表示
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states
        # 将嵌入表示进行处理，以便输入到生成器网络中
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        # 通过生成器网络计算生成的音频数据和重构音频数据之间的损失
        y_df_hat_g = self.wd(y_rec_embeddings)
        # 计算生成器损失
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        # 返回生成器损失
        return loss_gen
    # 定义一个鉴别器函数，接受音频数据和重构的音频数据作为输入
    def discriminator(self, wav, y_rec):
        # 禁止梯度计算
        with torch.no_grad():
            # 对输入音频进行重新采样为16kHz
            wav_16 = self.resample(wav)
            # 使用wavlm模型获取输入音频的嵌入表示
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            # 对重构的音频进行重新采样为16kHz
            y_rec_16 = self.resample(y_rec)
            # 使用wavlm模型获取重构音频的嵌入表示
            y_rec_embeddings = self.wavlm(
                input_values=y_rec_16, output_hidden_states=True
            ).hidden_states

            # 将输入音频的嵌入表示进行处理
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            # 将重构音频的嵌入表示进行处理
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        # 使用鉴别器模型对输入音频的嵌入表示进行判别
        y_d_rs = self.wd(y_embeddings)
        # 使用鉴别器模型对重构音频的嵌入表示进行判别
        y_d_gs = self.wd(y_rec_embeddings)

        # 计算真实音频的鉴别器损失
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        r_loss = torch.mean((1 - y_df_hat_r) ** 2)
        # 计算重构音频的鉴别器损失
        g_loss = torch.mean((y_df_hat_g) ** 2)

        # 计算总的鉴别器损失
        loss_disc_f = r_loss + g_loss

        # 返回平均鉴别器损失
        return loss_disc_f.mean()

    # 定义鉴别器前向传播函数，接受音频数据作为输入
    def discriminator_forward(self, wav):
        # 禁止梯度计算
        with torch.no_grad():
            # 对输入音频进行重新采样为16kHz
            wav_16 = self.resample(wav)
            # 使用wavlm模型获取输入音频的嵌入表示
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            # 将输入音频的嵌入表示进行处理
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        # 使用鉴别器模型对输入音频的嵌入表示进行判别
        y_d_rs = self.wd(y_embeddings)

        # 返回鉴别器对输入音频的判别结果
        return y_d_rs
```