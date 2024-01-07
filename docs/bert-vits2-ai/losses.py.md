# `Bert-VITS2\losses.py`

```

# 导入 torch 库
import torch
# 导入 torchaudio 库
import torchaudio
# 从 transformers 库中导入 AutoModel 类
from transformers import AutoModel

# 定义计算特征损失的函数
def feature_loss(fmap_r, fmap_g):
    # 初始化损失值
    loss = 0
    # 遍历真实特征图和生成特征图
    for dr, dg in zip(fmap_r, fmap_g):
        # 遍历每个特征图中的元素
        for rl, gl in zip(dr, dg):
            # 将真实特征图转换为浮点数并断开梯度
            rl = rl.float().detach()
            # 将生成特征图转换为浮点数
            gl = gl.float()
            # 计算绝对值误差并累加到损失值中
            loss += torch.mean(torch.abs(rl - gl))
    # 返回损失值乘以2
    return loss * 2

# 定义判别器损失函数
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
        # 计算真实损失和生成损失
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        # 将真实损失和生成损失累加到总损失中
        loss += r_loss + g_loss
        # 将真实损失和生成损失添加到对应的列表中
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    # 返回总损失和真实损失列表、生成损失列表
    return loss, r_losses, g_losses

# 定义生成器损失函数
def generator_loss(disc_outputs):
    # 初始化损失值
    loss = 0
    # 初始化生成损失列表
    gen_losses = []
    # 遍历生成器输出
    for dg in disc_outputs:
        # 将生成器输出转换为浮点数
        dg = dg.float()
        # 计算损失并累加到总损失中
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    # 返回总损失和生成损失列表
    return loss, gen_losses

# 定义 KL 散度损失函数
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

# 定义 WavLM 损失类
class WavLMLoss(torch.nn.Module):
    def __init__(self, model, wd, model_sr, slm_sr=16000):
        # 调用父类构造函数
        super(WavLMLoss, self).__init__()
        # 加载预训练的模型
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def forward(self, wav, y_rec):
        # 计算输入音频和重构音频的损失
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16.squeeze(), output_hidden_states=True
        ).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self, y_rec):
        # 计算生成器的损失
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        return loss_gen

    def discriminator(self, wav, y_rec):
        # 计算判别器的损失
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(
                input_values=y_rec_16, output_hidden_states=True
            ).hidden_states

            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)
        g_loss = torch.mean((y_df_hat_g) ** 2)

        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        # 计算判别器的前向传播
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs

```