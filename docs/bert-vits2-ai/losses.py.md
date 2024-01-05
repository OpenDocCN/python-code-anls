# `d:/src/tocomm/Bert-VITS2\losses.py`

```
import torch  # 导入 PyTorch 库
import torchaudio  # 导入 PyTorch 音频处理库
from transformers import AutoModel  # 从 transformers 库中导入 AutoModel 类


def feature_loss(fmap_r, fmap_g):
    loss = 0  # 初始化损失值
    for dr, dg in zip(fmap_r, fmap_g):  # 遍历两个特征图列表
        for rl, gl in zip(dr, dg):  # 遍历每个特征图中的元素
            rl = rl.float().detach()  # 将特征图转换为浮点数并分离计算图
            gl = gl.float()  # 将特征图转换为浮点数
            loss += torch.mean(torch.abs(rl - gl))  # 计算特征图之间的绝对值差的平均值并累加到损失值中

    return loss * 2  # 返回损失值乘以2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0  # 初始化损失值
    r_losses = []  # 初始化真实输出的损失列表
    g_losses = []  # 初始化生成输出的损失列表
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # 将disc_real_outputs和disc_generated_outputs中的元素转换为浮点数
        dr = dr.float()
        dg = dg.float()
        # 计算真实数据的损失
        r_loss = torch.mean((1 - dr) ** 2)
        # 计算生成数据的损失
        g_loss = torch.mean(dg**2)
        # 将真实数据损失和生成数据损失相加得到总损失
        loss += r_loss + g_loss
        # 将真实数据损失和生成数据损失添加到对应的列表中
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    # 返回总损失和真实数据损失、生成数据损失的列表
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    # 初始化损失为0
    loss = 0
    # 初始化生成器损失的列表
    gen_losses = []
    # 遍历disc_outputs中的元素
    for dg in disc_outputs:
        # 将元素转换为浮点数
        dg = dg.float()
        # 计算生成数据的损失
        l = torch.mean((1 - dg) ** 2)
        # 将生成数据的损失添加到列表中
        gen_losses.append(l)
        # 将生成数据的损失加到总损失中
        loss += l
    return loss, gen_losses
```
这行代码返回变量loss和gen_losses的值。

```
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
```
这行代码定义了一个名为kl_loss的函数，它有5个参数z_p, logs_q, m_p, logs_p, z_mask。

```
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
```
这是一个多行注释，用来说明函数的参数z_p, logs_q, m_p, logs_p的形状。

```
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()
```
这几行代码将参数z_p, logs_q, m_p, logs_p, z_mask转换为浮点数类型。

```
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
```
这几行代码计算KL散度（Kullback-Leibler divergence）的值，并返回结果。KL散度是用来衡量两个概率分布之间的差异的指标。
class WavLMLoss(torch.nn.Module):
    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        # 从预训练模型中加载自动模型
        self.wavlm = AutoModel.from_pretrained(model)
        # 设置权重衰减
        self.wd = wd
        # 对输入的音频进行重采样
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        # 将wavlm模型设置为评估模式
        self.wavlm.eval()
        # 冻结wavlm模型的参数，不进行梯度更新
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def forward(self, wav, y_rec):
        # 在不计算梯度的情况下进行前向传播
        with torch.no_grad():
            # 对输入的音频进行重采样
            wav_16 = self.resample(wav)
            # 获取输入音频的嵌入表示
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
        # 对目标音频进行重采样
        y_rec_16 = self.resample(y_rec)
        # 获取目标音频的嵌入表示
        y_rec_embeddings = self.wavlm(
        input_values=y_rec_16.squeeze(), output_hidden_states=True
    ).hidden_states
```
这行代码是调用一个模型的方法，传入y_rec_16并调用squeeze()方法，然后设置output_hidden_states为True，返回模型的hidden_states。

```
    floss = 0
    for er, eg in zip(wav_embeddings, y_rec_embeddings):
        floss += torch.mean(torch.abs(er - eg))
```
这段代码是计算两个嵌入向量之间的损失。使用zip函数将wav_embeddings和y_rec_embeddings进行逐个配对，然后计算它们的绝对差值的平均值，并将结果累加到floss中。

```
    return floss.mean()
```
这行代码是返回损失的平均值作为函数的输出。

```
def generator(self, y_rec):
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
```
这段代码是一个生成器函数，它接受y_rec作为输入。首先对y_rec进行重采样，然后使用wavlm模型获取其嵌入向量。接着对嵌入向量进行一系列的操作，最终通过wd模型得到y_df_hat_g。
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)  # 计算生成器的损失，使用均方误差作为损失函数

        return loss_gen  # 返回生成器的损失值

    def discriminator(self, wav, y_rec):
        with torch.no_grad():  # 在这个上下文中，禁用梯度计算
            wav_16 = self.resample(wav)  # 对输入的音频进行重采样
            wav_embeddings = self.wavlm(  # 使用wav2vec模型提取音频的嵌入表示
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_rec_16 = self.resample(y_rec)  # 对重构的音频进行重采样
            y_rec_embeddings = self.wavlm(  # 使用wav2vec模型提取重构音频的嵌入表示
                input_values=y_rec_16, output_hidden_states=True
            ).hidden_states

            y_embeddings = (  # 将音频的嵌入表示进行处理，以便用于鉴别器的输入
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)  # 将y_rec_embeddings列表中的张量按照指定维度进行堆叠
                .transpose(-1, -2)  # 对堆叠后的张量进行维度转置
                .flatten(start_dim=1, end_dim=2)  # 对转置后的张量进行扁平化处理
            )

        y_d_rs = self.wd(y_embeddings)  # 使用self.wd对y_embeddings进行处理得到y_d_rs
        y_d_gs = self.wd(y_rec_embeddings)  # 使用self.wd对y_rec_embeddings进行处理得到y_d_gs

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs  # 将y_d_rs和y_d_gs分别赋值给y_df_hat_r和y_df_hat_g

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)  # 计算r_loss
        g_loss = torch.mean((y_df_hat_g) ** 2)  # 计算g_loss

        loss_disc_f = r_loss + g_loss  # 计算loss_disc_f

        return loss_disc_f.mean()  # 返回loss_disc_f的均值

    def discriminator_forward(self, wav):  # 定义discriminator_forward方法，接受wav作为输入参数
        with torch.no_grad():  # 使用torch.no_grad()上下文管理器，禁用梯度计算
            # 对音频进行重采样
            wav_16 = self.resample(wav)
            # 使用wav2vec模型对重采样后的音频进行编码，返回隐藏状态
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            # 将隐藏状态堆叠起来，并进行维度转换和展平操作
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        # 使用线性层对编码后的音频进行降维
        y_d_rs = self.wd(y_embeddings)

        # 返回降维后的音频编码
        return y_d_rs
```