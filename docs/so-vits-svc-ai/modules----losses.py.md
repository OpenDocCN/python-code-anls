# `so-vits-svc\modules\losses.py`

```py
# 导入 torch 库
import torch


# 计算特征损失
def feature_loss(fmap_r, fmap_g):
  # 初始化损失
  loss = 0
  # 遍历真实特征图和生成特征图
  for dr, dg in zip(fmap_r, fmap_g):
    # 遍历每个像素
    for rl, gl in zip(dr, dg):
      # 将真实特征图转换为浮点数并断开梯度
      rl = rl.float().detach()
      # 将生成特征图转换为浮点数
      gl = gl.float()
      # 计算绝对值平均损失并累加
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 


# 计算判别器损失
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  # 初始化损失
  loss = 0
  # 初始化真实损失和生成损失列表
  r_losses = []
  g_losses = []
  # 遍历真实输出和生成输出
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    # 将真实输出和生成输出转换为浮点数
    dr = dr.float()
    dg = dg.float()
    # 计算真实损失和生成损失
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    # 累加损失
    loss += (r_loss + g_loss)
    # 将真实损失和生成损失添加到列表中
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


# 计算生成器损失
def generator_loss(disc_outputs):
  # 初始化损失
  loss = 0
  # 初始化生成损失列表
  gen_losses = []
  # 遍历生成器输出
  for dg in disc_outputs:
    # 将生成器输出转换为浮点数
    dg = dg.float()
    # 计算生成损失并累加
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


# 计算 KL 散度损失
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
  # 计算 KL 散度损失
  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l
```