# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\IFNet_2R.py`

```py
# 从相对路径引入 refine_2R 模块的所有内容
from .refine_2R import *


# 定义反卷积层，输入和输出通道数、卷积核大小、步幅和填充
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    # 返回一个序列，包含反卷积层和 PReLU 激活函数
    return nn.Sequential(
        # 创建反卷积层，指定输入通道数、输出通道数、卷积核大小、步幅和填充
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        # 添加 PReLU 激活函数，输出通道数为 out_planes
        nn.PReLU(out_planes),
    )


# 定义卷积层，输入和输出通道数、卷积核大小、步幅、填充和扩张
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    # 返回一个序列，包含卷积层和 PReLU 激活函数
    return nn.Sequential(
        nn.Conv2d(
            # 输入通道数
            in_planes,
            # 输出通道数
            out_planes,
            # 卷积核大小
            kernel_size=kernel_size,
            # 步幅
            stride=stride,
            # 填充
            padding=padding,
            # 扩张
            dilation=dilation,
            # 启用偏置项
            bias=True,
        ),
        # 添加 PReLU 激活函数，输出通道数为 out_planes
        nn.PReLU(out_planes),
    )


# 定义 IFBlock 类，继承自 nn.Module
class IFBlock(nn.Module):
    # 初始化方法，定义输入通道数和常量 c
    def __init__(self, in_planes, c=64):
        # 调用父类构造函数
        super(IFBlock, self).__init__()
        # 定义第一个卷积序列
        self.conv0 = nn.Sequential(
            # 第一个卷积，输入通道为 in_planes，输出通道为 c // 2
            conv(in_planes, c // 2, 3, 1, 1),
            # 第二个卷积，输入通道为 c // 2，输出通道为 c
            conv(c // 2, c, 3, 2, 1),
        )
        # 定义主卷积块，由多个卷积层组成
        self.convblock = nn.Sequential(
            # 重复调用卷积函数，输入输出通道均为 c
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        # 定义最后的反卷积层，输入通道为 c，输出通道为 5
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    # 前向传播方法，接受输入 x、光流 flow 和缩放 scale
    def forward(self, x, flow, scale):
        # 如果缩放不为 1，则对 x 进行上采样
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        # 如果光流不为空，则对 flow 进行上采样并与 x 连接
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            # 将 x 和 flow 在通道维度上连接
            x = torch.cat((x, flow), 1)
        # 通过 conv0 处理 x
        x = self.conv0(x)
        # 通过 convblock 处理 x，并与原始 x 相加
        x = self.convblock(x) + x
        # 通过最后的反卷积层处理 x
        tmp = self.lastconv(x)
        # 对 tmp 进行上采样
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        # 从 tmp 中提取 flow，缩放回原始尺寸
        flow = tmp[:, :4] * scale
        # 提取 mask，形状为 (batch_size, 1, H, W)
        mask = tmp[:, 4:5]
        # 返回光流和掩码
        return flow, mask


# 定义 IFNet 类，继承自 nn.Module
class IFNet(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类构造函数
        super(IFNet, self).__init__()
        # 定义多个 IFBlock 实例，输入通道数和常量 c
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13 + 4, c=150)
        self.block2 = IFBlock(13 + 4, c=90)
        self.block_tea = IFBlock(16 + 4, c=90)
        # 创建上下文网络和 U-Net 实例
        self.contextnet = Contextnet()
        self.unet = Unet()
    # 定义前向传播函数，接受输入张量 x，缩放比例和时间步长
    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        # 从输入 x 中提取第一张图像的 RGB 通道
        img0 = x[:, :3]
        # 从输入 x 中提取第二张图像的 RGB 通道
        img1 = x[:, 3:6]
        # 从输入 x 中提取地面真值，推理时该值为 None
        gt = x[:, 6:]  # In inference time, gt is None
        # 初始化流列表
        flow_list = []
        # 初始化合并结果列表
        merged = []
        # 初始化掩膜列表
        mask_list = []
        # 初始化扭曲的第一张图像为原始图像
        warped_img0 = img0
        # 初始化扭曲的第二张图像为原始图像
        warped_img1 = img1
        # 初始化流为 None
        flow = None
        # 初始化蒸馏损失
        loss_distill = 0
        # 获取模型的不同块
        stu = [self.block0, self.block1, self.block2]
        # 循环处理三个块
        for i in range(3):
            # 如果流不为 None，则进行流的更新
            if flow != None:
                # 从当前块中计算流和掩膜
                flow_d, mask_d = stu[i](
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i]
                )
                # 更新流
                flow = flow + flow_d
                # 更新掩膜
                mask = mask + mask_d
            else:
                # 对于第一轮，流为 None，直接计算流和掩膜
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            # 将当前掩膜应用 sigmoid 函数，添加到掩膜列表
            mask_list.append(torch.sigmoid(mask))
            # 将当前流添加到流列表
            flow_list.append(flow)
            # 基于流扭曲第一张图像
            warped_img0 = warp(img0, flow[:, :2])
            # 基于流扭曲第二张图像
            warped_img1 = warp(img1, flow[:, 2:4])
            # 合并扭曲后的图像
            merged_student = (warped_img0, warped_img1)
            # 添加合并结果到合并列表
            merged.append(merged_student)
        # 如果地面真值的通道数为 3
        if gt.shape[1] == 3:
            # 从教师模型计算流和掩膜
            flow_d, mask_d = self.block_tea(
                torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1
            )
            # 更新教师流
            flow_teacher = flow + flow_d
            # 基于教师流扭曲第一张图像
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            # 基于教师流扭曲第二张图像
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            # 更新教师掩膜
            mask_teacher = torch.sigmoid(mask + mask_d)
            # 根据教师掩膜合并图像
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            # 如果没有地面真值，则教师流和合并教师图像为 None
            flow_teacher = None
            merged_teacher = None
        # 循环处理三个块
        for i in range(3):
            # 根据掩膜合并当前图像
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            # 如果地面真值的通道数为 3
            if gt.shape[1] == 3:
                # 计算损失掩膜，判断当前合并结果是否优于教师合并结果
                loss_mask = (
                    ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01)
                    .float()
                    .detach()
                )
                # 更新蒸馏损失
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        # 通过上下文网络计算第一张图像的上下文特征
        c0 = self.contextnet(img0, flow[:, :2])
        # 通过上下文网络计算第二张图像的上下文特征
        c1 = self.contextnet(img1, flow[:, 2:4])
        # 通过 U-Net 计算临时结果
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        # 处理临时结果，准备返回
        res = tmp[:, :3] * 2 - 1
        # 更新合并结果，确保在 [0, 1] 范围内
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        # 返回流列表、第二个掩膜、合并结果、教师流、教师合并结果和蒸馏损失
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill
```