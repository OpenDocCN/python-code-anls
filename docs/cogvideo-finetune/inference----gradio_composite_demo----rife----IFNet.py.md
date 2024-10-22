# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\IFNet.py`

```py
# 从当前模块导入所有内容
from .refine import *


# 定义转置卷积函数，创建反卷积层
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    # 返回一个包含转置卷积和PReLU激活的顺序容器
    return nn.Sequential(
        # 创建转置卷积层，设置输入输出通道、卷积核大小、步幅和填充
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        # 添加PReLU激活层
        nn.PReLU(out_planes),
    )


# 定义卷积函数，创建卷积层
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    # 返回一个包含卷积层和PReLU激活的顺序容器
    return nn.Sequential(
        # 创建卷积层，设置输入输出通道、卷积核大小、步幅、填充、扩张和偏置
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        # 添加PReLU激活层
        nn.PReLU(out_planes),
    )


# 定义IFBlock类，继承自nn.Module
class IFBlock(nn.Module):
    # 初始化方法，设置输入通道和常量c
    def __init__(self, in_planes, c=64):
        # 调用父类构造函数
        super(IFBlock, self).__init__()
        # 定义初始卷积序列
        self.conv0 = nn.Sequential(
            # 创建第一个卷积层
            conv(in_planes, c // 2, 3, 2, 1),
            # 创建第二个卷积层
            conv(c // 2, c, 3, 2, 1),
        )
        # 定义卷积块，包含多个卷积层
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        # 定义最后的转置卷积层
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    # 前向传播方法，定义数据流动
    def forward(self, x, flow, scale):
        # 如果scale不等于1，则进行上采样
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        # 如果flow不为None，则进行上采样并与x拼接
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            x = torch.cat((x, flow), 1)
        # 经过初始卷积序列
        x = self.conv0(x)
        # 经过卷积块，并与输入相加实现残差连接
        x = self.convblock(x) + x
        # 经过最后的转置卷积层
        tmp = self.lastconv(x)
        # 对输出进行上采样
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        # 分离出flow和mask
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        # 返回flow和mask
        return flow, mask


# 定义IFNet类，继承自nn.Module
class IFNet(nn.Module):
    # 初始化方法，定义多个IFBlock
    def __init__(self):
        # 调用父类构造函数
        super(IFNet, self).__init__()
        # 创建第一个IFBlock，输入通道为6，c=240
        self.block0 = IFBlock(6, c=240)
        # 创建第二个IFBlock，输入通道为13+4，c=150
        self.block1 = IFBlock(13 + 4, c=150)
        # 创建第三个IFBlock，输入通道为13+4，c=90
        self.block2 = IFBlock(13 + 4, c=90)
        # 创建教师网络的IFBlock，输入通道为16+4，c=90
        self.block_tea = IFBlock(16 + 4, c=90)
        # 创建上下文网络
        self.contextnet = Contextnet()
        # 创建UNet网络
        self.unet = Unet()
    # 前向传播函数，接受输入图像和其他参数
        def forward(self, x, scale=[4, 2, 1], timestep=0.5):
            # 将输入图像分为三部分：前景图像、背景图像和真实图像
            img0 = x[:, :3]
            img1 = x[:, 3:6]
            gt = x[:, 6:]  # 在推理时，gt 为 None
            flow_list = []  # 用于存储流信息的列表
            merged = []  # 用于存储合并结果的列表
            mask_list = []  # 用于存储掩膜信息的列表
            warped_img0 = img0  # 初始化扭曲后的前景图像
            warped_img1 = img1  # 初始化扭曲后的背景图像
            flow = None  # 初始化流为 None
            loss_distill = 0  # 初始化蒸馏损失
            stu = [self.block0, self.block1, self.block2]  # 学生模型的块列表
            for i in range(3):  # 遍历三个模型块
                if flow != None:  # 如果流信息不为 None
                    # 合并图像和流信息，进行前向传播以获取流和掩膜
                    flow_d, mask_d = stu[i](
                        torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i]
                    )
                    flow = flow + flow_d  # 更新流信息
                    mask = mask + mask_d  # 更新掩膜信息
                else:  # 如果流为 None
                    # 仅使用图像进行前向传播，获取流和掩膜
                    flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
                mask_list.append(torch.sigmoid(mask))  # 应用sigmoid函数并存储掩膜
                flow_list.append(flow)  # 存储流信息
                # 根据流信息扭曲图像
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                merged_student = (warped_img0, warped_img1)  # 合并扭曲后的图像
                merged.append(merged_student)  # 存储合并结果
            if gt.shape[1] == 3:  # 如果真实图像的通道数为3
                # 进行教师模型的前向传播
                flow_d, mask_d = self.block_tea(
                    torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1
                )
                flow_teacher = flow + flow_d  # 更新教师流信息
                warped_img0_teacher = warp(img0, flow_teacher[:, :2])  # 扭曲教师图像
                warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])  # 扭曲教师图像
                mask_teacher = torch.sigmoid(mask + mask_d)  # 更新教师掩膜
                merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)  # 合并教师图像
            else:  # 如果真实图像不为3通道
                flow_teacher = None  # 教师流信息为 None
                merged_teacher = None  # 教师合并结果为 None
            for i in range(3):  # 遍历三个模型块
                merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])  # 根据掩膜合并图像
                if gt.shape[1] == 3:  # 如果真实图像的通道数为3
                    # 计算损失掩膜
                    loss_mask = (
                        ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01)
                        .float()
                        .detach()
                    )
                    # 累加蒸馏损失
                    loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
            c0 = self.contextnet(img0, flow[:, :2])  # 计算上下文信息
            c1 = self.contextnet(img1, flow[:, 2:4])  # 计算上下文信息
            # 使用 U-Net 进行图像处理
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            res = tmp[:, :3] * 2 - 1  # 处理 U-Net 输出结果
            merged[2] = torch.clamp(merged[2] + res, 0, 1)  # 限制合并结果在 [0, 1] 范围内
            # 返回流列表、掩膜列表、合并结果、教师流信息和教师合并结果，以及蒸馏损失
            return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill
```