# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\IFNet_m.py`

```py
# 从当前包中导入 refine 模块的所有内容
from .refine import *


# 定义反卷积层的构造函数
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    # 返回一个序列，包括一个反卷积层和一个 PReLU 激活层
    return nn.Sequential(
        # 定义反卷积层，输入通道数、输出通道数、卷积核大小、步幅和填充
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        # 定义 PReLU 激活函数，输出通道数
        nn.PReLU(out_planes),
    )


# 定义卷积层的构造函数
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    # 返回一个序列，包括一个卷积层和一个 PReLU 激活层
    return nn.Sequential(
        # 定义卷积层，输入通道数、输出通道数、卷积核大小、步幅、填充、扩张和偏置
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        # 定义 PReLU 激活函数，输出通道数
        nn.PReLU(out_planes),
    )


# 定义 IFBlock 类，继承自 nn.Module
class IFBlock(nn.Module):
    # 初始化方法，定义输入通道和常量 c 的值
    def __init__(self, in_planes, c=64):
        # 调用父类初始化方法
        super(IFBlock, self).__init__()
        # 定义第一个卷积模块，包含两层卷积
        self.conv0 = nn.Sequential(
            # 第一层卷积，输入通道为 in_planes，输出通道为 c // 2
            conv(in_planes, c // 2, 3, 2, 1),
            # 第二层卷积，输入通道为 c // 2，输出通道为 c
            conv(c // 2, c, 3, 2, 1),
        )
        # 定义卷积块，包含多层卷积
        self.convblock = nn.Sequential(
            # 逐层定义卷积，均为输入通道 c，输出通道 c
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        # 定义最后的反卷积层，输出通道为 5
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    # 前向传播方法，处理输入 x、流量 flow 和缩放比例 scale
    def forward(self, x, flow, scale):
        # 如果缩放比例不为 1，调整输入 x 的大小
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        # 如果 flow 不为 None，调整 flow 的大小
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            # 将 x 和 flow 在通道维度上进行拼接
            x = torch.cat((x, flow), 1)
        # 通过卷积模块 conv0 处理 x
        x = self.conv0(x)
        # 通过卷积块处理 x，并与原始 x 相加
        x = self.convblock(x) + x
        # 通过最后的反卷积层处理 x，得到 tmp
        tmp = self.lastconv(x)
        # 调整 tmp 的大小以匹配原始缩放
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        # 从 tmp 中提取 flow，并进行缩放
        flow = tmp[:, :4] * scale * 2
        # 提取掩码
        mask = tmp[:, 4:5]
        # 返回 flow 和掩码
        return flow, mask


# 定义 IFNet_m 类，继承自 nn.Module
class IFNet_m(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类初始化方法
        super(IFNet_m, self).__init__()
        # 定义多个 IFBlock 实例，输入通道数和常量 c 的值不同
        self.block0 = IFBlock(6 + 1, c=240)
        self.block1 = IFBlock(13 + 4 + 1, c=150)
        self.block2 = IFBlock(13 + 4 + 1, c=90)
        self.block_tea = IFBlock(16 + 4 + 1, c=90)
        # 定义上下文网络
        self.contextnet = Contextnet()
        # 定义 U-Net 网络
        self.unet = Unet()
    # 定义前向传播函数，接受输入x、缩放比例、时间步长和是否返回流的标志
        def forward(self, x, scale=[4, 2, 1], timestep=0.5, returnflow=False):
            # 计算时间步长，使用x的第一个通道并设置为默认值
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
            # 获取输入x的前三个通道作为img0
            img0 = x[:, :3]
            # 获取输入x的第四到第六个通道作为img1
            img1 = x[:, 3:6]
            # 获取输入x的其余部分作为gt，在推理时gt为None
            gt = x[:, 6:]  # In inference time, gt is None
            # 初始化流列表和合并结果列表
            flow_list = []
            merged = []
            mask_list = []
            # 将img0和img1赋值给扭曲后的图像
            warped_img0 = img0
            warped_img1 = img1
            # 初始化流和蒸馏损失
            flow = None
            loss_distill = 0
            # 定义包含多个网络模块的列表
            stu = [self.block0, self.block1, self.block2]
            # 对于每个模块，进行三次循环
            for i in range(3):
                # 如果已有流，则进行流和掩码的计算
                if flow != None:
                    flow_d, mask_d = stu[i](
                        # 拼接输入，包含img0、img1、时间步长、扭曲后的图像和掩码
                        torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask), 1), flow, scale=scale[i]
                    )
                    # 更新流和掩码
                    flow = flow + flow_d
                    mask = mask + mask_d
                else:
                    # 第一次计算流和掩码
                    flow, mask = stu[i](torch.cat((img0, img1, timestep), 1), None, scale=scale[i])
                # 将掩码经过sigmoid激活后加入掩码列表
                mask_list.append(torch.sigmoid(mask))
                # 将流加入流列表
                flow_list.append(flow)
                # 使用流对img0和img1进行扭曲
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                # 合并扭曲后的图像
                merged_student = (warped_img0, warped_img1)
                merged.append(merged_student)
            # 如果gt的通道数为3，则进行教师网络的计算
            if gt.shape[1] == 3:
                flow_d, mask_d = self.block_tea(
                    # 拼接输入，包括img0、img1、时间步长、扭曲后的图像、掩码和gt
                    torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask, gt), 1), flow, scale=1
                )
                # 更新教师网络的流
                flow_teacher = flow + flow_d
                # 扭曲教师网络的图像
                warped_img0_teacher = warp(img0, flow_teacher[:, :2])
                warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
                # 计算教师网络的掩码
                mask_teacher = torch.sigmoid(mask + mask_d)
                # 合并教师网络的结果
                merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
            else:
                # 如果没有gt，教师网络的流和合并结果为None
                flow_teacher = None
                merged_teacher = None
            # 对于每个模块，合并结果并计算损失
            for i in range(3):
                merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
                # 如果gt的通道数为3，计算蒸馏损失
                if gt.shape[1] == 3:
                    loss_mask = (
                        # 判断合并结果的绝对误差是否大于教师网络的误差加0.01
                        ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01)
                        .float()
                        .detach()
                    )
                    # 累加蒸馏损失
                    loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
            # 根据返回流的标志决定返回内容
            if returnflow:
                return flow
            else:
                # 使用上下文网络对图像进行处理
                c0 = self.contextnet(img0, flow[:, :2])
                c1 = self.contextnet(img1, flow[:, 2:4])
                # 使用U-Net生成最终结果
                tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
                # 调整结果范围
                res = tmp[:, :3] * 2 - 1
                # 更新合并结果
                merged[2] = torch.clamp(merged[2] + res, 0, 1)
            # 返回流列表、掩码列表、合并结果、教师流、教师合并结果和蒸馏损失
            return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill
```