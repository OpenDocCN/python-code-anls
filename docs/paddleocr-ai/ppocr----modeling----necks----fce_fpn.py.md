# `.\PaddleOCR\ppocr\modeling\necks\fce_fpn.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码参考自：
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/ppdet/modeling/necks/fpn.py

# 导入所需的库
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import XavierUniform
from paddle.nn.initializer import Normal
from paddle.regularizer import L2Decay

# 定义模块的导出列表
__all__ = ['FCEFPN']

# 定义 ConvNormLayer 类，继承自 nn.Layer
class ConvNormLayer(nn.Layer):
    # 定义 ConvNormLayer 类，继承自 nn.Layer 类
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 norm_groups=32,
                 lr_scale=1.,
                 freeze_norm=False,
                 initializer=Normal(
                     mean=0., std=0.01)):
        # 调用父类的构造函数
        super(ConvNormLayer, self).__init__()
        # 断言判断 norm_type 是否在 ['bn', 'sync_bn', 'gn'] 中
        assert norm_type in ['bn', 'sync_bn', 'gn']
    
        # 初始化 bias_attr 为 False
        bias_attr = False
    
        # 创建卷积层对象
        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(
                initializer=initializer, learning_rate=1.),
            bias_attr=bias_attr)
    
        # 根据 freeze_norm 判断是否需要冻结 norm 层的学习率
        norm_lr = 0. if freeze_norm else 1.
        # 初始化 param_attr 和 bias_attr
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        # 根据 norm_type 创建不同类型的归一化层
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2D(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'sync_bn':
            self.norm = nn.SyncBatchNorm(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(
                num_groups=norm_groups,
                num_channels=ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr)
    
    # 定义前向传播函数
    def forward(self, inputs):
        # 经过卷积层
        out = self.conv(inputs)
        # 经过归一化层
        out = self.norm(out)
        return out
class FCEFPN(nn.Layer):
    """
    Feature Pyramid Network, see https://arxiv.org/abs/1612.03144
    Args:
        in_channels (list[int]): input channels of each level which can be 
            derived from the output shape of backbone by from_config
        out_channels (list[int]): output channel of each level
        spatial_scales (list[float]): the spatial scales between input feature
            maps and original input image which can be derived from the output 
            shape of backbone by from_config
        has_extra_convs (bool): whether to add extra conv to the last level.
            default False
        extra_stage (int): the number of extra stages added to the last level.
            default 1
        use_c5 (bool): Whether to use c5 as the input of extra stage, 
            otherwise p5 is used. default True
        norm_type (string|None): The normalization type in FPN module. If 
            norm_type is None, norm will not be used after conv and if 
            norm_type is string, bn, gn, sync_bn are available. default None
        norm_decay (float): weight decay for normalization layer weights.
            default 0.
        freeze_norm (bool): whether to freeze normalization layer.  
            default False
        relu_before_extra_convs (bool): whether to add relu before extra convs.
            default False
        
    """

    @classmethod
    def from_config(cls, cfg, input_shape):
        # 从配置和输入形状中获取输入通道数列表
        return {
            'in_channels': [i.channels for i in input_shape],
            # 从输入形状中获取空间尺度列表
            'spatial_scales': [1.0 / i.stride for i in input_shape],
        }
    # 前向传播函数，接收来自主干网络的特征
    def forward(self, body_feats):
        # 存储侧边特征
        laterals = []
        # 获取主干网络特征的层数
        num_levels = len(body_feats)

        # 对每一层主干网络特征进行侧边卷积操作
        for i in range(num_levels):
            laterals.append(self.lateral_convs[i](body_feats[i]))

        # 从最底层开始向上采样并融合特征
        for i in range(1, num_levels):
            lvl = num_levels - i
            upsample = F.interpolate(
                laterals[lvl],
                scale_factor=2.,
                mode='nearest', )
            laterals[lvl - 1] += upsample

        # 存储最终的特征金字塔网络输出
        fpn_output = []
        # 对每一层特征进行进一步卷积操作
        for lvl in range(num_levels):
            fpn_output.append(self.fpn_convs[lvl](laterals[lvl]))

        # 如果需要额外的阶段
        if self.extra_stage > 0:
            # 使用最大池化来获取更多的顶部级别输出（Faster R-CNN，Mask R-CNN）
            if not self.has_extra_convs:
                assert self.extra_stage == 1, 'extra_stage should be 1 if FPN has not extra convs'
                fpn_output.append(F.max_pool2d(fpn_output[-1], 1, stride=2))
            # 为RetinaNet（使用_c5）/FCOS（使用_p5）添加额外的卷积级别
            else:
                if self.use_c5:
                    extra_source = body_feats[-1]
                else:
                    extra_source = fpn_output[-1]
                fpn_output.append(self.fpn_convs[num_levels](extra_source))

                # 对额外的阶段进行进一步卷积操作
                for i in range(1, self.extra_stage):
                    if self.relu_before_extra_convs:
                        fpn_output.append(self.fpn_convs[num_levels + i](F.relu(
                            fpn_output[-1])))
                    else:
                        fpn_output.append(self.fpn_convs[num_levels + i](
                            fpn_output[-1]))
        # 返回特征金字塔网络输出
        return fpn_output
```