# `.\pytorch\torch\csrc\api\include\torch\nn\options.h`

```py
// 一次性导入了一系列与神经网络模块相关的选项头文件

#pragma once

// 包含批归一化层的选项
#include <torch/nn/options/batchnorm.h>

// 包含卷积层的选项
#include <torch/nn/options/conv.h>

// 包含dropout层的选项
#include <torch/nn/options/dropout.h>

// 包含fold函数的选项
#include <torch/nn/options/fold.h>

// 包含线性层的选项
#include <torch/nn/options/linear.h>

// 包含损失函数的选项
#include <torch/nn/options/loss.h>

// 包含标准化操作的选项
#include <torch/nn/options/normalization.h>

// 包含填充操作的选项
#include <torch/nn/options/padding.h>

// 包含像素混洗操作的选项
#include <torch/nn/options/pixelshuffle.h>

// 包含池化层的选项
#include <torch/nn/options/pooling.h>

// 包含循环神经网络（RNN）模块的选项
#include <torch/nn/options/rnn.h>

// 包含变换器模块的选项
#include <torch/nn/options/transformer.h>

// 包含Transformer编码器模块的选项
#include <torch/nn/options/transformercoder.h>

// 包含Transformer层的选项
#include <torch/nn/options/transformerlayer.h>

// 包含上采样模块的选项
#include <torch/nn/options/upsampling.h>

// 包含视觉模块的选项
#include <torch/nn/options/vision.h>
```