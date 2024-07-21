# `.\pytorch\torch\csrc\api\include\torch\nn\modules.h`

```
// 一次性导入所有常用的 Torch 库中的模块

// Common
#include <torch/nn/modules/common.h>

// Containers
#include <torch/nn/modules/container/any.h>           // 引入容器模块中的 any.h
#include <torch/nn/modules/container/functional.h>    // 引入容器模块中的 functional.h
#include <torch/nn/modules/container/moduledict.h>    // 引入容器模块中的 moduledict.h
#include <torch/nn/modules/container/modulelist.h>    // 引入容器模块中的 modulelist.h
#include <torch/nn/modules/container/named_any.h>     // 引入容器模块中的 named_any.h
#include <torch/nn/modules/container/parameterdict.h> // 引入容器模块中的 parameterdict.h
#include <torch/nn/modules/container/parameterlist.h> // 引入容器模块中的 parameterlist.h
#include <torch/nn/modules/container/sequential.h>    // 引入容器模块中的 sequential.h

// Layers
#include <torch/nn/modules/activation.h>              // 引入神经网络层中的 activation.h
#include <torch/nn/modules/adaptive.h>                // 引入神经网络层中的 adaptive.h
#include <torch/nn/modules/batchnorm.h>               // 引入神经网络层中的 batchnorm.h
#include <torch/nn/modules/conv.h>                    // 引入神经网络层中的 conv.h
#include <torch/nn/modules/distance.h>                // 引入神经网络层中的 distance.h
#include <torch/nn/modules/dropout.h>                 // 引入神经网络层中的 dropout.h
#include <torch/nn/modules/embedding.h>               // 引入神经网络层中的 embedding.h
#include <torch/nn/modules/fold.h>                    // 引入神经网络层中的 fold.h
#include <torch/nn/modules/instancenorm.h>            // 引入神经网络层中的 instancenorm.h
#include <torch/nn/modules/linear.h>                  // 引入神经网络层中的 linear.h
#include <torch/nn/modules/loss.h>                    // 引入神经网络层中的 loss.h
#include <torch/nn/modules/normalization.h>           // 引入神经网络层中的 normalization.h
#include <torch/nn/modules/padding.h>                 // 引入神经网络层中的 padding.h
#include <torch/nn/modules/pixelshuffle.h>            // 引入神经网络层中的 pixelshuffle.h
#include <torch/nn/modules/pooling.h>                 // 引入神经网络层中的 pooling.h
#include <torch/nn/modules/rnn.h>                     // 引入神经网络层中的 rnn.h
#include <torch/nn/modules/transformer.h>             // 引入神经网络层中的 transformer.h
#include <torch/nn/modules/transformercoder.h>        // 引入神经网络层中的 transformercoder.h
#include <torch/nn/modules/transformerlayer.h>        // 引入神经网络层中的 transformerlayer.h
#include <torch/nn/modules/upsampling.h>              // 引入神经网络层中的 upsampling.h
```