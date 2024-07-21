# `.\pytorch\aten\src\ATen\native\metal\mpscnn\MPSCNNClampOp.h`

```
#import <ATen/native/metal/mpscnn/MPSCNNOp.h>

# 导入 Metal 下的 ATen 库中 MPSCNNOp.h 文件，这是为了使用 Metal Performance Shaders (MPS) CNN 操作。


@interface MPSCNNClampOp : NSObject<MPSCNNShaderOp>

# 定义 MPSCNNClampOp 类，继承自 NSObject，并遵循 MPSCNNShaderOp 协议。这表明 MPSCNNClampOp 类是一个 Metal Performance Shaders CNN 着色器操作的实现。
```