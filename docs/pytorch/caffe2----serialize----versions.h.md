# `.\pytorch\caffe2\serialize\versions.h`

```
#pragma once
#include <cstdint>

namespace caffe2 {
namespace serialize {

// 最小支持的文件格式版本号，使用十六进制表示
constexpr uint64_t kMinSupportedFileFormatVersion = 0x1L;

// 最大支持的文件格式版本号，使用十六进制表示
constexpr uint64_t kMaxSupportedFileFormatVersion = 0xAL;

// 版本（即为什么版本号会增加？）

// 注意 [动态版本和 torch.jit.save vs. torch.save]
//
// 我们的版本方案有一个“生成的文件格式版本”，描述了如何读取存档。
// 存档中写入的版本至少是当前生成的文件格式版本，但如果包含某些符号，
// 则可以更大。我们称这些条件版本为“动态版本”，因为它们在运行时确定。
//
// 动态版本在操作符语义更新时非常有用。
// 当使用 torch.jit.save 时，我们希望这些语义被保留。
// 然而，如果在每次更改时都提升生成的文件格式版本，
// 旧版本的 PyTorch 就无法从新版本的 PyTorch 中读取简单的存档，比如单个张量。
// 因此，我们为这些更改分配动态版本，根据需要覆盖生成的文件格式版本。
// 例如，当 torch.div 的语义更改时，它被分配为动态版本 4，
// 而使用 torch.div 的 torch.jit.save 模块的存档也至少具有版本 4。
// 这样可以防止早期版本的 PyTorch 错误执行错误类型的除法。
// 不使用 torch.div 或其他具有动态版本的运算符的模块可以写入生成的文件格式版本，
// 这些程序将在早期版本的 PyTorch 上按预期运行。
//
// 虽然 torch.jit.save 尝试保留操作符语义，torch.save 则不尝试。
// torch.save 类似于 Python 的 pickling，因此如果跨 PyTorch 版本保存和加载
// 使用 torch.div 的函数，行为将不同。
// 从技术上讲，torch.save 忽略动态版本。

// 1. 初始版本
// 2. 删除了 op_version_set 版本号
// 3. 添加了类型标签以 pickle 序列化容器类型
// 4. （动态）停止使用 torch.div 进行整数除法
//    （一个有版本的符号保留了版本 1-3 的历史行为）
// 5. （动态）在给定布尔或整数填充值时，停止 torch.full 推断浮点类型
// 6. 将版本字符串写入 `./data/version` 而不是 `version`

// [2021年12月15日]
// 由于对文件格式版本的不同解释，将 kProducedFileFormatVersion 从 3 设置为 7。
// 每当引入新的升级器时，应增加此数字。
// 过去增加版本号的原因：
//    1. 在版本 4 修改了 aten::div
//    2. 在版本 5 修改了 aten::full
//    3. torch.package 使用了版本 6
//    4. 引入新的升级器设计，并将版本号设置为 7，标记此更改
// --------------------------------------------------
// 描述新操作符版本升级的原因列表：
// 1) [01/24/2022]
//     将版本号提升到8，以更新aten::linspace和aten::linspace.out，使其在未提供steps时报错。
//     (参见：https://github.com/pytorch/pytorch/issues/55951)
// 2) [01/30/2022]
//     将版本号提升到9，以更新aten::logspace和aten::logspace.out，使其在未提供steps时报错。
//     (参见：https://github.com/pytorch/pytorch/issues/55951)
// 3) [02/11/2022]
//     将版本号提升到10，以更新aten::gelu和aten::gelu.out，支持新的approximate参数。
//     (参见：https://github.com/pytorch/pytorch/pull/61439)
constexpr uint64_t kProducedFileFormatVersion = 0xAL;

// 写入包时的绝对最低版本号。这意味着从现在开始的每个包都将大于此数字。
constexpr uint64_t kMinProducedFileFormatVersion = 0x3L;

// 包含字节码时写入的版本号。必须大于或等于kProducedFileFormatVersion。
// 因为torchscript的更改可能会引入字节码更改。
// 如果增加kProducedFileFormatVersion，则kProducedBytecodeVersion也应该增加。
// 关系为：
// kMaxSupportedFileFormatVersion >= (most likely ==) kProducedBytecodeVersion
//   >= kProducedFileFormatVersion
// 如果格式更改是向前兼容的（仍然可被旧版本可读），我们不会增加版本号，以最小化破坏现有客户端的风险。
// TODO: 更好的方式是允许创建模型的调用者指定其客户端可以接受的最大版本。
// 版本：
//  0x1L: 初始版本
//  0x2L: （缺少注释）
//  0x3L: （缺少注释）
//  0x4L: （update）向函数元组添加模式。向前兼容更改。
//  0x5L: （update）更新字节码共享常量张量文件，仅序列化不在torchscript常量表中的额外张量。
//        同时更新张量存储模式以适应统一格式，张量存储的根键从{index}更新为{the_pointer_value_the_tensor.storage}，例如：
//        `140245072983168.storage`。向前兼容更改。
//  0x6L: 使用指定参数数量隐式操作符版本化。详细信息请参阅https://github.com/pytorch/pytorch/pull/56845的总结。
//  0x7L: 启用具有默认参数和输出参数的操作符支持。参见https://github.com/pytorch/pytorch/pull/63651的详细信息。
//  0x8L: 将提升的操作符作为指令输出。详细信息请参阅https://github.com/pytorch/pytorch/pull/71662。
//  0x9L: 将序列化格式从pickle更改为format。此版本用于迁移。v8 pickle和v9 flatbuffer是相同的。
//        有关更多详细信息，请参阅https://github.com/pytorch/pytorch/pull/75201的总结。
// 定义一个常量，表示生成的字节码版本号为 0x8L
constexpr uint64_t kProducedBytecodeVersion = 0x8L;

// 使用静态断言检查生成的字节码版本号是否不低于生成的文件格式版本号
// 如果不满足条件，会触发静态断言错误信息："kProducedBytecodeVersion must be higher or equal to kProducedFileFormatVersion."
static_assert(
    kProducedBytecodeVersion >= kProducedFileFormatVersion,
    "kProducedBytecodeVersion must be higher or equal to kProducedFileFormatVersion.");

// 引入 kMinSupportedBytecodeVersion 和 kMaxSupportedBytecodeVersion 用于限定字节码的向后/向前兼容性支持。
// 如果在加载器中，模型版本号处于 kMinSupportedBytecodeVersion 到 kMaxSupportedBytecodeVersion 的范围内，
// 则应该支持该模型版本。例如，我们可能会提供一个包装器来处理更新的操作符。
constexpr uint64_t kMinSupportedBytecodeVersion = 0x4L;
constexpr uint64_t kMaxSupportedBytecodeVersion = 0x9L;

} // namespace serialize
} // namespace caffe2
```