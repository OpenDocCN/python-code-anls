# `.\pytorch\torch\csrc\jit\python\utf8_decoding_ignore.cpp`

```py
#include <torch/csrc/jit/python/utf8_decoding_ignore.h>

namespace torch::jit {

namespace {
thread_local bool kIgnore = false; // 定义一个线程局部变量 kIgnore，并初始化为 false
}

void setUTF8DecodingIgnore(bool o) { // 定义一个函数，用于设置 UTF-8 解码是否忽略错误
  kIgnore = o; // 将传入的参数 o 赋值给 kIgnore
}
bool getUTF8DecodingIgnore() { // 定义一个函数，用于获取当前 UTF-8 解码是否忽略错误的状态
  return kIgnore; // 返回当前 kIgnore 的值
}

} // namespace torch::jit
```