# `.\pytorch\torch\csrc\jit\tensorexpr\cuda_random.h`

```py
#pragma once

namespace torch {
namespace jit {
namespace tensorexpr {

// 定义了一个字符串常量 philox_random_string，用于描述 Philox 类的实现
constexpr auto philox_random_string = R"(

class Philox {
public:
  // 构造函数，初始化 Philox 类，接受种子、子序列和偏移量作为参数
  __device__ inline Philox(unsigned long long seed,
                           unsigned long long subsequence,
                           unsigned long long offset) {
    // 将种子分别存储到 key 的低位和高位
    key.x = (unsigned int)seed;
    key.y = (unsigned int)(seed >> 32);
    // 初始化计数器 counter，并根据子序列设置其中的 z 和 w
    counter = make_uint4(0, 0, 0, 0);
    counter.z = (unsigned int)(subsequence);
    counter.w = (unsigned int)(subsequence >> 32);
    // 初始化 STATE 为 0，调用 incr_n 方法根据偏移量进行增加操作
    STATE = 0;
    incr_n(offset / 4);
  }

  // 重载操作符 ()，返回随机数
  __device__ inline unsigned long operator()() {
    // 根据 STATE 的值进行不同的处理
    if(STATE == 0) {
      // 备份 counter 和 key，并进行 10 轮单步计算
      uint4 counter_ = counter;
      uint2 key_ = key;
      for(int i = 0; i < 9; i++) {
        counter_ = single_round(counter_, key_);
        key_.x += (kPhilox10A); key_.y += (kPhilox10B);
      }
      // 计算最终的输出值
      output = single_round(counter_, key_);
      // 执行增加操作
      incr();
    }
    // 根据 STATE 的不同值返回不同的输出成员
    unsigned long ret;
    switch(STATE) {
      case 0: ret = output.x; break;
      case 1: ret = output.y; break;
      case 2: ret = output.z; break;
      case 3: ret = output.w; break;
    }
    // 更新 STATE 的值，确保下次调用会返回下一个成员
    STATE = (STATE + 1) % 4;
    return ret;
  }

private:
  // 定义 Philox 类私有成员
  uint4 counter;   // 计数器
  uint4 output;    // 输出结果
  uint2 key;       // 种子 key
  unsigned int STATE; // 状态

  // 内部方法，根据给定的 n 值增加计数器 counter 的值
  __device__ inline void incr_n(unsigned long long n) {
    unsigned int nlo = (unsigned int)(n);
    unsigned int nhi = (unsigned int)(n >> 32);
    counter.x += nlo;
    if (counter.x < nlo)
      nhi++;
    counter.y += nhi;
    if (nhi <= counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }

  // 内部方法，增加计数器 counter 的值
  __device__ inline void incr() {
    if (++counter.x)
      return;
    if (++counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }

  // 内部方法，用于实现 32 位乘法并返回高位结果
  __device__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                    unsigned int *result_high) {
    *result_high = __umulhi(a, b);
    return a*b;
  }

  // 内部方法，单步计算 Philox 算法
  __device__ inline uint4 single_round(uint4 ctr, uint2 key) {
    unsigned int hi0;
    unsigned int hi1;
    // 调用 mulhilo32 方法分别计算两次乘法结果
    unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
    unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);

    // 根据乘法结果和输入参数计算并返回结果
    uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
    return ret;
  }

  // Philox 算法中使用的常量
  static const unsigned long kPhilox10A = 0x9E3779B9;
  static const unsigned long kPhilox10B = 0xBB67AE85;
  static const unsigned long kPhiloxSA = 0xD2511F53;
  static const unsigned long kPhiloxSB = 0xCD9E8D57;
};

// 定义常量 M_RAN_INVM32，表示 2^(-32) 的倒数
#define M_RAN_INVM32 2.3283064e-10f

// 定义设备函数 Uint32ToFloat，将输入的无符号整数转换为浮点数并返回
__device__  __inline__ float Uint32ToFloat(unsigned int x) {
  return x * M_RAN_INVM32;
}

)";

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```