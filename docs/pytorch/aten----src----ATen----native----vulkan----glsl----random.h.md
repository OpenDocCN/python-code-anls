# `.\pytorch\aten\src\ATen\native\vulkan\glsl\random.h`

```py
/*
 * Random utility functions
 */

// 定义常数 PI 为圆周率的值
#define PI 3.14159265358979323846264

// 实现 PCG 哈希函数，用于生成伪随机数
uint pcg_hash(uint v) {
  // 根据输入值 v 计算出新的状态值
  uint state = v * 747796405u + 2891336453u;
  // 利用状态值生成新的哈希值 word
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  // 返回经过处理的伪随机数
  return (word >> 22u) ^ word;
}

// 根据输入的四维整数向量 pos 生成一个 [0, 1) 范围内的随机浮点数
float rand2(ivec4 pos) {
  // 将四个方向上的哈希值相加得到种子 s
  uint s =
      pcg_hash(pos.x) + pcg_hash(pos.y) + pcg_hash(pos.z) + pcg_hash(pos.w);
  // 返回 [0, 1) 范围内的随机浮点数
  return fract(s / 1234567.0);
}

// 生成一个非零的随机浮点数，保证 v 不等于 0.0
float rand2_nonzero(ivec4 pos) {
  // 调用 rand2 函数生成随机数 v
  float v = rand2(pos);
  // 如果 v 等于 0.0，则循环生成新的随机数，直到 v 不为 0.0
  int offset = 0;
  while (v == 0.0) {
    offset++;
    v = rand2(ivec4(pos.x + offset, pos.y, pos.z, pos.w));
  }
  // 返回非零的随机数 v
  return v;
}

// 生成一个介于 from 和 to 之间的均匀分布随机数
float get_uniform(ivec4 pos, float from, float to) {
  // 调用 rand2 函数生成随机数 v
  float v = rand2(pos);
  // 将随机数 v 映射到 [from, to] 区间并返回
  return from + v * (to - from);
}

// 生成一个具有均值 mean 和标准差 std 的高斯分布随机数
float get_gaussrand(ivec4 pos, float mean, float std) {
  // 使用 Box-Muller 转换算法实现高斯分布随机数的生成
  float u1 = rand2_nonzero(pos);
  float u2 =
      rand2_nonzero(ivec4(pos.x + 10, pos.y + 20, pos.z + 30, pos.w + 40));
  float u3 =
      rand2_nonzero(ivec4(pos.x - 10, pos.y - 20, pos.z - 30, pos.w - 40));

  float mag = std * sqrt(-2.0 * log(u1));
  float v;
  // 根据第三个随机数 u3 决定选择哪一种生成方式
  if (u3 > 0.5)
    v = mag * cos(2.0 * PI * u2) + mean;
  else
    v = mag * sin(2.0 * PI * u2) + mean;
  // 返回生成的高斯分布随机数 v
  return v;
}
```