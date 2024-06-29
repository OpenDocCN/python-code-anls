# `.\numpy\numpy\random\src\philox\philox.c`

```py
# 包含 "philox.h" 头文件，这里假设包含了 Philox 算法的定义和声明
#include "philox.h"

# 声明一个内联函数，返回一个 64 位的无符号整数，使用 Philox 状态作为参数
extern inline uint64_t philox_next64(philox_state *state);

# 声明一个内联函数，返回一个 32 位的无符号整数，使用 Philox 状态作为参数
extern inline uint32_t philox_next32(philox_state *state);

# 定义 Philox 状态跳跃函数
extern void philox_jump(philox_state *state) {
  # 使状态进一步，仿佛进行了 2^128 次抽样
  state->ctr->v[2]++;
  # 如果第三个元素达到最大值，增加第四个元素
  if (state->ctr->v[2] == 0) {
    state->ctr->v[3]++;
  }
}

# 定义 Philox 状态推进函数，接受一个步长数组和状态作为参数
extern void philox_advance(uint64_t *step, philox_state *state) {
  # 初始化循环和进位变量
  int i, carry = 0;
  uint64_t v_orig;
  # 遍历状态的四个元素
  for (i = 0; i < 4; i++) {
    # 如果有进位，增加当前元素的值并更新进位状态
    if (carry == 1) {
      state->ctr->v[i]++;
      carry = state->ctr->v[i] == 0 ? 1 : 0;
    }
    # 保存当前元素的原始值
    v_orig = state->ctr->v[i];
    # 增加当前元素的值和步长数组对应元素的值
    state->ctr->v[i] += step[i];
    # 如果增加后的值小于原始值并且当前无进位，设置进位标志
    if (state->ctr->v[i] < v_orig && carry == 0) {
      carry = 1;
    }
  }
}
```