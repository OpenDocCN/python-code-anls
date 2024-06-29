# `.\numpy\numpy\random\src\splitmix64\splitmix64.h`

```
# 包含 C 语言标准库 inttypes.h，提供整数类型定义
#include <inttypes.h>

# 定义结构体 s_splitmix64_state，用于保存 splitmix64 算法的状态
typedef struct s_splitmix64_state {
  uint64_t state;        # 64 位整数，保存 splitmix64 算法的状态
  int has_uint32;        # 表示是否有未使用的 uint32_t 类型整数
  uint32_t uinteger;     # 存储未使用的 uint32_t 类型整数
} splitmix64_state;

# 定义静态内联函数 splitmix64_next，生成下一个随机数
static inline uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (state[0] += 0x9e3779b97f4a7c15);   # 更新状态值
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;        # 混合位移操作
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;        # 混合位移操作
  return z ^ (z >> 31);                            # 最终随机数生成
}

# 定义静态内联函数 splitmix64_next64，生成下一个 64 位随机数
static inline uint64_t splitmix64_next64(splitmix64_state *state) {
  return splitmix64_next(&state->state);           # 调用 splitmix64_next 函数并返回结果
}

# 定义静态内联函数 splitmix64_next32，生成下一个 32 位随机数
static inline uint32_t splitmix64_next32(splitmix64_state *state) {
  uint64_t next;
  if (state->has_uint32) {                          # 如果存在未使用的 uint32_t 类型整数
    state->has_uint32 = 0;                          # 将状态标记为已使用
    return state->uinteger;                         # 直接返回未使用的整数
  }
  next = splitmix64_next64(state);                  # 否则生成新的 64 位随机数
  state->has_uint32 = 1;                            # 标记存在未使用的 uint32_t 类型整数
  state->uinteger = (uint32_t)(next >> 32);         # 将高 32 位作为 uint32_t 整数存储
  return (uint32_t)(next & 0xffffffff);             # 返回低 32 位作为 uint32_t 整数
}
```