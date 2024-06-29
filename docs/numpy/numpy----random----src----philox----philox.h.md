# `.\numpy\numpy\random\src\philox\philox.h`

```
// 如果未定义 _RANDOMDGEN__PHILOX_H_，则定义该头文件
#ifndef _RANDOMDGEN__PHILOX_H_
#define _RANDOMDGEN__PHILOX_H_

// 包含 numpy 库中的公共头文件
#include "numpy/npy_common.h"
// 包含用于整数类型的头文件
#include <inttypes.h>

// 定义 PHILOX 缓冲区大小为 4
#define PHILOX_BUFFER_SIZE 4L

// 定义包含两个 uint64_t 类型变量的结构体
struct r123array2x64 {
  uint64_t v[2];
};
// 定义包含四个 uint64_t 类型变量的结构体
struct r123array4x64 {
  uint64_t v[4];
};

// 定义 philox4x64 轮数为 10
enum r123_enum_philox4x64 { philox4x64_rounds = 10 };
// 定义 philox4x64 的计数器结构体
typedef struct r123array4x64 philox4x64_ctr_t;
// 定义 philox4x64 的密钥结构体
typedef struct r123array2x64 philox4x64_key_t;
// 定义 philox4x64 的不可逆密钥结构体
typedef struct r123array2x64 philox4x64_ukey_t;

// 定义内联函数，用于增加 philox4x64 密钥的值
static inline struct r123array2x64
_philox4x64bumpkey(struct r123array2x64 key) {
  key.v[0] += (0x9E3779B97F4A7C15ULL);
  key.v[1] += (0xBB67AE8584CAA73BULL);
  return key;
}

// 如果支持 uint128_t 类型（如 GCC, clang, ICC），定义 mulhilo64 函数
#ifdef __SIZEOF_INT128__
static inline uint64_t mulhilo64(uint64_t a, uint64_t b, uint64_t *hip) {
  __uint128_t product = ((__uint128_t)a) * ((__uint128_t)b);
  *hip = product >> 64;
  return (uint64_t)product;
}
// 否则，根据不同平台和编译器定义 mulhilo64 函数
#else
#if defined(_WIN32) && !defined(__MINGW32__)
#include <intrin.h>
// Windows 平台下，根据不同架构选择不同的内联函数 _umul128
#if defined(_WIN64) && defined(_M_AMD64)
#pragma intrinsic(_umul128)
#elif defined(_WIN64) && defined(_M_ARM64)
#pragma intrinsic(__umulh)
static inline uint64_t _umul128(uint64_t a, uint64_t b, uint64_t *high) {
  *high = __umulh(a, b);
  return a * b;
}
#else
#pragma intrinsic(__emulu)
static inline uint64_t _umul128(uint64_t a, uint64_t b, uint64_t *high) {
  uint64_t a_lo, a_hi, b_lo, b_hi, a_x_b_hi, a_x_b_mid, a_x_b_lo, b_x_a_mid,
      carry_bit;
  a_lo = (uint32_t)a;
  a_hi = a >> 32;
  b_lo = (uint32_t)b;
  b_hi = b >> 32;

  a_x_b_hi = __emulu(a_hi, b_hi);
  a_x_b_mid = __emulu(a_hi, b_lo);
  b_x_a_mid = __emulu(b_hi, a_lo);
  a_x_b_lo = __emulu(a_lo, b_lo);

  carry_bit = ((uint64_t)(uint32_t)a_x_b_mid + (uint64_t)(uint32_t)b_x_a_mid +
               (a_x_b_lo >> 32)) >>
              32;

  *high = a_x_b_hi + (a_x_b_mid >> 32) + (b_x_a_mid >> 32) + carry_bit;

  return a_x_b_lo + ((a_x_b_mid + b_x_a_mid) << 32);
}
#endif
static inline uint64_t mulhilo64(uint64_t a, uint64_t b, uint64_t *hip) {
  return _umul128(a, b, hip);
}
#else
static inline uint64_t _umul128(uint64_t a, uint64_t b, uint64_t *high) {
  uint64_t a_lo, a_hi, b_lo, b_hi, a_x_b_hi, a_x_b_mid, a_x_b_lo, b_x_a_mid,
      carry_bit;
  a_lo = (uint32_t)a;
  a_hi = a >> 32;
  b_lo = (uint32_t)b;
  b_hi = b >> 32;

  a_x_b_hi = a_hi * b_hi;
  a_x_b_mid = a_hi * b_lo;
  b_x_a_mid = b_hi * a_lo;
  a_x_b_lo = a_lo * b_lo;

  carry_bit = ((uint64_t)(uint32_t)a_x_b_mid + (uint64_t)(uint32_t)b_x_a_mid +
               (a_x_b_lo >> 32)) >>
              32;

  *high = a_x_b_hi + (a_x_b_mid >> 32) + (b_x_a_mid >> 32) + carry_bit;

  return a_x_b_lo + ((a_x_b_mid + b_x_a_mid) << 32);
}
static inline uint64_t mulhilo64(uint64_t a, uint64_t b, uint64_t *hip) {
  return _umul128(a, b, hip);
}
#endif
#endif

// 定义内联函数，用于执行 philox4x64 轮函数
static inline struct r123array4x64 _philox4x64round(struct r123array4x64 ctr,
                                                    struct r123array2x64 key);
#endif
static inline struct r123array4x64 _philox4x64round(struct r123array4x64 ctr,
                                                    struct r123array2x64 key) {
  // 计算第一个轮的高低位乘积
  uint64_t hi0;
  uint64_t hi1;
  uint64_t lo0 = mulhilo64((0xD2E7470EE14C6C93ULL), ctr.v[0], &hi0);
  // 计算第三个轮的高低位乘积
  uint64_t lo1 = mulhilo64((0xCA5A826395121157ULL), ctr.v[2], &hi1);
  // 构造输出数组
  struct r123array4x64 out = {
      {hi1 ^ ctr.v[1] ^ key.v[0], lo1, hi0 ^ ctr.v[3] ^ key.v[1], lo0}};
  return out;
}

static inline philox4x64_key_t philox4x64keyinit(philox4x64_ukey_t uk) {
  return uk;
}

static inline philox4x64_ctr_t philox4x64_R(unsigned int R,
                                            philox4x64_ctr_t ctr,
                                            philox4x64_key_t key);

static inline philox4x64_ctr_t philox4x64_R(unsigned int R,
                                            philox4x64_ctr_t ctr,
                                            philox4x64_key_t key) {
  // 根据轮数 R 执行不同次数的轮函数调用
  if (R > 0) {
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 1) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 2) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 3) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 4) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 5) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 6) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 7) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 8) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 9) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 10) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 11) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 12) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 13) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 14) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 15) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  return ctr;
}

typedef struct s_philox_state {
  philox4x64_ctr_t *ctr;
  philox4x64_key_t *key;
  int buffer_pos;
  uint64_t buffer[PHILOX_BUFFER_SIZE];
  int has_uint32;
  uint32_t uinteger;
} philox_state;

static inline uint64_t philox_next(philox_state *state) {
  uint64_t out;
  int i;
  philox4x64_ctr_t ct;

  if (state->buffer_pos < PHILOX_BUFFER_SIZE) {
    // 如果缓冲区中有剩余值，直接返回该值并移动指针
    out = state->buffer[state->buffer_pos];
    state->buffer_pos++;
    return out;
  }
  // 生成 4 个新的 uint64_t 值
  state->ctr->v[0]++;
  // 处理进位
  if (state->ctr->v[0] == 0) {
    state->ctr->v[1]++;
    ```
    // 检查状态结构体中计数器的第二个元素是否为0
    if (state->ctr->v[1] == 0) {
      // 若为0，增加计数器的第三个元素
      state->ctr->v[2]++;
      // 若第三个元素溢出，增加计数器的第四个元素
      if (state->ctr->v[2] == 0) {
        state->ctr->v[3]++;
      }
    }
  }
  // 调用 philox4x64_R 函数执行 Philox 算法加密，得到一个结构体 ct
  ct = philox4x64_R(philox4x64_rounds, *state->ctr, *state->key);
  // 将 ct 结构体中的四个元素依次赋值给状态结构体的缓冲区
  for (i = 0; i < 4; i++) {
    state->buffer[i] = ct.v[i];
  }
  // 设置状态结构体的缓冲区位置为1
  state->buffer_pos = 1;
  // 返回状态结构体缓冲区的第一个元素
  return state->buffer[0];
}

static inline uint64_t philox_next64(philox_state *state) {
  return philox_next(state);
}



# 返回下一个 64 位的随机数
static inline uint64_t philox_next64(philox_state *state) {
  调用 philox_next 函数返回下一个随机数
  return philox_next(state);
}



static inline uint32_t philox_next32(philox_state *state) {
  uint64_t next;

  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = philox_next(state);

  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}



# 返回下一个 32 位的随机数
static inline uint32_t philox_next32(philox_state *state) {
  声明一个变量 next 用于存储下一个随机数
  if (state->has_uint32) {
    如果 state 已经有一个 32 位整数
    state->has_uint32 = 0;
    返回存储在 state->uinteger 中的整数
    return state->uinteger;
  }
  否则获取下一个随机数
  next = philox_next(state);

  设置 state->has_uint32 为 1，表示 state 中有一个 32 位整数
  state->has_uint32 = 1;
  将 next 的高 32 位存储到 state->uinteger 中
  state->uinteger = (uint32_t)(next >> 32);
  返回 next 的低 32 位
  return (uint32_t)(next & 0xffffffff);
}



extern void philox_jump(philox_state *state);

extern void philox_advance(uint64_t *step, philox_state *state);

#endif



# 声明 philox_jump 函数的外部接口
extern void philox_jump(philox_state *state);

# 声明 philox_advance 函数的外部接口，接受一个步长和状态作为参数
extern void philox_advance(uint64_t *step, philox_state *state);
```