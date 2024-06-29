# `.\numpy\numpy\random\src\mt19937\randomkit.c`

```
/* Random kit 1.3 */

/* static char const rcsid[] =
  "@(#) $Jeannot: randomkit.c,v 1.28 2005/07/21 22:14:09 js Exp $"; */

#ifdef _WIN32
/*
 * Windows
 * XXX: we have to use this ugly defined(__GNUC__) because it is not easy to
 * detect the compiler used in distutils itself
 */
#if (defined(__GNUC__) && defined(NPY_NEEDS_MINGW_TIME_WORKAROUND))

/*
 * FIXME: ideally, we should set this to the real version of MSVCRT. We need
 * something higher than 0x601 to enable _ftime64 and co
 */
#define __MSVCRT_VERSION__ 0x0700
#include <sys/timeb.h>
#include <time.h>

/*
 * mingw msvcr lib import wrongly export _ftime, which does not exist in the
 * actual msvc runtime for version >= 8; we make it an alias to _ftime64, which
 * is available in those versions of the runtime
 */
#define _FTIME(x) _ftime64((x))
#else
#include <sys/timeb.h>
#include <time.h>

#define _FTIME(x) _ftime((x))
#endif

#ifndef RK_NO_WINCRYPT
/* Windows crypto */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0400
#endif
#include <wincrypt.h>
#include <windows.h>

#endif

/*
 * Do not move this include. randomkit.h must be included
 * after windows timeb.h is included.
 */
#include "randomkit.h"

#else
/* Unix */
#include "randomkit.h"
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#endif

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef RK_DEV_URANDOM
#define RK_DEV_URANDOM "/dev/urandom"
#endif

#ifndef RK_DEV_RANDOM
#define RK_DEV_RANDOM "/dev/random"
#endif

char *rk_strerror[RK_ERR_MAX] = {"no error", "random device unvavailable"};

/* static functions */
/* 
 * 哈希函数，使用Thomas Wang的32位整数哈希算法
 */
static unsigned long rk_hash(unsigned long key);

/*
 * 初始化随机数发生器状态，使用Knuth的PRNG算法
 */
void rk_seed(unsigned long seed, rk_state *state) {
  int pos;
  seed &= 0xffffffffUL;

  /* Knuth's PRNG as used in the Mersenne Twister reference implementation */
  for (pos = 0; pos < RK_STATE_LEN; pos++) {
    state->key[pos] = seed;
    seed = (1812433253UL * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffUL;
  }
  state->pos = RK_STATE_LEN;
  state->gauss = 0;
  state->has_gauss = 0;
  state->has_binomial = 0;
}

/* 
 * Thomas Wang的32位整数哈希函数
 */
unsigned long rk_hash(unsigned long key) {
  key += ~(key << 15);
  key ^= (key >> 10);
  key += (key << 3);
  key ^= (key >> 6);
  key += ~(key << 11);
  key ^= (key >> 16);
  return key;
}

/*
 * 从系统随机源填充种子数据，确保生成的种子不为零
 */
rk_error rk_randomseed(rk_state *state) {
#ifndef _WIN32
  struct timeval tv;
#else
  struct _timeb tv;
#endif
  int i;

  if (rk_devfill(state->key, sizeof(state->key), 0) == RK_NOERR) {
    /* ensures non-zero key */
    state->key[0] |= 0x80000000UL;
    state->pos = RK_STATE_LEN;
    state->gauss = 0;
    state->has_gauss = 0;
    state->has_binomial = 0;

    for (i = 0; i < 624; i++) {
      state->key[i] &= 0xffffffffUL;
    }
    return RK_NOERR;
  }

#ifndef _WIN32
  gettimeofday(&tv, NULL);
  rk_seed(rk_hash(getpid()) ^ rk_hash(tv.tv_sec) ^ rk_hash(tv.tv_usec) ^
              rk_hash(clock()),
          state);
#else
  _FTIME(&tv);  // 获取当前时间到结构体 tv 中
  rk_seed(rk_hash(tv.time) ^ rk_hash(tv.millitm) ^ rk_hash(clock()), state);  // 使用当前时间、毫秒部分和系统时钟的哈希值作为种子初始化随机数发生器
#endif

  return RK_ENODEV;  // 返回设备不存在错误码
}

/* Magic Mersenne Twister constants */
#define N 624  // MT 算法中使用的数组长度
#define M 397   // MT 算法中用于生成随机数的参数
#define MATRIX_A 0x9908b0dfUL  // MT 算法中的矩阵 A 常数
#define UPPER_MASK 0x80000000UL  // MT 算法中的掩码，用于获取上半部分位
#define LOWER_MASK 0x7fffffffUL  // MT 算法中的掩码，用于获取下半部分位

/*
 * Slightly optimised reference implementation of the Mersenne Twister
 * Note that regardless of the precision of long, only 32 bit random
 * integers are produced
 */
unsigned long rk_random(rk_state *state) {
  unsigned long y;

  if (state->pos == RK_STATE_LEN) {
    int i;

    for (i = 0; i < N - M; i++) {
      y = (state->key[i] & UPPER_MASK) | (state->key[i + 1] & LOWER_MASK);
      state->key[i] = state->key[i + M] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);
    }
    for (; i < N - 1; i++) {
      y = (state->key[i] & UPPER_MASK) | (state->key[i + 1] & LOWER_MASK);
      state->key[i] =
          state->key[i + (M - N)] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);
    }
    y = (state->key[N - 1] & UPPER_MASK) | (state->key[0] & LOWER_MASK);
    state->key[N - 1] = state->key[M - 1] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);

    state->pos = 0;
  }
  y = state->key[state->pos++];

  /* Tempering */
  y ^= (y >> 11);  // 随机数的一种混淆操作
  y ^= (y << 7) & 0x9d2c5680UL;  // 随机数的一种混淆操作
  y ^= (y << 15) & 0xefc60000UL;  // 随机数的一种混淆操作
  y ^= (y >> 18);  // 随机数的一种混淆操作

  return y;  // 返回生成的随机数
}

/*
 * Returns an unsigned 64 bit random integer.
 */
static inline npy_uint64 rk_uint64(rk_state *state) {
  npy_uint64 upper = (npy_uint64)rk_random(state) << 32;  // 生成一个 64 位的随机数，取高 32 位
  npy_uint64 lower = (npy_uint64)rk_random(state);  // 生成一个 64 位的随机数，取低 32 位
  return upper | lower;  // 合并高位和低位得到完整的 64 位随机数
}

/*
 * Returns an unsigned 32 bit random integer.
 */
static inline npy_uint32 rk_uint32(rk_state *state) {
  return (npy_uint32)rk_random(state);  // 生成一个 32 位的随机数
}

/*
 * Fills an array with cnt random npy_uint64 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void rk_random_uint64(npy_uint64 off, npy_uint64 rng, npy_intp cnt,
                      npy_uint64 *out, rk_state *state) {
  npy_uint64 val, mask = rng;
  npy_intp i;

  if (rng == 0) {
    for (i = 0; i < cnt; i++) {
      out[i] = off;
    }
    return;
  }

  /* Smallest bit mask >= max */
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;

  for (i = 0; i < cnt; i++) {
    if (rng <= 0xffffffffUL) {
      while ((val = (rk_uint32(state) & mask)) > rng)
        ;
    } else {
      while ((val = (rk_uint64(state) & mask)) > rng)
        ;
    }
    out[i] = off + val;  // 将生成的随机数放入数组中
  }
}

/*
 * Fills an array with cnt random npy_uint32 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void rk_random_uint32(npy_uint32 off, npy_uint32 rng, npy_intp cnt,
                      npy_uint32 *out, rk_state *state) {
  npy_uint32 val, mask = rng;
  npy_intp i;

  if (rng == 0) {
    for (i = 0; i < cnt; i++) {
      out[i] = off;
    }
    return;
  }
    return;
  }

  /* Smallest bit mask >= max */
  // 初始化掩码为最小的大于等于 max 的 2 的幂次掩码
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;

  // 对于每个要生成的随机数
  for (i = 0; i < cnt; i++) {
    // 生成在指定范围内的随机数，确保其不大于 rng
    while ((val = (rk_uint32(state) & mask)) > rng)
      ;  // 循环直到生成的随机数满足要求
    // 将生成的随机数加上偏移 off，存入输出数组
    out[i] = off + val;
  }
/*
 * Fills an array with cnt random npy_uint16 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void rk_random_uint16(npy_uint16 off, npy_uint16 rng, npy_intp cnt,
                      npy_uint16 *out, rk_state *state) {
  npy_uint16 val, mask = rng;
  npy_intp i;
  npy_uint32 buf;
  int bcnt = 0;

  if (rng == 0) {
    for (i = 0; i < cnt; i++) {
      out[i] = off;
    }
    return;
  }

  /* Smallest bit mask >= max */
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;

  for (i = 0; i < cnt; i++) {
    do {
      if (!bcnt) {
        buf = rk_uint32(state);  // 从状态结构中获取一个32位随机数
        bcnt = 1;  // 设置位计数器为1
      } else {
        buf >>= 16;  // 右移16位，准备获取下一个16位的随机数
        bcnt--;
      }
      val = (npy_uint16)buf & mask;  // 将buf与掩码进行按位与运算，获取符合范围的随机数
    } while (val > rng);  // 如果随机数超出范围则重新生成
    out[i] = off + val;  // 将生成的随机数加上偏移量存入数组
  }
}

/*
 * Fills an array with cnt random npy_uint8 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void rk_random_uint8(npy_uint8 off, npy_uint8 rng, npy_intp cnt, npy_uint8 *out,
                     rk_state *state) {
  npy_uint8 val, mask = rng;
  npy_intp i;
  npy_uint32 buf;
  int bcnt = 0;

  if (rng == 0) {
    for (i = 0; i < cnt; i++) {
      out[i] = off;
    }
    return;
  }

  /* Smallest bit mask >= max */
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;

  for (i = 0; i < cnt; i++) {
    do {
      if (!bcnt) {
        buf = rk_uint32(state);  // 从状态结构中获取一个32位随机数
        bcnt = 3;  // 设置位计数器为3
      } else {
        buf >>= 8;  // 右移8位，准备获取下一个8位的随机数
        bcnt--;
      }
      val = (npy_uint8)buf & mask;  // 将buf与掩码进行按位与运算，获取符合范围的随机数
    } while (val > rng);  // 如果随机数超出范围则重新生成
    out[i] = off + val;  // 将生成的随机数加上偏移量存入数组
  }
}

/*
 * Fills an array with cnt random npy_bool between off and off + rng
 * inclusive.
 */
void rk_random_bool(npy_bool off, npy_bool rng, npy_intp cnt, npy_bool *out,
                    rk_state *state) {
  npy_intp i;
  npy_uint32 buf;
  int bcnt = 0;

  if (rng == 0) {
    for (i = 0; i < cnt; i++) {
      out[i] = off;
    }
    return;
  }

  /* If we reach here rng and mask are one and off is zero */
  assert(rng == 1 && off == 0);  // 断言rng为1且off为0
  for (i = 0; i < cnt; i++) {
    if (!bcnt) {
      buf = rk_uint32(state);  // 从状态结构中获取一个32位随机数
      bcnt = 31;  // 设置位计数器为31
    } else {
      buf >>= 1;  // 右移1位，准备获取下一个1位的随机数
      bcnt--;
    }
    out[i] = (buf & 0x00000001) != 0;  // 将buf的最低位作为布尔值存入数组
  }
}

long rk_long(rk_state *state) { return rk_ulong(state) >> 1; }  // 返回一个长整型随机数

unsigned long rk_ulong(rk_state *state) {
#if ULONG_MAX <= 0xffffffffUL
  return rk_random(state);  // 如果unsigned long的最大值小于等于0xffffffffUL，则返回一个随机数
#else
  return (rk_random(state) << 32) | (rk_random(state));  // 否则返回一个64位随机数
#endif
}

unsigned long rk_interval(unsigned long max, rk_state *state) {
  unsigned long mask = max, value;

  if (max == 0) {
    return 0;
  }
  /* Smallest bit mask >= max */
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
#if ULONG_MAX > 0xffffffffUL
  mask |= mask >> 32;
#endif

  /* Search a random value in [0..mask] <= max */
#if ULONG_MAX > 0xffffffffUL
  if (max <= 0xffffffffUL) {
    while ((value = (rk_random(state) & mask)) > max)
      ;
  } else {
    while ((value = (rk_ulong(state) & mask)) > max)
      ;
  }
#else
  while ((value = (rk_ulong(state) & mask)) > max)
    ;
#endif

  return value;  // 返回找到的随机值
}
    # 当条件 ((value = (rk_ulong(state) & mask)) > max) 满足时执行循环，直到条件不再满足。
    while ((value = (rk_ulong(state) & mask)) > max)
      ;
#else
  while ((value = (rk_ulong(state) & mask)) > max)
    ;
#endif
  return value;
}



double rk_double(rk_state *state) {
  /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
  // 生成两个随机数并进行右移，得到a和b
  long a = rk_random(state) >> 5, b = rk_random(state) >> 6;
  // 返回双精度浮点数，使用预先定义的常数进行计算
  return (a * 67108864.0 + b) / 9007199254740992.0;
}



void rk_fill(void *buffer, size_t size, rk_state *state) {
  unsigned long r;
  unsigned char *buf = buffer;

  // 对于每个4字节的块，填充随机数据
  for (; size >= 4; size -= 4) {
    r = rk_random(state);
    *(buf++) = r & 0xFF;
    *(buf++) = (r >> 8) & 0xFF;
    *(buf++) = (r >> 16) & 0xFF;
    *(buf++) = (r >> 24) & 0xFF;
  }

  // 处理剩余的字节（少于4字节）
  if (!size) {
    return;
  }
  r = rk_random(state);
  for (; size; r >>= 8, size--) {
    *(buf++) = (unsigned char)(r & 0xFF);
  }
}



rk_error rk_devfill(void *buffer, size_t size, int strong) {
#ifndef _WIN32
  FILE *rfile;
  int done;

  // 根据 strong 参数选择不同的随机设备文件
  if (strong) {
    rfile = fopen(RK_DEV_RANDOM, "rb");
  } else {
    rfile = fopen(RK_DEV_URANDOM, "rb");
  }
  // 打开文件失败时返回错误码
  if (rfile == NULL) {
    return RK_ENODEV;
  }
  // 从设备文件读取随机数据到缓冲区
  done = fread(buffer, size, 1, rfile);
  fclose(rfile);
  // 根据读取结果返回相应的错误码或成功码
  if (done) {
    return RK_NOERR;
  }
#else

#ifndef RK_NO_WINCRYPT
  HCRYPTPROV hCryptProv;
  BOOL done;

  // 在 Windows 下使用 CryptGenRandom 函数获取随机数据
  if (!CryptAcquireContext(&hCryptProv, NULL, NULL, PROV_RSA_FULL,
                           CRYPT_VERIFYCONTEXT) ||
      !hCryptProv) {
    return RK_ENODEV;
  }
  done = CryptGenRandom(hCryptProv, size, (unsigned char *)buffer);
  CryptReleaseContext(hCryptProv, 0);
  // 根据操作结果返回相应的错误码或成功码
  if (done) {
    return RK_NOERR;
  }
#endif

#endif
  // 默认返回设备不可用的错误码
  return RK_ENODEV;
}



rk_error rk_altfill(void *buffer, size_t size, int strong, rk_state *state) {
  rk_error err;

  // 调用 rk_devfill 函数填充缓冲区，根据返回值判断是否成功
  err = rk_devfill(buffer, size, strong);
  // 如果 rk_devfill 失败，则使用 rk_fill 函数填充缓冲区
  if (err) {
    rk_fill(buffer, size, state);
  }
  // 返回操作结果的错误码
  return err;
}



double rk_gauss(rk_state *state) {
  if (state->has_gauss) {
    // 如果状态中有预生成的高斯值，则返回并清空状态
    const double tmp = state->gauss;
    state->gauss = 0;
    state->has_gauss = 0;
    return tmp;
  } else {
    double f, x1, x2, r2;

    do {
      // 使用 Box-Muller 方法的极坐标法生成高斯分布的随机数对
      x1 = 2.0 * rk_double(state) - 1.0;
      x2 = 2.0 * rk_double(state) - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);

    // 计算正态分布的随机数并保留一个用于下一次调用
    f = sqrt(-2.0 * log(r2) / r2);
    state->gauss = f * x1;
    state->has_gauss = 1;
    return f * x2;
  }
}
```