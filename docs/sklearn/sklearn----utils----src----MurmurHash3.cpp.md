# `D:\src\scipysrc\scikit-learn\sklearn\utils\src\MurmurHash3.cpp`

```
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.

#include "MurmurHash3.h"

//-----------------------------------------------------------------------------
// Platform-specific functions and macros

// Microsoft Visual Studio

#if defined(_MSC_VER)

#define FORCE_INLINE    __forceinline

#include <stdlib.h>

#define ROTL32(x,y)    _rotl(x,y)
#define ROTL64(x,y)    _rotl64(x,y)

#define BIG_CONSTANT(x) (x)

// Other compilers

#else    // defined(_MSC_VER)

#if defined(GNUC) && ((GNUC > 4) || (GNUC == 4 && GNUC_MINOR >= 4))

/* gcc version >= 4.4 4.1 = RHEL 5, 4.4 = RHEL 6.
 * Don't inline for RHEL 5 gcc which is 4.1 */
#define FORCE_INLINE attribute((always_inline))

#else

#define FORCE_INLINE

#endif

// Define rotation functions for non-Microsoft compilers

inline uint32_t rotl32 ( uint32_t x, int8_t r )
{
  return (x << r) | (x >> (32 - r));
}

inline uint64_t rotl64 ( uint64_t x, int8_t r )
{
  return (x << r) | (x >> (64 - r));
}

#define    ROTL32(x,y)    rotl32(x,y)
#define ROTL64(x,y)    rotl64(x,y)

#define BIG_CONSTANT(x) (x##LLU)

#endif // !defined(_MSC_VER)

//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

FORCE_INLINE uint32_t getblock ( const uint32_t * p, int i )
{
  return p[i];
}

FORCE_INLINE uint64_t getblock ( const uint64_t * p, int i )
{
  return p[i];
}

//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche

FORCE_INLINE uint32_t fmix ( uint32_t h )
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

//----------

FORCE_INLINE uint64_t fmix ( uint64_t k )
{
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;

  return k;
}

//-----------------------------------------------------------------------------

void MurmurHash3_x86_32 ( const void * key, int len,
                          uint32_t seed, void * out )
{
  const uint8_t * data = (const uint8_t*)key;
  const int nblocks = len / 4;

  uint32_t h1 = seed;

  uint32_t c1 = 0xcc9e2d51;
  uint32_t c2 = 0x1b873593;

  //----------
  // body

  // Interpret data as array of 32-bit unsigned integers
  const uint32_t * blocks = (const uint32_t *)(data + nblocks*4);

  // Iterate through each block
  for(int i = -nblocks; i; i++)
  {
    // Retrieve a block of data
    uint32_t k1 = getblock(blocks,i);

    // Perform mixing operations
    k1 *= c1;
    k1 = ROTL32(k1,15);
    k1 *= c2;

    h1 ^= k1;
    h1 = ROTL32(h1,13);
    h1 = h1*5+0xe6546b64;

# 计算新的哈希值h1，使用MurmurHash3算法的混合步骤，乘以常数5并加上常数0xe6546b64

  }

  //----------
  // tail

  const uint8_t * tail = (const uint8_t*)(data + nblocks*4);

  // 从数据中找到尾部的未处理字节块，转换为uint8_t指针
  uint32_t k1 = 0;

  switch(len & 3)
  {
  case 3: k1 ^= tail[2] << 16;
  case 2: k1 ^= tail[1] << 8;
  case 1: k1 ^= tail[0];
          k1 *= c1; k1 = ROTL32(k1,15); k1 *= c2; h1 ^= k1;
  }

  //----------
  // finalization

  // 将数据的总长度与哈希值h1进行异或操作
  h1 ^= len;

  // 对哈希值h1进行最后的混合处理
  h1 = fmix(h1);

  // 将最终计算得到的哈希值h1写入输出指针out所指向的位置
  *(uint32_t*)out = h1;
// 结束之前的 MurmurHash3_x86_128 函数的定义

//-----------------------------------------------------------------------------

void MurmurHash3_x86_128 ( const void * key, const int len,
                           uint32_t seed, void * out )
{
  // 将输入的 key 转换为 uint8_t 类型的数据
  const uint8_t * data = (const uint8_t*)key;
  // 计算数据块的数量
  const int nblocks = len / 16;

  // 初始化四个 32 位的哈希值 h1, h2, h3, h4，并使用给定的种子进行初始化
  uint32_t h1 = seed;
  uint32_t h2 = seed;
  uint32_t h3 = seed;
  uint32_t h4 = seed;

  // 设置四个常数 c1, c2, c3, c4 作为混合时使用的参数
  uint32_t c1 = 0x239b961b;
  uint32_t c2 = 0xab0e9789;
  uint32_t c3 = 0x38b34ae5;
  uint32_t c4 = 0xa1e38b93;

  //----------
  // body

  // 将 data 转换为 uint32_t* 类型的数据，跳过已经处理过的数据块
  const uint32_t * blocks = (const uint32_t *)(data + nblocks*16);

  // 循环处理每一个数据块
  for(int i = -nblocks; i; i++)
  {
    // 获取第 i 个数据块的四个 32 位的数据
    uint32_t k1 = getblock(blocks,i*4+0);
    uint32_t k2 = getblock(blocks,i*4+1);
    uint32_t k3 = getblock(blocks,i*4+2);
    uint32_t k4 = getblock(blocks,i*4+3);

    // 混合第一个数据块
    k1 *= c1; k1  = ROTL32(k1,15); k1 *= c2; h1 ^= k1;
    h1 = ROTL32(h1,19); h1 += h2; h1 = h1*5+0x561ccd1b;

    // 混合第二个数据块
    k2 *= c2; k2  = ROTL32(k2,16); k2 *= c3; h2 ^= k2;
    h2 = ROTL32(h2,17); h2 += h3; h2 = h2*5+0x0bcaa747;

    // 混合第三个数据块
    k3 *= c3; k3  = ROTL32(k3,17); k3 *= c4; h3 ^= k3;
    h3 = ROTL32(h3,15); h3 += h4; h3 = h3*5+0x96cd1c35;

    // 混合第四个数据块
    k4 *= c4; k4  = ROTL32(k4,18); k4 *= c1; h4 ^= k4;
    h4 = ROTL32(h4,13); h4 += h1; h4 = h4*5+0x32ac3b17;
  }

  //----------
  // tail

  // 处理剩余的不足一个数据块的数据
  const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

  uint32_t k1 = 0;
  uint32_t k2 = 0;
  uint32_t k3 = 0;
  uint32_t k4 = 0;

  // 根据剩余数据的长度选择不同的混合方式
  switch(len & 15)
  {
  case 15: k4 ^= tail[14] << 16;
  case 14: k4 ^= tail[13] << 8;
  case 13: k4 ^= tail[12] << 0;
           k4 *= c4; k4  = ROTL32(k4,18); k4 *= c1; h4 ^= k4;

  case 12: k3 ^= tail[11] << 24;
  case 11: k3 ^= tail[10] << 16;
  case 10: k3 ^= tail[ 9] << 8;
  case  9: k3 ^= tail[ 8] << 0;
           k3 *= c3; k3  = ROTL32(k3,17); k3 *= c4; h3 ^= k3;

  case  8: k2 ^= tail[ 7] << 24;
  case  7: k2 ^= tail[ 6] << 16;
  case  6: k2 ^= tail[ 5] << 8;
  case  5: k2 ^= tail[ 4] << 0;
           k2 *= c2; k2  = ROTL32(k2,16); k2 *= c3; h2 ^= k2;

  case  4: k1 ^= tail[ 3] << 24;
  case  3: k1 ^= tail[ 2] << 16;
  case  2: k1 ^= tail[ 1] << 8;
  case  1: k1 ^= tail[ 0] << 0;
           k1 *= c1; k1  = ROTL32(k1,15); k1 *= c2; h1 ^= k1;
  }

  //----------
  // finalization

  // 对每个哈希值进行最后的混合和处理
  h1 ^= len; h2 ^= len; h3 ^= len; h4 ^= len;

  h1 += h2; h1 += h3; h1 += h4;
  h2 += h1; h3 += h1; h4 += h1;

  h1 = fmix(h1);
  h2 = fmix(h2);
  h3 = fmix(h3);
  h4 = fmix(h4);

  h1 += h2; h1 += h3; h1 += h4;
  h2 += h1; h3 += h1; h4 += h1;

  // 将最终的哈希值写入输出数组 out 中
  ((uint32_t*)out)[0] = h1;
  ((uint32_t*)out)[1] = h2;
  ((uint32_t*)out)[2] = h3;
  ((uint32_t*)out)[3] = h4;
}

//-----------------------------------------------------------------------------

// 开始定义 MurmurHash3_x64_128 函数
void MurmurHash3_x64_128 ( const void * key, const int len,
                           const uint32_t seed, void * out )
{
  // 将输入的密钥数据转换为 uint8_t 类型的指针
  const uint8_t * data = (const uint8_t*)key;
  // 计算输入数据的块数（每个块为 16 字节）
  const int nblocks = len / 16;

  // 初始化两个 64 位的哈希变量 h1 和 h2，使用给定的种子值
  uint64_t h1 = seed;
  uint64_t h2 = seed;

  // 初始化两个常量 c1 和 c2，这些常量用于混合哈希值的计算
  uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  //----------
  // body

  // 将输入数据转换为 uint64_t 类型的指针，以便按 64 位块进行处理
  const uint64_t * blocks = (const uint64_t *)(data);

  // 循环处理每个块
  for(int i = 0; i < nblocks; i++)
  {
    // 从块中获取两个 64 位的数据块 k1 和 k2
    uint64_t k1 = getblock(blocks,i*2+0);
    uint64_t k2 = getblock(blocks,i*2+1);

    // 对 k1 应用混合常量 c1，进行位旋转和混合操作，然后与 h1 进行异或操作
    k1 *= c1; k1  = ROTL64(k1,31); k1 *= c2; h1 ^= k1;

    // 对 h1 进行位旋转、加法混合和乘法混合
    h1 = ROTL64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;

    // 对 k2 应用混合常量 c2，进行位旋转和混合操作，然后与 h2 进行异或操作
    k2 *= c2; k2  = ROTL64(k2,33); k2 *= c1; h2 ^= k2;

    // 对 h2 进行位旋转、加法混合和乘法混合
    h2 = ROTL64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
  }

  //----------
  // tail

  // 处理剩余的不足一个块大小的数据（尾部数据）
  const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

  // 初始化两个 64 位的 k1 和 k2 变量
  uint64_t k1 = 0;
  uint64_t k2 = 0;

  // 根据尾部数据的长度进行处理
  switch(len & 15)
  {
  case 15: k2 ^= uint64_t(tail[14]) << 48;
  case 14: k2 ^= uint64_t(tail[13]) << 40;
  case 13: k2 ^= uint64_t(tail[12]) << 32;
  case 12: k2 ^= uint64_t(tail[11]) << 24;
  case 11: k2 ^= uint64_t(tail[10]) << 16;
  case 10: k2 ^= uint64_t(tail[ 9]) << 8;
  case  9: k2 ^= uint64_t(tail[ 8]) << 0;
           k2 *= c2; k2  = ROTL64(k2,33); k2 *= c1; h2 ^= k2;

  case  8: k1 ^= uint64_t(tail[ 7]) << 56;
  case  7: k1 ^= uint64_t(tail[ 6]) << 48;
  case  6: k1 ^= uint64_t(tail[ 5]) << 40;
  case  5: k1 ^= uint64_t(tail[ 4]) << 32;
  case  4: k1 ^= uint64_t(tail[ 3]) << 24;
  case  3: k1 ^= uint64_t(tail[ 2]) << 16;
  case  2: k1 ^= uint64_t(tail[ 1]) << 8;
  case  1: k1 ^= uint64_t(tail[ 0]) << 0;
           k1 *= c1; k1  = ROTL64(k1,31); k1 *= c2; h1 ^= k1;
  }

  //----------
  // finalization

  // 对 h1 和 h2 进行最后的混合操作
  h1 ^= len; h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix(h1);
  h2 = fmix(h2);

  h1 += h2;
  h2 += h1;

  // 将最终的哈希值写入输出数组 out
  ((uint64_t*)out)[0] = h1;
  ((uint64_t*)out)[1] = h2;
}
```