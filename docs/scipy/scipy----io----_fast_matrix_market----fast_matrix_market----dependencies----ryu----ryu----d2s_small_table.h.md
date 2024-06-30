# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\dependencies\ryu\ryu\d2s_small_table.h`

```
#ifndef RYU_D2S_SMALL_TABLE_H
#define RYU_D2S_SMALL_TABLE_H
// 如果 RYU_D2S_SMALL_TABLE_H 宏未定义，则定义它，避免多重包含

#include "ryu/d2s_intrinsics.h"
// 引入 ryu/d2s_intrinsics.h 文件，这个文件中可能定义了 HAS_UINT128 和 uint128_t

// 下面的常量定义由 PrintDoubleLookupTable 生成

#define DOUBLE_POW5_INV_BITCOUNT 125
#define DOUBLE_POW5_BITCOUNT 125
// 定义双精度浮点数的逆五次幂和五次幂的位数分别为 125

static const uint64_t DOUBLE_POW5_INV_SPLIT2[15][2] = {
  {                    1u, 2305843009213693952u },
  {  5955668970331000884u, 1784059615882449851u },
  {  8982663654677661702u, 1380349269358112757u },
  {  7286864317269821294u, 2135987035920910082u },
  {  7005857020398200553u, 1652639921975621497u },
  { 17965325103354776697u, 1278668206209430417u },
  {  8928596168509315048u, 1978643211784836272u },
  { 10075671573058298858u, 1530901034580419511u },
  {   597001226353042382u, 1184477304306571148u },
  {  1527430471115325346u, 1832889850782397517u },
  { 12533209867169019542u, 1418129833677084982u },
  {  5577825024675947042u, 2194449627517475473u },
  { 11006974540203867551u, 1697873161311732311u },
  { 10313493231639821582u, 1313665730009899186u },
  { 12701016819766672773u, 2032799256770390445u }
};
// 预先计算的双精度浮点数的逆五次幂的分割常量数组

static const uint32_t POW5_INV_OFFSETS[19] = {
  0x54544554, 0x04055545, 0x10041000, 0x00400414, 0x40010000, 0x41155555,
  0x00000454, 0x00010044, 0x40000000, 0x44000041, 0x50454450, 0x55550054,
  0x51655554, 0x40004000, 0x01000001, 0x00010500, 0x51515411, 0x05555554,
  0x00000000
};
// 预先计算的双精度浮点数的逆五次幂的偏移量数组

static const uint64_t DOUBLE_POW5_SPLIT2[13][2] = {
  {                    0u, 1152921504606846976u },
  {                    0u, 1490116119384765625u },
  {  1032610780636961552u, 1925929944387235853u },
  {  7910200175544436838u, 1244603055572228341u },
  { 16941905809032713930u, 1608611746708759036u },
  { 13024893955298202172u, 2079081953128979843u },
  {  6607496772837067824u, 1343575221513417750u },
  { 17332926989895652603u, 1736530273035216783u },
  { 13037379183483547984u, 2244412773384604712u },
  {  1605989338741628675u, 1450417759929778918u },
  {  9630225068416591280u, 1874621017369538693u },
  {   665883850346957067u, 1211445438634777304u },
  { 14931890668723713708u, 1565756531257009982u }
};
// 预先计算的双精度浮点数的五次幂的分割常量数组

#endif
// 用于快速计算5的幂次方的偏移表，共21个预定义值
static const uint32_t POW5_OFFSETS[21] = {
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x40000000, 0x59695995,
  0x55545555, 0x56555515, 0x41150504, 0x40555410, 0x44555145, 0x44504540,
  0x45555550, 0x40004000, 0x96440440, 0x55565565, 0x54454045, 0x40154151,
  0x55559155, 0x51405555, 0x00000105
};

// 5的幂次方表的大小
#define POW5_TABLE_SIZE 26

// 64位无符号整数数组，存储5的幂次方表
static const uint64_t DOUBLE_POW5_TABLE[POW5_TABLE_SIZE] = {
  1ull, 5ull, 25ull, 125ull, 625ull, 3125ull, 15625ull, 78125ull, 390625ull,
  1953125ull, 9765625ull, 48828125ull, 244140625ull, 1220703125ull, 6103515625ull,
  30517578125ull, 152587890625ull, 762939453125ull, 3814697265625ull,
  19073486328125ull, 95367431640625ull, 476837158203125ull,
  2384185791015625ull, 11920928955078125ull, 59604644775390625ull,
  298023223876953125ull //, 1490116119384765625ull
};

#if defined(HAS_UINT128)

// 计算5^i的精确值，存储在给定的指针中，使用了Ryu所需的形式
static inline void double_computePow5(const uint32_t i, uint64_t* const result) {
  const uint32_t base = i / POW5_TABLE_SIZE;
  const uint32_t base2 = base * POW5_TABLE_SIZE;
  const uint32_t offset = i - base2;
  const uint64_t* const mul = DOUBLE_POW5_SPLIT2[base];

  if (offset == 0) {
    result[0] = mul[0];
    result[1] = mul[1];
    return;
  }

  const uint64_t m = DOUBLE_POW5_TABLE[offset];
  const uint128_t b0 = ((uint128_t) m) * mul[0];
  const uint128_t b2 = ((uint128_t) m) * mul[1];
  const uint32_t delta = pow5bits(i) - pow5bits(base2);
  const uint128_t shiftedSum = (b0 >> delta) + (b2 << (64 - delta)) + ((POW5_OFFSETS[i / 16] >> ((i % 16) << 1)) & 3);

  result[0] = (uint64_t) shiftedSum;
  result[1] = (uint64_t) (shiftedSum >> 64);
}

// 计算5^-i的精确值，存储在给定的指针中，使用了Ryu所需的形式
static inline void double_computeInvPow5(const uint32_t i, uint64_t* const result) {
  const uint32_t base = (i + POW5_TABLE_SIZE - 1) / POW5_TABLE_SIZE;
  const uint32_t base2 = base * POW5_TABLE_SIZE;
  const uint32_t offset = base2 - i;
  const uint64_t* const mul = DOUBLE_POW5_INV_SPLIT2[base]; // 1/5^base2

  if (offset == 0) {
    result[0] = mul[0];
    result[1] = mul[1];
    return;
  }

  const uint64_t m = DOUBLE_POW5_TABLE[offset]; // 5^offset
  const uint128_t b0 = ((uint128_t) m) * (mul[0] - 1);
  const uint128_t b2 = ((uint128_t) m) * mul[1]; // 1/5^base2 * 5^offset = 1/5^(base2-offset) = 1/5^i
  const uint32_t delta = pow5bits(base2) - pow5bits(i);
  const uint128_t shiftedSum =
    ((b0 >> delta) + (b2 << (64 - delta))) + 1 + ((POW5_INV_OFFSETS[i / 16] >> ((i % 16) << 1)) & 3);

  result[0] = (uint64_t) shiftedSum;
  result[1] = (uint64_t) (shiftedSum >> 64);
}

#else // defined(HAS_UINT128)

// 如果没有定义HAS_UINT128，这里应该补充对应的计算函数，但是当前未提供完整的代码示例

#endif // defined(HAS_UINT128)
// 计算 5 的 i 次方的结果，存储在 result 指向的数组中
static inline void double_computePow5(const uint32_t i, uint64_t* const result) {
    // 计算基数，即 i 所在的 POW5_TABLE_SIZE 范围
    const uint32_t base = i / POW5_TABLE_SIZE;
    // 计算基数的实际值
    const uint32_t base2 = base * POW5_TABLE_SIZE;
    // 计算偏移量，即 i 相对于 base2 的偏移
    const uint32_t offset = i - base2;
    // 获取乘法预计算表中 base 对应的数组，即 5 的 base2 次方分解为两个 uint64_t 的乘积
    const uint64_t* const mul = DOUBLE_POW5_SPLIT2[base];
    // 如果偏移量为 0，直接使用预计算结果
    if (offset == 0) {
        result[0] = mul[0];
        result[1] = mul[1];
        return;
    }
    // 否则，需要进行更复杂的乘法计算
    const uint64_t m = DOUBLE_POW5_TABLE[offset];
    uint64_t high1;
    const uint64_t low1 = umul128(m, mul[1], &high1);
    uint64_t high0;
    const uint64_t low0 = umul128(m, mul[0], &high0);
    const uint64_t sum = high0 + low1;
    // 处理溢出情况，如果 sum 比 high0 小，说明有溢出到 high1
    if (sum < high0) {
        ++high1; // 溢出到 high1
    }
    // 根据 delta 值进行右移操作，得到最终结果
    const uint32_t delta = pow5bits(i) - pow5bits(base2);
    result[0] = shiftright128(low0, sum, delta) + ((POW5_OFFSETS[i / 16] >> ((i % 16) << 1)) & 3);
    result[1] = shiftright128(sum, high1, delta);
}

// 计算 5 的 -i 次方的结果，存储在 result 指向的数组中，用于 Ryu 算法
static inline void double_computeInvPow5(const uint32_t i, uint64_t* const result) {
    // 计算基数，注意这里 i 可能为负数，所以需要特殊处理
    const uint32_t base = (i + POW5_TABLE_SIZE - 1) / POW5_TABLE_SIZE;
    // 计算实际的 base2 值
    const uint32_t base2 = base * POW5_TABLE_SIZE;
    // 计算偏移量，这里是 base2 和 i 之间的差值
    const uint32_t offset = base2 - i;
    // 获取乘法预计算表中 base 对应的数组，即 1 / 5 的 base2 次方分解为两个 uint64_t 的乘积
    const uint64_t* const mul = DOUBLE_POW5_INV_SPLIT2[base];
    // 如果偏移量为 0，直接使用预计算结果
    if (offset == 0) {
        result[0] = mul[0];
        result[1] = mul[1];
        return;
    }
    // 否则，进行复杂的乘法计算，这里 mul[0] - 1 是因为要计算 1 / 5 的 base2 次方
    const uint64_t m = DOUBLE_POW5_TABLE[offset];
    uint64_t high1;
    const uint64_t low1 = umul128(m, mul[1], &high1);
    uint64_t high0;
    const uint64_t low0 = umul128(m, mul[0] - 1, &high0);
    const uint64_t sum = high0 + low1;
    // 处理溢出情况
    if (sum < high0) {
        ++high1; // 溢出到 high1
    }
    // 根据 delta 值进行右移操作，得到最终结果
    const uint32_t delta = pow5bits(base2) - pow5bits(i);
    result[0] = shiftright128(low0, sum, delta) + 1 + ((POW5_INV_OFFSETS[i / 16] >> ((i % 16) << 1)) & 3);
    result[1] = shiftright128(sum, high1, delta);
}
```