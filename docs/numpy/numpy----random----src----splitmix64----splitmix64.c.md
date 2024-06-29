# `.\numpy\numpy\random\src\splitmix64\splitmix64.c`

```
/*  Written in 2015 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>.

Modified 2018 by Kevin Sheppard.  Modifications licensed under the NCSA
license.
*/
/* 该部分是由Sebastiano Vigna（vigna@acm.org）于2015年编写的。

作者尽可能在法律允许的范围内，将此软件的所有版权以及相关的邻接权利贡献给了全球公共领域。
此软件不带任何保证地分发。

参见 <http://creativecommons.org/publicdomain/zero/1.0/>。

2018年由Kevin Sheppard修改。修改根据NCSA许可证授权。
*/

/* This is a fixed-increment version of Java 8's SplittableRandom generator
   See https://doi.org/10.1145/2714064.2660195 and
   https://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

   It is a very fast generator passing BigCrush, and it can be useful if
   for some reason you absolutely want 64 bits of state; otherwise, we
   rather suggest to use a xoroshiro128+ (for moderately parallel
   computations) or xorshift1024* (for massively parallel computations)
   generator. */
/* 这是Java 8的SplittableRandom生成器的固定增量版本。
   参见 https://doi.org/10.1145/2714064.2660195 和
   https://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

   这是一个非常快速的生成器，通过了BigCrush测试，如果出于某种原因你绝对需要64位的状态，
   它可能会很有用；否则，我们建议使用xoroshiro128+（用于中等并行计算）或xorshift1024*（用于大规模并行计算）生成器。
*/

#include "splitmix64.h"

extern inline uint64_t splitmix64_next(uint64_t *state);

extern inline uint64_t splitmix64_next64(splitmix64_state *state);

extern inline uint32_t splitmix64_next32(splitmix64_state *state);

/* 包含splitmix64.h头文件，并声明了splitmix64_next、splitmix64_next64和splitmix64_next32三个函数的外部内联定义。 */
```