# `.\numpy\numpy\random\src\pcg64\pcg64.orig.c`

```
// 包含自定义的pcg64随机数生成器头文件
#include "pcg64.orig.h"

// 声明设置PCG64随机数生成器种子的内联函数，初始化状态和序列
extern inline void pcg_setseq_128_srandom_r(pcg64_random_t *rng,
                                            pcg128_t initstate,
                                            pcg128_t initseq);

// 声明64位右旋函数，用于PCG64算法
extern uint64_t pcg_rotr_64(uint64_t value, unsigned int rot);

// 声明128位状态下XSL-RR算法的输出函数，返回64位整数
extern inline uint64_t pcg_output_xsl_rr_128_64(pcg128_t state);

// 声明PCG128状态设置序列的步进函数
extern void pcg_setseq_128_step_r(struct pcg_state_setseq_128 *rng);

// 声明基于PCG128状态设置序列的XSL-RR算法的64位随机数生成函数
extern uint64_t pcg_setseq_128_xsl_rr_64_random_r(struct pcg_state_setseq_128 *rng);
```