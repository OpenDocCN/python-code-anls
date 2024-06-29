# `.\numpy\numpy\random\src\distributions\random_mvhg_marginals.c`

```
#include "numpy/random/distributions.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#include "logfactorial.h"

// 定义一个函数，用于生成多元超几何分布的边缘分布样本
void random_multivariate_hypergeometric_marginals(bitgen_t *bitgen_state,
                           int64_t total,
                           size_t num_colors, int64_t *colors,
                           int64_t nsample,
                           size_t num_variates, int64_t *variates)
{
    bool more_than_half;

    // 如果总数、样本数或变量数为零，无需执行任何操作，直接返回
    if ((total == 0) || (nsample == 0) || (num_variates == 0)) {
        // Nothing to do.
        return;
    }

    // 判断样本数是否超过总数的一半
    more_than_half = nsample > (total / 2);
    if (more_than_half) {
        // 如果样本数超过总数的一半，则调整样本数为总数减去样本数
        nsample = total - nsample;
    }

    // 遍历每个变量和颜色的组合
    for (size_t i = 0; i < num_variates * num_colors; i += num_colors) {
        int64_t num_to_sample = nsample;
        int64_t remaining = total;

        // 对于每个颜色，采样指定数量的对象
        for (size_t j = 0; (num_to_sample > 0) && (j + 1 < num_colors); ++j) {
            int64_t r;
            remaining -= colors[j];
            // 使用随机超几何分布生成随机样本数
            r = random_hypergeometric(bitgen_state,
                                      colors[j], remaining, num_to_sample);
            variates[i + j] = r;
            num_to_sample -= r;
        }

        // 如果仍有剩余的样本需要采样，则将其放入最后一个颜色的位置
        if (num_to_sample > 0) {
            variates[i + num_colors - 1] = num_to_sample;
        }

        // 如果样本数超过了总数的一半，需要进行反转操作
        if (more_than_half) {
            for (size_t k = 0; k < num_colors; ++k) {
                variates[i + k] = colors[k] - variates[i + k];
            }
        }
    }
}
```