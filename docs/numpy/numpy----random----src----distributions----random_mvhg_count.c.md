# `.\numpy\numpy\random\src\distributions\random_mvhg_count.c`

```py
/*
 *  random_multivariate_hypergeometric_count
 *
 *  Draw variates from the multivariate hypergeometric distribution--
 *  the "count" algorithm.
 *
 *  Parameters
 *  ----------
 *  bitgen_t *bitgen_state
 *      Pointer to a `bitgen_t` instance.
 *  int64_t total
 *      The sum of the values in the array `colors`.  (This is redundant
 *      information, but we know the caller has already computed it, so
 *      we might as well use it.)
 *  size_t num_colors
 *      The length of the `colors` array.
 *  int64_t *colors
 *      The array of colors (i.e. the number of each type in the collection
 *      from which the random variate is drawn).
 *  int64_t nsample
 *      The number of objects drawn without replacement for each variate.
 *      `nsample` must not exceed sum(colors).  This condition is not checked;
 *      it is assumed that the caller has already validated the value.
 *  size_t num_variates
 *      The number of variates to be produced and put in the array
 *      pointed to by `variates`.  One variate is a vector of length
 *      `num_colors`, so the array pointed to by `variates` must have length
 *      `num_variates * num_colors`.
 *  int64_t *variates
 *      The array that will hold the result.  It must have length
 *      `num_variates * num_colors`.
 *      The array is not initialized in the function; it is expected that the
 *      array has been initialized with zeros when the function is called.
 *
 *  Notes
 *  -----
 *  The "count" algorithm for drawing one variate is roughly equivalent to the
 *  following numpy code:
 *
 *      choices = np.repeat(np.arange(len(colors)), colors)
 *      selection = np.random.choice(choices, nsample, replace=False)
 *      variate = np.bincount(selection, minlength=len(colors))
 *
 *  This function uses a temporary array with length sum(colors).
 *
 *  Assumptions on the arguments (not checked in the function):
 *    *  colors[k] >= 0  for k in range(num_colors)
 *    *  total = sum(colors)
 *    *  0 <= nsample <= total
 *    *  the product total * sizeof(size_t) does not exceed SIZE_MAX
 *    *  the product num_variates * num_colors does not overflow
 */

int random_multivariate_hypergeometric_count(bitgen_t *bitgen_state,
                      int64_t total,
                      size_t num_colors, int64_t *colors,
                      int64_t nsample,
                      size_t num_variates, int64_t *variates)
{
    size_t *choices;
    bool more_than_half;

    if ((total == 0) || (nsample == 0) || (num_variates == 0)) {
        // 如果 total, nsample 或者 num_variates 为 0，则不进行任何操作，直接返回
        return 0;
    }

    // 为 choices 数组分配内存空间，大小为 total 乘以 sizeof(size_t)
    choices = malloc(total * (sizeof *choices));
    if (choices == NULL) {
        // 如果分配内存失败，返回错误码 -1
        return -1;
    }

    /*
     *  如果 colors 数组包含，例如，[3 2 5]，那么 choices 数组将包含 [0 0 0 1 1 2 2 2 2 2]。
     */
    // 循环遍历每种颜色的个数，生成选择数组
    for (size_t i = 0, k = 0; i < num_colors; ++i) {
        // 根据每种颜色的个数，依次将该颜色索引添加到选择数组中
        for (int64_t j = 0; j < colors[i]; ++j) {
            choices[k] = i;
            ++k;
        }
    }

    // 检查采样数是否超过总数的一半
    more_than_half = nsample > (total / 2);
    if (more_than_half) {
        // 如果超过一半，则将采样数调整为总数减去采样数
        nsample = total - nsample;
    }

    // 对每个变量和每种颜色的组合进行采样和洗牌
    for (size_t i = 0; i < num_variates * num_colors; i += num_colors) {
        /*
         * Fisher-Yates 洗牌算法，但只针对选择数组的前 `nsample` 个条目。
         * 循环结束后，choices[:nsample] 包含从整个数组中随机采样的结果。
         */
        for (size_t j = 0; j < (size_t) nsample; ++j) {
            size_t tmp, k;
            // 注意：nsample 不大于 total，因此在 `(size_t) total - j - 1` 中不会发生整数下溢
            k = j + (size_t) random_interval(bitgen_state,
                                             (size_t) total - j - 1);
            tmp = choices[k];
            choices[k] = choices[j];
            choices[j] = tmp;
        }
        /*
         * 计算 choices[:nsample] 中每个值的出现次数。
         * 结果存储在 sample[i:i+num_colors] 中，表示多元超几何分布的样本。
         */
        for (size_t j = 0; j < (size_t) nsample; ++j) {
            variates[i + choices[j]] += 1;
        }

        // 如果采样数超过总数的一半，则进行特殊处理
        if (more_than_half) {
            for (size_t k = 0; k < num_colors; ++k) {
                variates[i + k] = colors[k] - variates[i + k];
            }
        }
    }

    // 释放动态分配的选择数组内存
    free(choices);

    // 返回成功代码
    return 0;
}



# 结束函数或代码块的定义，这里表示一个函数或类的结束。
```