# `.\numpy\numpy\random\src\distributions\random_hypergeometric.c`

```
/*
 *  Generate a sample from the hypergeometric distribution.
 *
 *  Assume sample is not greater than half the total.  See below
 *  for how the opposite case is handled.
 *
 *  We initialize the following:
 *      computed_sample = sample
 *      remaining_good = good
 *      remaining_total = good + bad
 *
 *  In the loop:
 *  * computed_sample counts down to 0;
 *  * remaining_good is the number of good choices not selected yet;
 *  * remaining_total is the total number of choices not selected yet.
 *
 *  In the loop, we select items by choosing a random integer in
 *  the interval [0, remaining_total), and if the value is less
 *  than remaining_good, it means we have selected a good one,
 *  so remaining_good is decremented.  Then, regardless of that
 *  result, computed_sample is decremented.  The loop continues
 *  until either computed_sample is 0, remaining_good is 0, or
 *  remaining_total == remaining_good.  In the latter case, it
 *  means there are only good choices left, so we can stop the
 *  loop early and select what is left of computed_sample from
 *  the good choices (i.e. decrease remaining_good by computed_sample).
 *
 *  When the loop exits, the actual number of good choices is
 *  good - remaining_good.
 *
 *  If sample is more than half the total, then initially we set
 *      computed_sample = total - sample
 *  and at the end we return remaining_good (i.e. the loop in effect
 *  selects the complement of the result).
 *
 *  It is assumed that when this function is called:
 *    * good, bad and sample are nonnegative;
 *    * the sum good+bad will not result in overflow; 
 *    * sample <= good+bad.
 */

static int64_t hypergeometric_sample(bitgen_t *bitgen_state,
                                     int64_t good, int64_t bad, int64_t sample)
{
    int64_t remaining_total, remaining_good, result, computed_sample;
    int64_t total = good + bad;

    if (sample > total/2) {
        computed_sample = total - sample;  // Set computed_sample to the complement when sample is more than half the total
    }
    else {
        computed_sample = sample;  // Use sample directly when it is less than or equal to half the total
    }

    remaining_total = total;
    remaining_good = good;

    while ((computed_sample > 0) && (remaining_good > 0) &&
           (remaining_total > remaining_good)) {
         // random_interval(bitgen_state, max) returns an integer in
         // [0, max] *inclusive*, so we decrement remaining_total before
         // passing it to random_interval().
        --remaining_total;  // Decrement remaining_total since random_interval is inclusive
        if ((int64_t) random_interval(bitgen_state,
                                      remaining_total) < remaining_good) {
            // Selected a "good" one, so decrement remaining_good.
            --remaining_good;  // Decrement remaining_good for each "good" choice selected
        }
        --computed_sample;  // Decrement computed_sample for each iteration of the loop
    }

    if (remaining_total == remaining_good) {
        // Only "good" choices are left, adjust remaining_good accordingly
        remaining_good -= computed_sample;
    }

    if (sample > total/2) {
        result = remaining_good;  // Return remaining_good as result when sample is more than half the total
    }
    # 如果条件不满足，则执行以下语句块
    else:
        # 计算 result 的值为 good 减去 remaining_good
        result = good - remaining_good
    
    # 返回变量 result 的值作为函数的结果
    return result
// 结束预处理指令部分

// 定义常数 D1 和 D2，这些常数用于计算超几何分布的变量
#define D1 1.7155277699214135
#define D2 0.8989161620588988

/*
 * 使用比例-均匀算法生成超几何分布的变量
 *
 * 在代码中，变量名 a, b, c, g, h, m, p, q, K, T, U 和 X 对应于
 * Stadlober 1989 年论文中“算法 HRUA”从第 82 页开始使用的名称。
 *
 * 假设调用此函数时：
 *   * good, bad 和 sample 都为非负数；
 *   * good+bad 的和不会导致溢出；
 *   * sample <= good+bad。
 *
 * 参考文献：
 * - Ernst Stadlober 的论文 "Sampling from Poisson, Binomial and
 *   Hypergeometric Distributions: Ratio of Uniforms as a Simple and
 *   Fast Alternative" (1989)
 * - Ernst Stadlober, "The ratio of uniforms approach for generating
 *   discrete random variates", Journal of Computational and Applied
 *   Mathematics, 31, pp. 181-189 (1990).
 */

// 声明静态函数 hypergeometric_hrua，返回值类型为 int64_t
static int64_t hypergeometric_hrua(bitgen_t *bitgen_state,
                                   int64_t good, int64_t bad, int64_t sample)
{
    // 声明变量
    int64_t mingoodbad, maxgoodbad, popsize;
    int64_t computed_sample;
    double p, q;
    double mu, var;
    double a, c, b, h, g;
    int64_t m, K;

    // 计算总体大小
    popsize = good + bad;
    // 计算实际采样量
    computed_sample = MIN(sample, popsize - sample);
    // 计算最小的 good 和 bad 的值
    mingoodbad = MIN(good, bad);
    // 计算最大的 good 和 bad 的值
    maxgoodbad = MAX(good, bad);

    /*
     *  不与 Stadlober (1989) 相符的变量
     *    这里             Stadlober
     *    ----------------   ---------
     *    mingoodbad            M
     *    popsize               N
     *    computed_sample       n
     */

    // 计算成功概率 p 和失败概率 q
    p = ((double) mingoodbad) / popsize;
    q = ((double) maxgoodbad) / popsize;

    // mu 是分布的均值
    mu = computed_sample * p;

    // var 是分布的方差
    var = ((double)(popsize - computed_sample) *
           computed_sample * p * q / (popsize - 1));

    // 计算参数 a 和 c
    a = mu + 0.5;
    c = sqrt(var + 0.5);

    /*
     *  h 是 2*s_hat（参见 Stadlober 的论文 (1989)，公式 (5.17)；
     *  或 Stadlober (1990)，公式 8）。s_hat 是主导标准化超几何 PMF 的
     *  “表山”函数的比例尺度（“标准化”意味着具有最大值为 1）。
     */
    h = D1*c + D2;

    // 计算参数 m
    m = (int64_t) floor((double)(computed_sample + 1) * (mingoodbad + 1) /
                        (popsize + 2));

    // 计算参数 g
    g = (logfactorial(m) +
         logfactorial(mingoodbad - m) +
         logfactorial(computed_sample - m) +
         logfactorial(maxgoodbad - computed_sample + m));
    /*
     *  b is the upper bound for random samples:
     *  ... min(computed_sample, mingoodbad) + 1 is the length of the support.
     *  ... floor(a + 16*c) is 16 standard deviations beyond the mean.
     *
     *  The idea behind the second upper bound is that values that far out in
     *  the tail have negligible probabilities.
     *
     *  There is a comment in a previous version of this algorithm that says
     *      "16 for 16-decimal-digit precision in D1 and D2",
     *  but there is no documented justification for this value.  A lower value
     *  might work just as well, but I've kept the value 16 here.
     */
    // 计算随机样本的上界
    b = MIN(MIN(computed_sample, mingoodbad) + 1, floor(a + 16*c));

    // 开始采样循环，直到满足条件退出循环
    while (1) {
        double U, V, X, T;
        double gp;
        // 生成两个均匀分布的随机数
        U = next_double(bitgen_state);
        V = next_double(bitgen_state);  // "U star" in Stadlober (1989)
        // 计算变量 X
        X = a + h*(V - 0.5) / U;

        // 快速拒绝策略：
        if ((X < 0.0) || (X >= b)) {
            // 如果 X 不在有效范围内，则继续下一次循环
            continue;
        }

        // 计算 K，并转换为整数
        K = (int64_t) floor(X);

        // 计算 gp 值
        gp = (logfactorial(K) +
              logfactorial(mingoodbad - K) +
              logfactorial(computed_sample - K) +
              logfactorial(maxgoodbad - computed_sample + K));

        // 计算 T 值
        T = g - gp;

        // 快速接受策略：
        if ((U*(4.0 - U) - 3.0) <= T) {
            // 如果条件满足，则退出循环
            break;
        }

        // 快速拒绝策略：
        if (U*(U - T) >= 1) {
            // 如果条件不满足，则继续下一次循环
            continue;
        }

        if (2.0*log(U) <= T) {
            // 接受样本
            break;  
        }
    }

    // 根据条件调整 K 的值
    if (good > bad) {
        K = computed_sample - K;
    }

    if (computed_sample < sample) {
        K = good - K;
    }

    // 返回最终确定的 K 值
    return K;
}
```