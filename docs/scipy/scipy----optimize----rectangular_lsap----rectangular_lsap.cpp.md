# `D:\src\scipysrc\scipy\scipy\optimize\rectangular_lsap\rectangular_lsap.cpp`

```
/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


This code implements the shortest augmenting path algorithm for the
rectangular assignment problem.  This implementation is based on the
pseudocode described in pages 1685-1686 of:

    DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems
    52(4):1679-1696, August 2016
    doi: 10.1109/TAES.2016.140952

Author: PM Larsen
*/

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include "rectangular_lsap.h"

// Template function to return indices that sort the input vector
template <typename T> std::vector<intptr_t> argsort_iter(const std::vector<T> &v)
{
    // Create a vector of indices [0, 1, 2, ..., v.size()-1]
    std::vector<intptr_t> index(v.size());
    std::iota(index.begin(), index.end(), 0);
    
    // Sort indices based on values in vector v
    std::sort(index.begin(), index.end(), [&v](intptr_t i, intptr_t j)
              { return v[i] < v[j]; });
    
    return index; // Return sorted indices
}

// Static function to implement the shortest augmenting path algorithm
static intptr_t
augmenting_path(intptr_t nc, double *cost, std::vector<double>& u,
                std::vector<double>& v, std::vector<intptr_t>& path,
                std::vector<intptr_t>& row4col,
                std::vector<double>& shortestPathCosts, intptr_t i,
                std::vector<bool>& SR, std::vector<bool>& SC,
                std::vector<intptr_t>& remaining, double* p_minVal)
{
    double minVal = 0; // Initialize minimum value variable
    
    // Initialize number of remaining nodes with nc (number of columns)
    intptr_t num_remaining = nc;
    for (intptr_t it = 0; it < nc; it++) {
        // 将 remaining 数组倒序填充，确保对于常数代价矩阵的解是单位矩阵（参见 #11602）。
        remaining[it] = nc - it - 1;
    }

    // 将 SR, SC, shortestPathCosts 数组初始化
    std::fill(SR.begin(), SR.end(), false);
    std::fill(SC.begin(), SC.end(), false);
    std::fill(shortestPathCosts.begin(), shortestPathCosts.end(), INFINITY);

    // 查找最短增广路径
    intptr_t sink = -1;
    while (sink == -1) {

        intptr_t index = -1;
        double lowest = INFINITY;
        SR[i] = true;

        // 遍历剩余节点进行最短路径计算
        for (intptr_t it = 0; it < num_remaining; it++) {
            intptr_t j = remaining[it];

            double r = minVal + cost[i * nc + j] - u[i] - v[j];
            if (r < shortestPathCosts[j]) {
                path[j] = i;
                shortestPathCosts[j] = r;
            }

            // 当多个节点具有最小成本时，选择一个可以给我们新的汇节点。这对于具有小系数的整数成本矩阵特别重要。
            if (shortestPathCosts[j] < lowest ||
                (shortestPathCosts[j] == lowest && row4col[j] == -1)) {
                lowest = shortestPathCosts[j];
                index = it;
            }
        }

        minVal = lowest;
        if (minVal == INFINITY) { // 无法实现的成本矩阵
            return -1;
        }

        intptr_t j = remaining[index];
        if (row4col[j] == -1) {
            sink = j;
        } else {
            i = row4col[j];
        }

        SC[j] = true;
        remaining[index] = remaining[--num_remaining];
    }

    // 将最小值赋给 p_minVal，返回汇节点 sink
    *p_minVal = minVal;
    return sink;
static int
solve(intptr_t nr, intptr_t nc, double* cost, bool maximize,
      int64_t* a, int64_t* b)
{
    // 处理特殊情况，如果行数或列数为0，则返回0
    if (nr == 0 || nc == 0) {
        return 0;
    }

    // 如果列数小于行数，需要对成本矩阵进行转置
    bool transpose = nc < nr;

    // 如果需要转置或者进行最大化处理，需要复制成本矩阵
    std::vector<double> temp;
    if (transpose || maximize) {
        temp.resize(nr * nc);

        // 如果需要转置
        if (transpose) {
            // 将原始矩阵转置存储到临时矩阵中
            for (intptr_t i = 0; i < nr; i++) {
                for (intptr_t j = 0; j < nc; j++) {
                    temp[j * nr + i] = cost[i * nc + j];
                }
            }

            // 交换行数和列数
            std::swap(nr, nc);
        }
        else {
            // 否则直接复制原始矩阵到临时矩阵
            std::copy(cost, cost + nr * nc, temp.begin());
        }

        // 如果需要最大化处理，则取负数
        if (maximize) {
            for (intptr_t i = 0; i < nr * nc; i++) {
                temp[i] = -temp[i];
            }
        }

        // 将成本指针指向临时矩阵的数据
        cost = temp.data();
    }

    // 检查成本矩阵中是否存在NaN或-inf条目
    for (intptr_t i = 0; i < nr * nc; i++) {
        if (cost[i] != cost[i] || cost[i] == -INFINITY) {
            return RECTANGULAR_LSAP_INVALID;
        }
    }

    // 初始化变量
    std::vector<double> u(nr, 0);
    std::vector<double> v(nc, 0);
    std::vector<double> shortestPathCosts(nc);
    std::vector<intptr_t> path(nc, -1);
    std::vector<intptr_t> col4row(nr, -1);
    std::vector<intptr_t> row4col(nc, -1);
    std::vector<bool> SR(nr);
    std::vector<bool> SC(nc);
    std::vector<intptr_t> remaining(nc);

    // 逐步构建解决方案
    for (intptr_t curRow = 0; curRow < nr; curRow++) {

        // 找到增广路径的最小值和对应的终点
        double minVal;
        intptr_t sink = augmenting_path(nc, cost, u, v, path, row4col,
                                        shortestPathCosts, curRow, SR, SC,
                                        remaining, &minVal);
        if (sink < 0) {
            return RECTANGULAR_LSAP_INFEASIBLE;
        }

        // 更新对偶变量
        u[curRow] += minVal;
        for (intptr_t i = 0; i < nr; i++) {
            if (SR[i] && i != curRow) {
                u[i] += minVal - shortestPathCosts[col4row[i]];
            }
        }

        for (intptr_t j = 0; j < nc; j++) {
            if (SC[j]) {
                v[j] -= minVal - shortestPathCosts[j];
            }
        }

        // 增广前一步的解决方案
        intptr_t j = sink;
        while (1) {
            intptr_t i = path[j];
            row4col[j] = i;
            std::swap(col4row[i], j);
            if (i == curRow) {
                break;
            }
        }
    }

    // 如果需要转置结果
    if (transpose) {
        intptr_t i = 0;
        // 使用argsort_iter函数对col4row进行排序并存储结果到a和b数组中
        for (auto v: argsort_iter(col4row)) {
            a[i] = col4row[v];
            b[i] = v;
            i++;
        }
    }
    else {
        // 否则直接将结果存储到a和b数组中
        for (intptr_t i = 0; i < nr; i++) {
            a[i] = i;
            b[i] = col4row[i];
        }
    }

    // 返回0表示成功完成
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

int



solve(intptr_t nr, intptr_t nc, double* cost, bool maximize,
      int64_t* a, int64_t* b)
{
    // 处理特殊情况，如果行数或列数为0，则返回0
    if (nr == 0 || nc == 0) {
        return 0;
    }

    // 如果列数小于行数，需要对成本矩阵进行转置
    bool transpose = nc < nr;

    // 如果需要转置或者进行最大化处理，需要复制成本矩阵
    std::vector<double> temp;
    if (transpose || maximize) {
        temp.resize(nr * nc);

        // 如果需要转置
        if (transpose) {
            // 将原始矩阵转置存储到临时矩阵中
            for (intptr_t i = 0; i < nr; i++) {
                for (intptr_t j = 0; j < nc; j++) {
                    temp[j * nr + i] = cost[i * nc + j];
                }
            }

            // 交换行数和列数
            std::swap(nr, nc);
        }
        else {
            // 否则直接复制原始矩阵到临时矩阵
            std::copy(cost, cost + nr * nc, temp.begin());
        }

        // 如果需要最大化处理，则取负数
        if (maximize) {
            for (intptr_t i = 0; i < nr * nc; i++) {
                temp[i] = -temp[i];
            }
        }

        // 将成本指针指向临时矩阵的数据
        cost = temp.data();
    }

    // 检查成本矩阵中是否存在NaN或-inf条目
    for (intptr_t i = 0; i < nr * nc; i++) {
        if (cost[i] != cost[i] || cost[i] == -INFINITY) {
            return RECTANGULAR_LSAP_INVALID;
        }
    }

    // 初始化变量
    std::vector<double> u(nr, 0);
    std::vector<double> v(nc, 0);
    std::vector<double> shortestPathCosts(nc);
    std::vector<intptr_t> path(nc, -1);
    std::vector<intptr_t> col4row(nr, -1);
    std::vector<intptr_t> row4col(nc, -1);
    std::vector<bool> SR(nr);
    std::vector<bool> SC(nc);
    std::vector<intptr_t> remaining(nc);

    // 逐步构建解决方案
    for (intptr_t curRow = 0; curRow < nr; curRow++) {

        // 找到增广路径的最小值和对应的终点
        double minVal;
        intptr_t sink = augmenting_path(nc, cost, u, v, path, row4col,
                                        shortestPathCosts, curRow, SR, SC,
                                        remaining, &minVal);
        if (sink < 0) {
            return RECTANGULAR_LSAP_INFEASIBLE;
        }

        // 更新对偶变量
        u[curRow] += minVal;
        for (intptr_t i = 0; i < nr; i++) {
            if (SR[i] && i != curRow) {
                u[i] += minVal - shortestPathCosts[col4row[i]];
            }
        }

        for (intptr_t j = 0; j < nc; j++) {
            if (SC[j]) {
                v[j] -= minVal - shortestPathCosts[j];
            }
        }

        // 增广前一步的解决方案
        intptr_t j = sink;
        while (1) {
            intptr_t i = path[j];
            row4col[j] = i;
            std::swap(col4row[i], j);
            if (i == curRow) {
                break;
            }
        }
    }

    // 如果需要转置结果
    if (transpose) {
        intptr_t i = 0;
        // 使用argsort_iter函数对col4row进行排序并存储结果到a和b数组中
        for (auto v: argsort_iter(col4row)) {
            a[i] = col4row[v];
            b[i] = v;
            i++;
        }
    }
    else {
        // 否则直接将结果存储到a和b数组中
        for (intptr_t i = 0; i < nr; i++) {
            a[i] = i;
            b[i] = col4row[i];
        }
    }

    // 返回0表示成功完成
    return 0;
}
# 定义一个函数 solve_rectangular_linear_sum_assignment，接受多个参数：
#   - nr: 表示行数
#   - nc: 表示列数
#   - input_cost: 指向双精度浮点数数组的指针，存储了输入的成本数据
#   - maximize: 布尔值，指示是否最大化问题的解（如果为 true，则最大化）
#   - a: 指向 int64_t 类型数组的指针，将存储行指派给列的结果
#   - b: 指向 int64_t 类型数组的指针，将存储列指派给行的结果
solve_rectangular_linear_sum_assignment(intptr_t nr, intptr_t nc,
                                        double* input_cost, bool maximize,
                                        int64_t* a, int64_t* b)
{
    # 调用 solve 函数进行实际的解决方案计算，并返回结果
    return solve(nr, nc, input_cost, maximize, a, b);
}

#ifdef __cplusplus
}
#endif
```