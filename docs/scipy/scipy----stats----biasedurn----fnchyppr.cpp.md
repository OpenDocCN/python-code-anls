# `D:\src\scipysrc\scipy\scipy\stats\biasedurn\fnchyppr.cpp`

```
/*************************** fnchyppr.cpp **********************************
* Author:        Agner Fog
* Date created:  2002-10-20
* Last modified: 2014-06-14
* Project:       stocc.zip
* Source URL:    www.agner.org/random
*
* Description:
* Calculation of univariate and multivariate Fisher's noncentral hypergeometric
* probability distribution.
*
* This file contains source code for the class CFishersNCHypergeometric 
* and CMultiFishersNCHypergeometric defined in stocc.h.
*
* Documentation:
* ==============
* The file stocc.h contains class definitions.
* The file ran-instructions.pdf contains further documentation and 
* instructions.
*
* Copyright 2002-2014 by Agner Fog. 
* Released under SciPy's license with permission of Agner Fog; see license.txt
*****************************************************************************/

#include <string.h>                    // memcpy function
#include "stocc.h"                     // class definition


/***********************************************************************
Methods for class CFishersNCHypergeometric
***********************************************************************/

// CFishersNCHypergeometric 类的构造函数，用于初始化对象
CFishersNCHypergeometric::CFishersNCHypergeometric(int32_t n, int32_t m, int32_t N, double odds, double accuracy) {
   // constructor
   // 设置参数
   this->n = n;  this->m = m;  this->N = N;
   this->odds = odds;  this->accuracy = accuracy;

   // 检查参数的有效性
   if (n < 0 || m < 0 || N < 0 || odds < 0. || n > N || m > N) {
      FatalError("Parameter out of range in class CFishersNCHypergeometric");
   }
   if (accuracy < 0) accuracy = 0;
   if (accuracy > 1) accuracy = 1;
   // 初始化
   logodds = log(odds);  scale = rsum = 0.;
   ParametersChanged = 1;
   // 计算 xmin 和 xmax
   xmin = m + n - N;  if (xmin < 0) xmin = 0;
   xmax = n;  if (xmax > m) xmax = m;
}

// 计算并返回模式（精确值）
int32_t CFishersNCHypergeometric::mode(void) {
   // Find mode (exact)
   // 使用 Liao 和 Rosen 的方法来计算模式，参考 The American Statistician, vol 55,
   // no 4, 2001, p. 366-369.
   // 注意，Liao 和 Rosen 的公式有误。在其公式中用 -1 替换 sgn(b)。

   double A, B, C, D;                  // 用于二次方程的系数
   double x;                           // 模式的值
   int32_t L = m + n - N;
   int32_t m1 = m+1, n1 = n+1;

   if (odds == 1.) { 
      // 简单超几何分布
      x = (m + 1.) * (n + 1.) / (N + 2.);
   }
   else {
      // 类似于 Cornfield 均值的计算方法
      A = 1. - odds;
      B = (m1+n1)*odds - L; 
      C = -(double)m1*n1*odds;
      D = B*B -4*A*C;
      D = D > 0. ? sqrt(D) : 0.;
      x = (D - B)/(A+A);
   }
   return (int32_t)x;
}
// 计算 Fisher 非中心超几何分布的均值
double CFishersNCHypergeometric::mean(void) {
   // 查找近似均值
   // 计算类似于众数的值
   double a, b;                        // 计算中的临时变量
   double mean;                        // 均值

   if (odds == 1.) {                   // 简单的超几何分布
      return double(m)*n/N;
   }
   // 计算 Cornfield 均值
   a = (m+n)*odds + (N-m-n); 
   b = a*a - 4.*odds*(odds-1.)*m*n;
   b = b > 0. ? sqrt(b) : 0.;
   mean = (a-b)/(2.*(odds-1.));
   return mean;
}

// 计算 Fisher 非中心超几何分布的方差
double CFishersNCHypergeometric::variance(void) {
   // 查找近似方差（粗略估计）
   double my = mean(); // 近似均值
   // 从 Fisher 非中心超几何分布的估计中找到近似方差
   double r1 = my * (m-my); double r2 = (n-my)*(my+N-n-m);
   if (r1 <= 0. || r2 <= 0.) return 0.;
   double var = N*r1*r2/((N-1)*(m*r2+(N-m)*r1));
   if (var < 0.) var = 0.;
   return var;
}

// 计算 Fisher 非中心超几何分布的矩
double CFishersNCHypergeometric::moments(double * mean_, double * var_) {
   // 计算精确均值和方差
   // 返回值 = f(x) 的总和，期望 = 1
   double y, sy=0, sxy=0, sxxy=0, me1;
   int32_t x, xm, x1;
   const double accur = 0.1 * accuracy;     // 计算精度
   xm = (int32_t)mean();                      // 均值的近似值
   for (x=xm; x<=xmax; x++) {
      y = probability(x);
      x1 = x - xm;  // 减去近似均值以避免在求和中丢失精度
      sy += y; sxy += x1 * y; sxxy += x1 * x1 * y;
      if (y < accur && x != xm) break;
   }
   for (x=xm-1; x>=xmin; x--) {
      y = probability(x);
      x1 = x - xm;  // 减去近似均值以避免在求和中丢失精度
      sy += y; sxy += x1 * y; sxxy += x1 * x1 * y;
      if (y < accur) break;
   }
   me1 = sxy / sy;
   *mean_ = me1 + xm;
   y = sxxy / sy - me1 * me1;
   if (y < 0) y=0;
   *var_ = y;
   return sy;
}
// 计算 Fisher's 非中心超几何分布的概率函数
double CFishersNCHypergeometric::probability(int32_t x) {
   // 精确度为计算的准确度乘以0.1
   const double accur = accuracy * 0.1;

   // 如果 x 小于最小值 xmin 或者大于最大值 xmax，则返回概率为 0
   if (x < xmin || x > xmax) return 0;

   // 如果总抽样次数 n 等于 0，则返回概率为 1
   if (n == 0) return 1.;

   // 如果 odds 等于 1，则为中心超几何分布
   if (odds == 1.) {
      // 返回概率值，使用指数函数计算中心超几何分布的对数
      return exp(
         LnFac(m)   - LnFac(x)   - LnFac(m-x) +
         LnFac(N-m) - LnFac(n-x) - LnFac((N-m)-(n-x)) -
        (LnFac(N)   - LnFac(n)   - LnFac(N-n)));
   }

   // 如果 odds 等于 0，则为极端超几何分布
   if (odds == 0.) {
      // 如果 n 大于 N-m，则抛出致命错误
      if (n > N-m) FatalError("Not enough items with nonzero weight in CFishersNCHypergeometric::probability");
      // 返回概率值，如果 x 等于 0 则为真，否则为假
      return x == 0;
   }

   // 如果 rsum 未定义
   if (!rsum) {
      // 首次计算，计算 rsum = 所有可能的 x 值上比例函数的倒数之和
      int32_t x1, x2;                    // x 循环变量
      double y;                          // 比例函数值
      x1 = (int32_t)mean();              // 从均值开始
      if (x1 < xmin) x1 = xmin;
      x2 = x1 + 1;
      scale = 0.; scale = lng(x1);       // 计算缩放因子以避免溢出
      rsum = 1.;                         // = exp(lng(x1))，带有此缩放因子
      for (x1--; x1 >= xmin; x1--) {
         rsum += y = exp(lng(x1));       // 从 x1 开始向下累加
         if (y < accur) break;           // 直到值变得可忽略
      }
      for (; x2 <= xmax; x2++) {         // 从 x2 开始向上累加
         rsum += y = exp(lng(x2));
         if (y < accur) break;           // 直到值变得可忽略
      }
      rsum = 1. / rsum;                  // 保存倒数之和
   }
   // 返回函数值，使用指数函数计算 x 的对数乘以 rsum
   return exp(lng(x)) * rsum;
}
double CFishersNCHypergeometric::probabilityRatio(int32_t x, int32_t x0) {
   // 计算概率比 f(x)/f(x0)
   // 这比计算单个概率要快得多，因为不需要 rsum
   double a1, a2, a3, a4, f1, f2, f3, f4;
   int32_t y, dx = x - x0;
   int invert = 0;

   // 如果 x 超出有效范围，则返回 0
   if (x < xmin || x > xmax) return 0.;
   // 如果 x0 超出有效范围，则抛出致命错误
   if (x0 < xmin || x0 > xmax) {
      FatalError("Infinity in CFishersNCHypergeometric::probabilityRatio");
   }
   // 如果 dx 等于 0，则返回 1
   if (dx == 0.) return 1.;
   // 如果 dx 小于 0，则进行反转操作，并更新 dx
   if (dx < 0.) {
      invert = 1;
      dx = -dx;
      y = x;  x = x0;  x0 = y;
   }
   // 初始化参数值
   a1 = m - x0;  a2 = n - x0;  a3 = x;  a4 = N - m - n + x;
   // 如果 dx <= 28 且 x <= 100000，避免溢出
   if (dx <= 28 && x <= 100000) {      // 避免溢出
      // 直接计算
      f1 = f2 = 1.;
      // 计算二项式比值
      for (y = 0; y < dx; y++) {
         f1 *= a1-- * a2--;
         f2 *= a3-- * a4--;
      }
      // 计算 odds^dx
      f3 = 1.;  f4 = odds;  y = dx;
      while (y) {
         if (f4 < 1.E-100) {
            f3 = 0.;  break;           // 避免下溢
         }
         if (y & 1) f3 *= f4;
         f4 *= f4;
         y = (unsigned long)(y) >> 1;
      }
      f1 = f3 * f1 / f2;
      if (invert) f1 = 1. / f1;
   }
   else {
      // 使用对数进行计算
      f1 = FallingFactorial(a1,dx) + FallingFactorial(a2,dx) -
           FallingFactorial(a3,dx) - FallingFactorial(a4,dx) +
           dx * log(odds);
      if (invert) f1 = -f1;
      f1 = exp(f1);
   }
   return f1;
}

double CFishersNCHypergeometric::lng(int32_t x) {
   // 比例函数的自然对数
   // 返回 lambda = log(m!*x!/(m-x)!*m2!*x2!/(m2-x2)!*odds^x)
   int32_t x2 = n - x,  m2 = N - m;
   // 如果参数已更改，则重新计算 mFac 和 xLast
   if (ParametersChanged) {
      mFac = LnFac(m) + LnFac(m2);
      xLast = -99; ParametersChanged = 0;
   }
   // 如果 m 和 m2 都小于 FAK_LEN，跳转到 DEFLT
   if (m < FAK_LEN && m2 < FAK_LEN)  goto DEFLT;
   // 根据 x 与 xLast 的差值进行计算
   switch (x - xLast) {
  case 0:   // x 未更改
     break;
  case 1:   // x 增加了，从前一个值计算
     xFac += log (double(x) * (m2-x2) / (double(x2+1)*(m-x+1)));
     break;
  case -1:  // x 减少了，从前一个值计算
     xFac += log (double(x2) * (m-x) / (double(x+1)*(m2-x2+1)));
     break;
  default: DEFLT: // 计算所有情况
     xFac = LnFac(x) + LnFac(x2) + LnFac(m-x) + LnFac(m2-x2);
   }
   // 更新 xLast 的值为当前的 x
   xLast = x;
   // 返回计算得到的结果
   return mFac - xFac + x * logodds - scale;
}


这段代码中，两个函数分别实现了概率比率的计算和对数函数的计算。第一个函数 `probabilityRatio` 计算两个参数 x 和 x0 之间的概率比率，根据参数的不同情况选择不同的计算方式。第二个函数 `lng` 则计算了比例函数的自然对数，其计算依赖于多个参数和函数调用。
// 构造函数，初始化 CMultiFishersNCHypergeometric 类
CMultiFishersNCHypergeometric::CMultiFishersNCHypergeometric(int32_t n_, int32_t * m_, double * odds_, int colors_, double accuracy_) {
   // 定义局部变量
   int32_t N1;
   int i;
   // 复制参数值
   n = n_;  m = m_;  odds = odds_;  colors = colors_;  accuracy = accuracy_;
   // 检查参数是否有效
   // （注意：BiasedUrn 包中还有更详细的有效性检查）
   for (N = N1 = 0, i = 0; i < colors; i++) {
      if (m[i] < 0 || odds[i] < 0) FatalError("Parameter negative in constructor for CMultiFishersNCHypergeometric");
      N += m[i];
      if (odds[i]) N1 += m[i];
   }
   // 检查总数是否满足需求
   if (N < n) FatalError("Not enough items in constructor for CMultiFishersNCHypergeometric");
   if (N1 < n) FatalError("Not enough items with nonzero weight in constructor for CMultiFishersNCHypergeometric");

   // 计算 mFac 和 logodds
   for (i = 0, mFac = 0.; i < colors; i++) {
      mFac += LnFac(m[i]);  // 计算 m[i] 的对数阶乘
      logodds[i] = log(odds[i]);  // 计算 odds[i] 的自然对数
   }
   // 初始化
   sn = 0;  // 初始化 sn 为 0
}


// 计算多元 Fisher 非中心超几何分布的近似均值，结果存储在 mu[0..colors-1] 中
void CMultiFishersNCHypergeometric::mean(double * mu) {
   // 迭代变量
   double r, r1;
   // 颜色均值
   double q;
   // 总权重
   double W;
   // 颜色索引
   int i;
   // 迭代计数器
   int iter = 0;

   // 简单情况处理
   if (colors < 3) {
      if (colors == 1) mu[0] = n;  // 只有一个颜色
      if (colors == 2) {
         // 两个颜色的情况
         mu[0] = CFishersNCHypergeometric(n, m[0], m[0] + m[1], odds[0] / odds[1]).mean();  // 使用 CFishersNCHypergeometric 类计算均值
         mu[1] = n - mu[0];  // 计算第二个颜色的均值
      }
      return;
   }
   // 取所有球的情况
   if (n == N) {
      for (i = 0; i < colors; i++) mu[i] = m[i];
      return;
   }

   // 初始化 r 的初始猜测值
   for (i = 0, W = 0.; i < colors; i++) W += m[i] * odds[i];
   r = (double)n * N / ((N - n) * W);

   // 迭代计算 r
   do {
      r1 = r;
      for (i = 0, q = 0.; i < colors; i++) {
         q += m[i] * r * odds[i] / (r * odds[i] + 1.);
      }
      r *= n * (N - q) / (q * (N - n));
      if (++iter > 100) FatalError("convergence problem in function CMultiFishersNCHypergeometric::mean");  // 迭代次数超过 100 次则报错
   } while (fabs(r - r1) > 1E-5);  // 当 r 的变化小于 1E-5 时停止迭代

   // 存储计算结果
   for (i = 0; i < colors; i++) {
      mu[i] = m[i] * r * odds[i] / (r * odds[i] + 1.);
   }
}
void CMultiFishersNCHypergeometric::variance(double * var) {
   // 计算多变量Fisher非中心超几何分布的近似方差（精度不太好）。
   // 结果存储在variance[0..colors-1]中。
   // 计算速度较快。
   // 注意：BiasedUrn软件包中的版本处理未使用的颜色。

   double r1, r2;                     // 中间变量
   double mu[MAXCOLORS];              // 均值数组
   int i;                             // 循环变量
   mean(mu);                          // 调用mean函数计算均值

   for (i = 0; i < colors; i++) {
      r1 = mu[i] * (m[i] - mu[i]);    // 计算方差的分子部分
      r2 = (n - mu[i]) * (mu[i] + N - n - m[i]); // 计算方差的分母部分
      if (r1 <= 0. || r2 <= 0.) {
         var[i] = 0.;                 // 处理分母为零的情况
      }
      else {
         var[i] = N * r1 * r2 / ((N - 1) * (m[i] * r2 + (N - m[i]) * r1)); // 计算方差
      }
   }
}


double CMultiFishersNCHypergeometric::probability(int32_t * x) {
   // 计算概率函数。
   // 注意：第一次调用需要很长时间，因为需要计算所有可能的x组合，其概率 > 精度，这可能非常大。
   // 计算使用对数以避免溢出。
   // （递归计算可能更快，但未实现）
   // 注意：BiasedUrn软件包中的版本处理未使用的颜色。

   int32_t xsum;                      // x的总和
   int i, em;                         // 循环变量，中间变量

   for (xsum = i = 0; i < colors; i++)  xsum += x[i]; // 计算x的总和
   if (xsum != n) {
      FatalError("sum of x values not equal to n in function CMultiFishersNCHypergeometric::probability"); // 错误处理
   }

   for (i = em = 0; i < colors; i++) {
      if (x[i] > m[i] || x[i] < 0 || x[i] < n - N + m[i]) return 0.; // 检查x的取值范围
      if (odds[i] == 0. && x[i]) return 0.; // 检查特殊情况下的概率
      if (x[i] == m[i] || odds[i] == 0.) em++; // 计算等概率事件的个数
   }

   if (n == 0 || em == colors) return 1.; // 处理特殊情况下的概率

   if (sn == 0) SumOfAll();            // 第一次初始化
   return exp(lng(x)) * rsum;          // 返回函数值
}


double CMultiFishersNCHypergeometric::moments(double * mean, double * variance, int32_t * combinations) {
   // 计算Fisher非中心超几何分布的均值和方差，通过计算所有概率 > 精度的x值组合。
   // 返回值为1。
   // 将均值存储在mean[0...colors-1]中。
   // 将方差存储在variance[0...colors-1]中。
   // 注意：BiasedUrn软件包中的版本处理未使用的颜色

   int i;                              // 循环变量

   if (sn == 0) {
      // 第一次初始化包括计算均值和方差
      SumOfAll();
   }

   // 直接复制结果
   for (i = 0; i < colors; i++) {
      mean[i] = sx[i];                 // 复制均值
      variance[i] = sxx[i];            // 复制方差
   }

   if (combinations) *combinations = sn; // 处理组合数

   return 1.;                          // 返回结果
}
void CMultiFishersNCHypergeometric::SumOfAll() {
   // this function does the very time consuming job of calculating the sum
   // of the proportional function g(x) over all possible combinations of
   // the x[i] values with probability > accuracy. These combinations are 
   // generated by the recursive function loop().
   // The mean and variance are generated as by-products.

   int i;                              // color index
   int32_t msum;                         // sum of m[i]

   // get approximate mean
   mean(sx);
   // round mean to integers
   for (i=0, msum=0; i < colors; i++) {
      // adjust each sx[i] to the nearest integer and compute msum
      msum += xm[i] = (int32_t)(sx[i]+0.4999999);
   }
   // adjust truncated x values to make the sum = n
   msum -= n;
   for (i = 0; msum < 0; i++) {
      // increment xm[i] and adjust msum to achieve sum of xm[i] = n
      if (xm[i] < m[i]) {
         xm[i]++; msum++;
      }
   }
   for (i = 0; msum > 0; i++) {
      // decrement xm[i] and adjust msum to achieve sum of xm[i] = n
      if (xm[i] > 0) {
         xm[i]--; msum--;
      }
   }

   // adjust scale factor to g(mean) to avoid overflow
   scale = 0.; 
   // calculate logarithm of product of xm[i]! for scaling factor
   scale = lng(xm);

   // initialize for recursive loops
   sn = 0;
   for (i = colors-1, msum = 0; i >= 0; i--) {
      // set up remaining values and calculate cumulative msum
      remaining[i] = msum;  msum += m[i];
   }
   for (i = 0; i < colors; i++) {
      // initialize sx[i] and sxx[i] for mean and variance calculations
      sx[i] = 0;  sxx[i] = 0;
   }

   // recursive loops to calculate sums of g(x) over all x combinations
   rsum = 1. / loop(n, 0);

   // calculate mean and variance
   for (i = 0; i < colors; i++) {
      // compute adjusted mean and variance using rsum
      sxx[i] = sxx[i]*rsum - sx[i]*sx[i]*rsum*rsum;
      sx[i] = sx[i]*rsum;
   }
}
// 定义 CMultiFishersNCHypergeometric 类中的 loop 方法，计算多重 Fisher's 非中心超几何分布的递归循环
double CMultiFishersNCHypergeometric::loop(int32_t n, int c) {
   // x 是颜色 c 的值，x0 是颜色 c 的初始值
   int32_t x, x0;
   // xmin 和 xmax 是 x[c] 的最小和最大值
   int32_t xmin, xmax;
   // s1 是 g(x) 值的总和，s2 是 g(x) 值的辅助变量，sum 是最终的总和
   double s1, s2, sum = 0.;
   // 循环计数器
   int i;

   // 如果 c < colors-1，即不是最后一个颜色
   if (c < colors-1) {
      // 计算给定 x[0]..x[c-1] 的情况下 x[c] 的最小和最大值
      xmin = n - remaining[c];  if (xmin < 0) xmin = 0;
      xmax = m[c];  if (xmax > n) xmax = n;
      x0 = xm[c];  if (x0 < xmin) x0 = xmin;  if (x0 > xmax) x0 = xmax;
      
      // 从均值 x0 开始循环 x[c] 的所有可能值
      for (x = x0, s2 = 0.; x <= xmax; x++) {
         xi[c] = x;
         // 递归调用，计算剩余颜色的循环
         sum += s1 = loop(n-x, c+1);
         // 当 s1 达到精度要求并且小于 s2 时，停止循环
         if (s1 < accuracy && s1 < s2) break;
         s2 = s1;
      }
      
      // 从均值 x0-1 开始循环 x[c] 的所有可能值
      for (x = x0-1; x >= xmin; x--) {
         xi[c] = x;
         // 递归调用，计算剩余颜色的循环
         sum += s1 = loop(n-x, c+1);
         // 当 s1 达到精度要求并且小于 s2 时，停止循环
         if (s1 < accuracy && s1 < s2) break;
         s2 = s1;
      }
   }
   else {
      // 最后一个颜色
      xi[c] = n;
      // 计算比例函数 g(x) 的值
      s1 = exp(lng(xi));
      // 更新 sums 和 squaresums
      for (i = 0; i < colors; i++) {
         sx[i]  += s1 * xi[i];
         sxx[i] += s1 * xi[i] * xi[i];
      }
      sn++;
      sum += s1;
   }
   // 返回总和
   return sum;
}

// 定义 CMultiFishersNCHypergeometric 类中的 lng 方法，计算比例函数 g(x) 的自然对数
double CMultiFishersNCHypergeometric::lng(int32_t * x) {
   // y 是 g(x) 的自然对数
   double y = 0.;
   // 循环计数器
   int i;
   // 计算 g(x) 的自然对数
   for (i = 0; i < colors; i++) {
      y += x[i]*logodds[i] - LnFac(x[i]) - LnFac(m[i]-x[i]);
   }
   // 返回 g(x) 的自然对数
   return mFac + y - scale;
}
```