# `D:\src\scipysrc\scipy\scipy\stats\biasedurn\stoc3.cpp`

```
/*************************** stoc3.cpp **********************************
* Author:        Agner Fog
* Date created:  2002-10-02
* Last modified: 2008-11-21
* Project:       stocc.zip
* Source URL:    www.agner.org/random
*
* Description:
* Non-uniform random number generator functions.
*
* This file contains source code for the class StochasticLib3 derived
* from StochasticLib1 or StochasticLib2, defined in stocc.h.
*
* This class implements methods for sampling from the noncentral and extended 
* hypergeometric distributions, as well as the multivariate versions of these.
*
* Documentation:
* ==============
* The file stocc.h contains class definitions.
* The file stocc.htm contains further instructions.
* The file nchyp.pdf, available from www.agner.org/random/theory contains 
* theoretical description of Wallenius' and Fisher's noncentral hypergeometric
* distributions and the methods used in this code to sample from these.
* The file ran-instructions.pdf contains general instructions.
*
* Copyright 2002-2008 by Agner Fog. 
* Released under SciPy's license with permission of Agner Fog; see license.txt
*****************************************************************************/

#include <string.h>                    // memcpy function
#include "stocc.h"                     // class definitions
//#include "wnchyppr.cpp"              // calculate Wallenius noncentral hypergeometric probability
//#include "fnchyppr.cpp"              // calculate Fisher's noncentral hypergeometric probability


/******************************************************************************
Methods for class StochasticLib3
******************************************************************************/


/***********************************************************************
Constructor
***********************************************************************/
StochasticLib3::StochasticLib3(int seed) : StochasticLib1(seed) {
   SetAccuracy(1.E-8);                  // set default accuracy
   // Initialize variables
   fnc_n_last = -1, fnc_m_last = -1, fnc_N_last = -1;
   fnc_o_last = -1.;
   wnc_n_last = -1, wnc_m_last = -1, wnc_N_last = -1;
   wnc_o_last = -1.;
}


/***********************************************************************
SetAccuracy
***********************************************************************/
void StochasticLib3::SetAccuracy(double accur) {
   // define accuracy of calculations for 
   // WalleniusNCHyp and MultiWalleniusNCHyp
   if (accur < 0.) accur = 0.;
   if (accur > 0.01) accur = 0.01;
   accuracy = accur;
}


注释：


/*************************** stoc3.cpp **********************************
* Author:        Agner Fog
* Date created:  2002-10-02
* Last modified: 2008-11-21
* Project:       stocc.zip
* Source URL:    www.agner.org/random
* 
* Description:
* Non-uniform random number generator functions.
* 
* This file contains source code for the class StochasticLib3 derived
* from StochasticLib1 or StochasticLib2, defined in stocc.h.
* 
* This class implements methods for sampling from the noncentral and extended 
* hypergeometric distributions, as well as the multivariate versions of these.
* 
* Documentation:
* ==============
* The file stocc.h contains class definitions.
* The file stocc.htm contains further instructions.
* The file nchyp.pdf, available from www.agner.org/random/theory contains 
* theoretical description of Wallenius' and Fisher's noncentral hypergeometric
* distributions and the methods used in this code to sample from these.
* The file ran-instructions.pdf contains general instructions.
* 
* Copyright 2002-2008 by Agner Fog. 
* Released under SciPy's license with permission of Agner Fog; see license.txt
*****************************************************************************/

#include <string.h>                    // memcpy function
#include "stocc.h"                     // class definitions
//#include "wnchyppr.cpp"              // calculate Wallenius noncentral hypergeometric probability
//#include "fnchyppr.cpp"              // calculate Fisher's noncentral hypergeometric probability


/******************************************************************************
Methods for class StochasticLib3
******************************************************************************/


/***********************************************************************
Constructor
***********************************************************************/
StochasticLib3::StochasticLib3(int seed) : StochasticLib1(seed) {
   SetAccuracy(1.E-8);                  // set default accuracy
   // Initialize variables
   fnc_n_last = -1, fnc_m_last = -1, fnc_N_last = -1;
   fnc_o_last = -1.;
   wnc_n_last = -1, wnc_m_last = -1, wnc_N_last = -1;
   wnc_o_last = -1.;
}


/***********************************************************************
SetAccuracy
***********************************************************************/
void StochasticLib3::SetAccuracy(double accur) {
   // define accuracy of calculations for 
   // WalleniusNCHyp and MultiWalleniusNCHyp
   if (accur < 0.) accur = 0.;
   if (accur > 0.01) accur = 0.01;
   accuracy = accur;
}
// 定义 StochasticLib3 类中的 WalleniusNCHyp 方法，计算 Wallenius 非中心超几何分布的随机变量
int32_t StochasticLib3::WalleniusNCHyp(int32_t n, int32_t m, int32_t N, double odds) {
   /*
   这个函数生成一个具有 Wallenius 非中心超几何分布的随机变量。

   Wallenius 非中心超几何分布描述了从一个包含红色和白色球的罐中无放回抽取球时的分布，考虑了偏差。

   我们定义球的权重，使得取出某个球的概率与其权重成比例。odds 是归一化的赔率比：odds = weight(red) / weight(white)。
   如果所有球的权重相同，即 odds = 1，则得到超几何分布。

   n 是抽取的球的数量，
   m 是罐中红色球的数量，
   N 是罐中总球的数量，
   odds 是赔率比，
   返回值是抽取到的红色球的数量。

   根据参数的不同，实现了四种不同的计算方法。该函数根据参数决定使用哪种方法。
   */

   // 检查参数
   if (n >= N || m >= N || n <= 0 || m <= 0 || odds <= 0.) {
      // 简单情况的处理
      if (n == 0 || m == 0) return 0;
      if (m == N) return n;
      if (n == N) return m;
      if (odds == 0.) {
         if (n > N-m) FatalError("Not enough items with nonzero weight in function WalleniusNCHyp");
         return 0;
      }
      // 非法参数
      FatalError("Parameter out of range in function WalleniusNCHyp");
   }

   if (odds == 1.) {
      // 如果 odds == 1，使用超几何分布函数
      return Hypergeometric(n, m, N);
   }

   if (n < 30) {
      // 如果 n < 30，使用 Urn 模型进行抽样
      return WalleniusNCHypUrn(n, m, N, odds);
   }

   if (double(n) * N < 10000) {
      // 如果 n * N < 10000，使用表格方法进行计算
      return WalleniusNCHypTable(n, m, N, odds);
   }

   // 否则，使用比例均匀法进行计算
   return WalleniusNCHypRatioOfUnifoms(n, m, N, odds);
   // 决定是否在 WalleniusNCHypRatioOfUnifoms 内部使用 NoncentralHypergeometricInversion，根据计算的方差来确定。
}


/***********************************************************************
WalleniusNCHyp 的子函数
***********************************************************************/

int32_t StochasticLib3::WalleniusNCHypUrn(int32_t n, int32_t m, int32_t N, double odds) {
   // 通过模拟 Urn 模型抽样 Wallenius 非中心超几何分布
   int32_t x;         // 抽样结果
   int32_t m2;        // 罐中第二种颜色的物品数量
   double mw1, mw2;   // 颜色 1 或 2 的球的总权重

   x = 0;  
   m2 = N - m;
   mw1 = m * odds;  
   mw2 = m2;

   do {
      if (Random() * (mw1 + mw2) < mw1) {
         x++;  
         m--;
         if (m == 0) break;
         mw1 = m * odds;
      }
      else {
         m2--;
         if (m2 == 0) {
            x += n - 1; 
            break;
         }
         mw2 = m2;
      }
   } while (--n);

   return x;
}
int32_t StochasticLib3::WalleniusNCHypTable (int32_t n, int32_t m, int32_t N, double odds) {
   // Sampling from Wallenius noncentral hypergeometric distribution 
   // using chop-down search from a table created by recursive calculation.
   // This method is fast when n is low or when called repeatedly with
   // the same parameters.

   int32_t x2;                          // upper x limit for table
   int32_t x;                           // sample
   double u;                            // uniform random number
   int success;                         // table long enough

   if (n != wnc_n_last || m != wnc_m_last || N != wnc_N_last || odds != wnc_o_last) {
      // set-up: This is done only when parameters have changed
      wnc_n_last = n;  wnc_m_last = m;  wnc_N_last = N;  wnc_o_last = odds;

      CWalleniusNCHypergeometric wnch(n,m,N,odds);   // make object for calculation
      success = wnch.MakeTable(wall_ytable, WALL_TABLELENGTH, &wall_x1, &x2); // make table of probability values
      if (success) {
         wall_tablen = x2 - wall_x1 + 1;         // table long enough. remember length
      }
      else {
         wall_tablen = 0;                        // remember failure
      }
   }

   if (wall_tablen == 0) {
      // table not long enough. Use another method
      return WalleniusNCHypRatioOfUnifoms(n,m,N,odds);
   }

   while (1) {                                   // repeat in the rare case of failure
      u = Random();                              // uniform variate to convert
      for (x=0; x<wall_tablen; x++) {            // chop-down search
         u -= wall_ytable[x];
         if (u < 0.) return x + wall_x1;         // value found
      }
   }

   #if 0 // use rejection in x-domain
      if (xi == wnc_mode) break;                 // accept      
      f = wnch.probability(xi);                  // function value
      if (f > wnc_k * u * u) {
         break;                                  // acceptance
      }
   #else // use rejection in t-domain (this is faster)
      double hx, s2, xma2;                       // compute h(x)
      s2 = wnc_h * 0.5;  s2 *= s2; 
      xma2 = xi - (wnc_a-0.5);
      xma2 *= xma2;
      hx = (s2 >= xma2) ? 1. : s2 / xma2;
      // rejection in t-domain implemented in CWalleniusNCHypergeometric::BernouilliH
      if (wnch.BernouilliH(xi, hx * wnc_k * 1.01, u * u * wnc_k  * 1.01, this)) {
         break;                                  // acceptance
      }
   #endif      
   }                                             // rejection
   return xi;
}
// 定义函数 StochasticLib3::WalleniusNCHypInversion，计算 Wallenius 非中心超几何分布的随机抽样
int32_t StochasticLib3::WalleniusNCHypInversion (int32_t n, int32_t m, int32_t N, double odds) {
   // 使用从均值开始的自下而上搜索和切分技术，抽样 Wallenius 非中心超几何分布
   // 当方差较小时，此方法比拒绝法更快。
   int32_t wall_x1, x2;                          // 搜索值
   int32_t xmin, xmax;                           // x 的限制范围
   double   u;                                   // 要转换的均匀随机数
   double   f;                                   // 概率函数值
   double   accura;                              // 绝对精度
   int      updown;                              // 1 = 向下搜索, 2 = 向上搜索, 3 = 同时向上向下搜索

   // 创建两个用于计算均值和概率的对象。
   // 由于它们针对连续的 x 值进行了优化，因此具有相同参数的两个对象更有效。
   CWalleniusNCHypergeometric wnch1(n, m, N, odds, accuracy);
   CWalleniusNCHypergeometric wnch2(n, m, N, odds, accuracy);

   accura = accuracy * 0.01;
   if (accura > 1E-7) accura = 1E-7;             // 设置绝对精度

   wall_x1 = (int32_t)(wnch1.mean());            // 从均值的 floor 和 ceil 开始
   x2 = wall_x1 + 1;
   xmin = m+n-N; if (xmin<0) xmin = 0;           // 计算限制范围
   xmax = n;     if (xmax>m) xmax = m;
   updown = 3;                                   // 同时向上和向下搜索

   while(1) {                                    // 循环直到接受（通常只执行一次）
      u = Random();                              // 获得一个均匀随机数
      while (updown) {                           // 搜索循环
         if (updown & 1) {                       // 向下搜索
            if (wall_x1 < xmin) {
               updown &= ~1;}                    // 停止向下搜索
            else {
               f = wnch1.probability(wall_x1);
               u -= f;                           // 减去概率直到为零
               if (u <= 0.) return wall_x1;
               wall_x1--;
               if (f < accura) updown &= ~1;     // 停止向下搜索
            }
         }
         if (updown & 2) {                       // 向上搜索
            if (x2 > xmax) {
               updown &= ~2;                     // 停止向上搜索
            }
            else {
               f = wnch2.probability(x2);
               u -= f;                           // 减去概率直到为零
               if (u <= 0.) return x2;
               x2++;
               if (f < accura) updown &= ~2;     // 停止向上搜索
            }
         }
      }
   }
}


这段代码实现了通过 Wallenius 非中心超几何分布的倒推算法进行随机抽样。
// 这个函数生成具有多元补充Wallenius非中心超几何分布的随机变量向量。
// 详细信息请参见MultiWalleniusNCHyp。
void StochasticLib3::MultiComplWalleniusNCHyp (
    int32_t * destination, int32_t * source, double * weights, int32_t n, int colors) {
   // reciprocal weights
   double rweights[MAXCOLORS];
   // balls sampled
   int32_t sample[MAXCOLORS];
   // weight
   double w;
   // total number of balls
   int32_t N;
   // color index
   int i;

   // 计算逆权重并计算总数N
   for (i=0, N=0; i<colors; i++) {
      w = weights[i];
      // 如果权重为零，则发生致命错误
      if (w == 0) FatalError("Zero weight in function MultiComplWalleniusNCHyp");
      rweights[i] = 1. / w;
      N += source[i];
   }

   // 使用多元Wallenius非中心超几何分布
   MultiWalleniusNCHyp(sample, source, rweights, N - n, colors);

   // 补充分布 = 没有被取走的球
   for (i=0; i<colors; i++) {
      destination[i] = source[i] - sample[i];
   }
}
// 生成 Fisher's 非中心超几何分布的随机变量
int32_t StochasticLib3::FishersNCHyp(int32_t n, int32_t m, int32_t N, double odds) {
   /*
   这个函数生成 Fisher's 非中心超几何分布的随机变量。

   这个分布类似于 Wallenius 非中心超几何分布，有时这两个分布会混淆。关于这个分布的更详细解释
   可以在 MultiFishersNCHyp 下面找到，关于更多文档请参阅 nchyp.pdf，可从 www.agner.org/random 获取。

   当参数较小时，使用从零开始的递归搜索进行反转，当参数过大时或者会导致溢出时，使用比例-均匀拒绝法。
   */   
   int32_t fak, addd;                  // 用于撤销变换的变量
   int32_t x;                          // 结果变量

   // 检查参数是否有效
   if (n > N || m > N || n < 0 || m < 0 || odds <= 0.) {
      if (odds == 0.) {
         if (n > N - m) FatalError("Not enough items with nonzero weight in function FishersNCHyp");
         return 0;
      }
      FatalError("Parameter out of range in function FishersNCHyp");
   }

   if (odds == 1.) {
      // 如果 odds == 1 使用超几何函数
      return Hypergeometric(n, m, N);
   }

   // 对称变换
   fak = 1;  addd = 0;
   if (m > N/2) {
      // 反转 m
      m = N - m;
      fak = -1;  addd = n;
   }

   if (n > N/2) {
      // 反转 n
      n = N - n;
      addd += fak * m;  fak = - fak;
   }

   if (n > m) {
      // 交换 n 和 m
      x = n;  n = m;  m = x;
   }

   // 只有一个可能结果的情况到此结束
   if (n == 0 || odds == 0.) return addd;

   if (fak == -1) {
      // 如果进行反转，使用倒数 odds
      odds = 1. / odds;
   }

   // 选择方法
   if (n < 30 && N < 1024 && odds > 1.E-5 && odds < 1.E5) {
      // 使用递归搜索的反转法
      x = FishersNCHypInversion(n, m, N, odds);
   }
   else {
      // 使用比例-均匀拒绝法
      x = FishersNCHypRatioOfUnifoms(n, m, N, odds);
   }

   // 撤销对称变换  
   return x * fak + addd;
}
/*
   Subfunction for FishersNCHyp distribution.
   Implements Fisher's noncentral hypergeometric distribution by inversion 
   method, using chop-down search starting at zero.

   Valid only for 0 <= n <= m <= N/2.
   Without overflow check the parameters must be limited to n < 30, N < 1024,
   and 1.E-5 < odds < 1.E5. This limitation is acceptable because this method 
   is slow for higher n.

   The execution time of this function grows with n.

   See the file nchyp.pdf for theoretical explanation.
*/ 
int32_t StochasticLib3::FishersNCHypInversion (int32_t n, int32_t m, int32_t N, double odds) {
   int32_t x;                          // x value
   int32_t L;                          // derived parameter
   double f;                           // scaled function value 
   double sum;                         // scaled sum of function values
   double a1, a2, b1, b2, f1, f2;      // factors in recursive calculation
   double u;                           // uniform random variate

   L = N - m - n;

   if (n != fnc_n_last || m != fnc_m_last || N != fnc_N_last || odds != fnc_o_last) {
      // parameters have changed. set-up
      fnc_n_last = n; fnc_m_last = m; fnc_N_last = N; fnc_o_last = odds;

      // f(0) is set to an arbitrary value because it cancels out.
      // A low value is chosen to avoid overflow.
      fnc_f0 = 1.E-100;

      // calculate summation of e(x), using the formula:
      // f(x) = f(x-1) * (m-x+1)*(n-x+1)*odds / (x*(L+x))
      // All divisions are avoided by scaling the parameters
      sum = f = fnc_f0;  fnc_scale = 1.;
      a1 = m;  a2 = n;  b1 = 1;  b2 = L + 1;
      for (x = 1; x <= n; x++) {
         f1 = a1 * a2 * odds;
         f2 = b1 * b2;
         a1--;  a2--;  b1++;  b2++;
         f *= f1;
         sum *= f2;
         fnc_scale *= f2;
         sum += f;
         // overflow check. not needed if parameters are limited:
         // if (sum > 1E100) {sum *= 1E-100; f *= 1E-100; fnc_scale *= 1E-100;}
      }
      fnc_f0 *= fnc_scale;
      fnc_scale = sum;
      // now f(0) = fnc_f0 / fnc_scale.
      // We are still avoiding all divisions by saving the scale factor
   }

   // uniform random
   u = Random() * fnc_scale;

   // recursive calculation:
   // f(x) = f(x-1) * (m-x+1)*(n-x+1)*odds / (x*(L+x))
   f = fnc_f0;  x = 0;  a1 = m;  a2 = n;  b1 = 0;  b2 = L;
   do {
      u -= f;
      if (u <= 0) break;
      x++;  b1++;  b2++;
      f *= a1 * a2 * odds;
      u *= b1 * b2;
      // overflow check. not needed if parameters are limited:
      // if (u > 1.E100) {u *= 1E-100;  f *= 1E-100;}
      a1--;  a2--;
   }
   while (x < n);
   return x;
}
int32_t StochasticLib3::FishersNCHypRatioOfUnifoms (int32_t n, int32_t m, int32_t N, double odds) {
   /* 
   Subfunction for FishersNCHyp distribution. 
   Valid for 0 <= n <= m <= N/2, odds != 1

   Fisher's noncentral hypergeometric distribution by ratio-of-uniforms 
   rejection method.

   The execution time of this function is almost independent of the parameters.
   */ 
   int32_t L;                          // N-m-n
   int32_t mode;                       // mode
   double mean;                        // mean
   double variance;                    // variance
   double x;                           // real sample
   int32_t k;                          // integer sample
   double u;                           // uniform random
   double lf;                          // ln(f(x))
   double AA, BB, g1, g2;              // temporary

   L = N - m - n;

   if (n != fnc_n_last || m != fnc_m_last || N != fnc_N_last || odds != fnc_o_last) {
      // parameters have changed. set-up
      fnc_n_last = n;  fnc_m_last = m;  fnc_N_last = N;  fnc_o_last = odds;

      // find approximate mean
      AA = (m+n)*odds+L; BB = sqrt(AA*AA - 4*odds*(odds-1)*m*n);
      mean = (AA-BB)/(2*(odds-1));

      // find approximate variance
      AA = mean * (m-mean); BB = (n-mean)*(mean+L);
      variance = N*AA*BB/((N-1)*(m*BB+(n+L)*AA));

      // compute log(odds)
      fnc_logb = log(odds);

      // find center and width of hat function
      fnc_a = mean + 0.5;
      fnc_h = 1.028 + 1.717*sqrt(variance+0.5) + 0.032*fabs(fnc_logb);

      // find safety bound
      fnc_bound = (int32_t)(mean + 4.0 * fnc_h);
      if (fnc_bound > n) fnc_bound = n;

      // find mode
      mode = (int32_t)(mean);
      g1 =(double)(m-mode)*(n-mode)*odds;
      g2 =(double)(mode+1)*(L+mode+1);
      if (g1 > g2 && mode < n) mode++;

      // value at mode to scale with:
      fnc_lfm = mode * fnc_logb - fc_lnpk(mode, L, m, n);
   }

   while(1) {
      u = Random();
      if (u == 0) continue;                      // avoid divide by 0
      x = fnc_a + fnc_h * (Random()-0.5)/u;
      if (x < 0. || x > 2E9) continue;           // reject, avoid overflow
      k = (int32_t)(x);                          // truncate
      if (k > fnc_bound) continue;               // reject if outside safety bound
      lf = k*fnc_logb - fc_lnpk(k,L,m,n) - fnc_lfm;// compute function value
      if (u * (4.0 - u) - 3.0 <= lf) break;      // lower squeeze accept
      if (u * (u-lf) > 1.0) continue;            // upper squeeze reject
      if (2.0 * log(u) <= lf) break;             // final acceptance
   }
   return k;
}



/***********************************************************************
Multivariate Fisher's noncentral hypergeometric distribution
***********************************************************************/
void StochasticLib3::MultiFishersNCHyp (int32_t * destination, 
```