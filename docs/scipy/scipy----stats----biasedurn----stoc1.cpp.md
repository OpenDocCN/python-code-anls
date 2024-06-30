# `D:\src\scipysrc\scipy\scipy\stats\biasedurn\stoc1.cpp`

```
/*************************** stoc1.cpp **********************************
* Author:        Agner Fog
* Date created:  2002-01-04
* Last modified: 2008-11-30
* Project:       stocc.zip
* Source URL:    www.agner.org/random
*
* Description:
* Non-uniform random number generator functions.
*
* This file contains source code for the class StochasticLib1 defined in stocc.h.
*
* Documentation:
* ==============
* The file stocc.h contains class definitions.
* The file stocc.htm contains further instructions.
* The file distrib.pdf contains definitions of the statistic distributions.
* The file sampmet.pdf contains theoretical descriptions of the methods used
* for sampling from these distributions.
* The file ran-instructions.pdf contains general instructions.
*
* Copyright 2002-2008 by Agner Fog. 
* Released under SciPy's license with permission of Agner Fog; see license.txt
*****************************************************************************/

#include "stocc.h"     // class definition


/***********************************************************************
constants
***********************************************************************/
const double SHAT1 = 2.943035529371538573;    // 8/e
const double SHAT2 = 0.8989161620588987408;   // 3-sqrt(12/e)


/***********************************************************************
Log factorial function
***********************************************************************/
double LnFac(int32_t n) {
   // log factorial function. gives natural logarithm of n!

   // define constants
   static const double                 // coefficients in Stirling approximation     
      C0 =  0.918938533204672722,      // ln(sqrt(2*pi))
      C1 =  1./12., 
      C3 = -1./360.;
   // C5 =  1./1260.,                  // use r^5 term if FAK_LEN < 50
   // C7 = -1./1680.;                  // use r^7 term if FAK_LEN < 20
   // static variables
   static double fac_table[FAK_LEN];   // table of ln(n!):
   static int initialized = 0;         // remember if fac_table has been initialized

   if (n < FAK_LEN) {
      if (n <= 1) {
         if (n < 0) FatalError("Parameter negative in LnFac function");  
         return 0;
      }
      if (!initialized) {              // first time. Must initialize table
         // make table of ln(n!)
         double sum = fac_table[0] = 0.;
         for (int i=1; i<FAK_LEN; i++) {
            sum += log(double(i));
            fac_table[i] = sum;
         }
         initialized = 1;
      }
      return fac_table[n];
   }
   // not found in table. use Stirling approximation
   double  n1, r;
   n1 = n;  r  = 1. / n1;
   return (n1 + 0.5)*log(n1) - n1 + C0 + r*(C1 + r*r*C3);
}


/***********************************************************************
Constructor
***********************************************************************/
StochasticLib1::StochasticLib1 (int seed)
: STOC_BASE(seed) {
   // 初始化各种分布的变量
   normal_x2_valid = 0;  // 正态分布相关变量初始化
   hyp_n_last = hyp_m_last = hyp_N_last = -1; // 超几何分布参数的最后值
   pois_L_last = -1.;                         // 泊松分布参数的最后值
   bino_n_last = -1;  bino_p_last = -1.;      // 二项分布参数的最后值
}


/***********************************************************************
超几何分布
***********************************************************************/
int32_t StochasticLib1::Hypergeometric (int32_t n, int32_t m, int32_t N) {
   /*
   该函数生成符合超几何分布的随机变量。这种分布描述了从一个包含两种颜色球的罐中无放回地抽取球的情况。
   n 是抽取的球数，m 是罐中红球的数量，N 是罐中总球数。函数返回抽取到的红球的数量。

   当参数较小时，使用从众数开始的切分搜索方法进行反演；当参数较大时或前一种方法过慢或可能溢出时，使用比值-均匀变量法。
   */   

   int32_t fak, addd;                    // 用于撤销变换的变量
   int32_t x;                            // 结果

   // 检查参数是否有效
   if (n > N || m > N || n < 0 || m < 0) {
      FatalError("超几何分布函数中的参数超出范围");}
   
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
   // 只有一个可能结果的情况到此为止
   if (n == 0)  return addd;

   //------------------------------------------------------------------
   //                 选择方法
   //------------------------------------------------------------------
   if (N > 680 || n > 70) {
      // 使用比值-均匀变量法
      x = HypRatioOfUnifoms (n, m, N);
   }
   else {
      // 反演方法，使用从众数开始的切分搜索
      x = HypInversionMod (n, m, N);
   }
   // 撤销对称变换  
   return x * fak + addd;
}


/***********************************************************************
超几何分布使用的子函数
***********************************************************************/

}
/*
Subfunction for computing a sample from the Hypergeometric distribution using
the ratio-of-uniforms rejection method. This method is valid for 0 < n <= m <= N/2.

The computation time varies depending on parameter values, particularly whether
they fall within the range where the LnFac function is precomputed.

Reference: E. Stadlober, "The ratio of uniforms approach for generating
discrete random variates". Journal of Computational and Applied Mathematics,
vol. 31, no. 1, 1990, pp. 181-189.
*/
int32_t StochasticLib1::HypRatioOfUnifoms (int32_t n, int32_t m, int32_t N) {
   int32_t L;                          // N-m-n
   int32_t mode;                       // mode
   int32_t k;                          // integer sample
   double x;                           // real sample
   double rNN;                         // 1/(N*(N+2))
   double my;                          // mean
   double var;                         // variance
   double u;                           // uniform random
   double lf;                          // ln(f(x))

   L = N - m - n;                     // Calculate N-m-n
   if (hyp_N_last != N || hyp_m_last != m || hyp_n_last != n) {
      hyp_N_last = N;  hyp_m_last = m;  hyp_n_last = n;         // Set-up
      rNN = 1. / ((double)N*(N+2));                               // 1/(N*(N+2))
      my = (double)n * m * rNN * (N+2);                           // mean = n*m/N
      mode = (int32_t)(double(n+1) * double(m+1) * rNN * N);      // mode = floor((n+1)*(m+1)/(N+2))
      var = (double)n * m * (N-m) * (N-n) / ((double)N*N*(N-1));  // variance
      hyp_h = sqrt(SHAT1 * (var+0.5)) + SHAT2;                    // hat width
      hyp_a = my + 0.5;                                           // hat center
      hyp_fm = fc_lnpk(mode, L, m, n);                            // maximum
      hyp_bound = (int32_t)(hyp_a + 4.0 * hyp_h);                 // safety-bound
      if (hyp_bound > n) hyp_bound = n;                           // cap bound at n
   }    
   while(1) {
      u = Random();                              // uniform random number
      if (u == 0) continue;                      // avoid division by 0
      x = hyp_a + hyp_h * (Random()-0.5) / u;    // generate hat distribution
      if (x < 0. || x > 2E9) continue;           // reject, avoid overflow
      k = (int32_t)x;
      if (k > hyp_bound) continue;               // reject if outside range
      lf = hyp_fm - fc_lnpk(k,L,m,n);            // ln(f(k))
      if (u * (4.0 - u) - 3.0 <= lf) break;      // lower squeeze accept
      if (u * (u-lf) > 1.0) continue;            // upper squeeze reject
      if (2.0 * log(u) <= lf) break;             // final acceptance
   }
   return k;                                      // return the generated sample
}


/*
Subfunction used by the HypRatioOfUnifoms function to compute ln(f(k))
for hypergeometric and Fisher's noncentral hypergeometric distributions.
*/
double StochasticLib1::fc_lnpk(int32_t k, int32_t L, int32_t m, int32_t n) {
   return(LnFac(k) + LnFac(m - k) + LnFac(n - k) + LnFac(L + k));   // compute ln(f(k))
}


/*
Conditional compilation guard to exclude this code when building the R interface.
Not needed for the R interface.
*/
#ifndef R_BUILD
/***********************************************************************
Multivariate hypergeometric distribution
***********************************************************************/
void StochasticLib1::MultiHypergeometric (int32_t * destination, int32_t * source, int32_t n, int colors) {
   /*
   This function generates a vector of random variates, each with the
   hypergeometric distribution.
   
   The multivariate hypergeometric distribution is the distribution you 
   get when drawing balls from an urn with more than two colors, without
   replacement.
   
   Parameters:
   destination:    An output array to receive the number of balls of each 
                   color. Must have space for at least 'colors' elements.
   source:         An input array containing the number of balls of each 
                   color in the urn. Must have 'colors' elements.
                   All elements must be non-negative.
   n:              The number of balls drawn from the urn.
                   Can't exceed the total number of balls in the urn.
   colors:         The number of possible colors. 
   */
   int32_t sum, x, y;
   int i;
   // Check if n or colors is negative; if so, raise a fatal error
   if (n < 0 || colors < 0) FatalError("Parameter negative in multihypergeo function");
   // If there are no colors, return early as there's no work to be done
   if (colors == 0) return;

   // Compute the total number of balls in the urn
   for (i=0, sum=0; i<colors; i++) { 
      y = source[i];
      // Check if any color has a negative number of balls; if so, raise a fatal error
      if (y < 0) FatalError("Parameter negative in multihypergeo function");
      sum += y;
   }
   // Check if the number of balls drawn (n) exceeds the total number of balls in the urn (sum); if so, raise a fatal error
   if (n > sum) FatalError("n > sum in multihypergeo function");

   // Generate random variates according to the hypergeometric distribution for each color except the last one
   for (i=0; i<colors-1; i++) { 
      y = source[i];
      // Generate a random variate using the Hypergeometric distribution
      x = Hypergeometric(n, y, sum);
      // Update the number of balls drawn and the total number of balls remaining
      n -= x; sum -= y;
      // Store the generated variate in the destination array
      destination[i] = x;
   }
   // Handle the last color separately to ensure all balls are accounted for
   destination[i] = n;
}


/***********************************************************************
Poisson distribution
***********************************************************************/
int32_t StochasticLib1::Poisson (double L) {
   /*
   This function generates a random variate with the poisson distribution.

   Uses inversion by chop-down method for L < 17, and ratio-of-uniforms
   method for L >= 17.

   For L < 1.E-6 numerical inaccuracy is avoided by direct calculation.
   */

   //------------------------------------------------------------------
   //                 choose method
   //------------------------------------------------------------------
   // 根据参数 L 的大小选择不同的生成泊松分布随机变量的方法

   if (L < 17) {
      if (L < 1.E-6) {
         if (L == 0) return 0;
         if (L < 0) FatalError("Parameter negative in poisson function");

         //--------------------------------------------------------------
         // calculate probabilities
         //--------------------------------------------------------------
         // 对于极小的 L，计算 x = 1 和 x = 2 的概率（忽略更高的 x 值）。
         // 使用这种方法是为了避免其他方法中的数值不准确性。
         //--------------------------------------------------------------
         return PoissonLow(L);
      }    
      else {
         //--------------------------------------------------------------
         // inversion method
         //--------------------------------------------------------------
         // 使用反演方法生成泊松分布随机变量
         // 此方法的计算时间随着 L 的增大而增加。
         // 当 L > 80 时会溢出。
         //--------------------------------------------------------------
         return PoissonInver(L);
      }
   }      
   else {
      if (L > 2.E9) FatalError("Parameter too big in poisson function");

      //----------------------------------------------------------------
      // ratio-of-uniforms method
      //----------------------------------------------------------------
      // 使用比例均匀方法生成泊松分布随机变量
      // 此方法的计算时间与 L 无关。
      // 在其他方法速度较慢时使用。
      //----------------------------------------------------------------
      return PoissonRatioUniforms(L);
   }
}


/***********************************************************************
Subfunctions used by poisson
***********************************************************************/
int32_t StochasticLib1::PoissonLow(double L) {
   /*
   This subfunction generates a random variate with the poisson 
   distribution for extremely low values of L.

   The method is a simple calculation of the probabilities of x = 1
   and x = 2. Higher values are ignored.

   The reason for using this method is to avoid the numerical inaccuracies 
   in other methods.
   */   
   // 对于极小的 L 值，使用此子函数计算泊松分布随机变量
   // 方法简单地计算 x = 1 和 x = 2 的概率。更高的值被忽略。
   // 使用这种方法是为了避免其他方法中的数值不准确性。
   
   double d, r;
   d = sqrt(L);
   if (Random() >= d) return 0;
   r = Random() * d;
   if (r > L * (1.-L)) return 0;
   if (r > 0.5 * L*L * (1.-L)) return 1;
   return 2;
}
// 采用倒推方法（PIN）生成泊松分布的随机变量
// 根据指数增长时间与L有关，当L > 80时可能溢出
int32_t StochasticLib1::PoissonInver(double L) {
   // 安全边界，必须大于 L + 8*sqrt(L)
   const int bound = 130;
   // 均匀随机数
   double r;
   // 函数值
   double f;
   // 返回值
   int32_t x;

   // 如果 L 不等于上次保存的 pois_L_last
   if (L != pois_L_last) {
      // 设置初始化
      pois_L_last = L;
      // 计算 f(0)，即 x=0 的概率
      pois_f0 = exp(-L);
   }

   // 无限循环，生成泊松分布随机变量
   while (1) {  
      // 生成均匀随机数 r
      r = Random();
      // 初始化 x 为 0，f 为 pois_f0
      x = 0;
      f = pois_f0;

      // 递归计算泊松分布的函数值 f(x) = f(x-1) * L / x
      do {
         r -= f;
         // 如果 r <= 0，则返回 x
         if (r <= 0) return x;
         x++;
         f *= L;
         r *= x;  // 相当于 f /= x
      } while (x <= bound);  // 当 x <= bound 时继续循环
   }
}


// 使用比例-均匀拒绝法（PRUAt）生成泊松分布的随机变量
// 执行时间与 L 无关，但是取决于 ln(n!) 是否在表格内
// 参考文献：E. Stadlober: "The ratio of uniforms approach for generating
// discrete random variates". Journal of Computational and Applied Mathematics,
// vol. 31, no. 1, 1990, pp. 181-189.
int32_t StochasticLib1::PoissonRatioUniforms(double L) {
   // 均匀随机数
   double u;
   // ln(f(x))
   double lf;
   // 实数样本
   double x;
   // 整数样本
   int32_t k;

   // 如果 pois_L_last 不等于 L
   if (pois_L_last != L) {
      // 设置初始化
      pois_L_last = L;
      // hat center
      pois_a = L + 0.5;
      // mode
      int32_t mode = (int32_t)L;
      // pois_g
      pois_g  = log(L);
      // value at mode
      pois_f0 = mode * pois_g - LnFac(mode);
      // hat width
      pois_h = sqrt(SHAT1 * (L+0.5)) + SHAT2;
      // safety-bound
      pois_bound = (int32_t)(pois_a + 6.0 * pois_h);
   }

   // 无限循环，生成泊松分布的随机变量
   while(1) {
      // 生成均匀随机数 u
      u = Random();
      // 如果 u 等于 0，则继续循环，避免除以 0
      if (u == 0) continue;
      // 生成样本 x
      x = pois_a + pois_h * (Random() - 0.5) / u;
      // 如果 x 不在有效范围内，则继续循环
      if (x < 0 || x >= pois_bound) continue;
      // 将 x 转换为整数 k
      k = (int32_t)(x);
      // 计算 ln(f(x))
      lf = k * pois_g - LnFac(k) - pois_f0;
      // 快速接受
      if (lf >= u * (4.0 - u) - 3.0) break;
      // 快速拒绝
      if (u * (u - lf) > 1.0) continue;
      // 最终接受
      if (2.0 * log(u) <= lf) break;
   }

   // 返回生成的泊松分布随机变量 k
   return k;
}
/***********************************************************************/
int32_t StochasticLib1::Binomial (int32_t n, double p) {
   /*
   This function generates a random variate with the binomial distribution.

   Uses inversion by chop-down method for n*p < 35, and ratio-of-uniforms
   method for n*p >= 35.

   For n*p < 1.E-6 numerical inaccuracy is avoided by poisson approximation.
   */
   int inv = 0;                        // invert
   int32_t x;                          // result
   double np = n * p;

   if (p > 0.5) {                      // faster calculation by inversion
      p = 1. - p;  inv = 1;
   }
   if (n <= 0 || p <= 0) {
      if (n == 0 || p == 0) {
         return inv * n;  // only one possible result
      }
      // error exit
      FatalError("Parameter out of range in binomial function"); 
   }

   //------------------------------------------------------------------
   //                 choose method
   //------------------------------------------------------------------
   if (np < 35.) {
      if (np < 1.E-6) {
         // Poisson approximation for extremely low np
         x = PoissonLow(np);
      }
      else {
         // inversion method, using chop-down search from 0
         x = BinomialInver(n, p);
      }
   }  
   else {
      // ratio of uniforms method
      x = BinomialRatioOfUniforms(n, p);
   }
   if (inv) {
      x = n - x;      // undo inversion
   }
   return x;
}


/***********************************************************************
Subfunctions used by binomial
***********************************************************************/

int32_t StochasticLib1::BinomialInver (int32_t n, double p) {
   /* 
   Subfunction for Binomial distribution. Assumes p < 0.5.

   Uses inversion method by search starting at 0.

   Gives overflow for n*p > 60.

   This method is fast when n*p is low. 
   */   
   double f0, f, q; 
   int32_t bound;
   double pn, r, rc; 
   int32_t x, n1, i;

   // f(0) = probability of x=0 is (1-p)^n
   // fast calculation of (1-p)^n
   f0 = 1.;  pn = 1.-p;  n1 = n;
   while (n1) {
      if (n1 & 1) f0 *= pn;
      pn *= pn;  n1 >>= 1;
   }
   // calculate safety bound
   rc = (n + 1) * p;
   bound = (int32_t)(rc + 11.0*(sqrt(rc) + 1.0));
   if (bound > n) bound = n; 
   q = p / (1. - p);

   while (1) {
      r = Random();
      // recursive calculation: f(x) = f(x-1) * (n-x+1)/x*p/(1-p)
      f = f0;  x = 0;  i = n;
      do {
         r -= f;
         if (r <= 0) return x;
         x++;
         f *= q * i;
         r *= x;       // it is faster to multiply r by x than dividing f by x
         i--;
      }
      while (x <= bound);
   }
}
/*
Subfunction for Binomial distribution. Assumes p < 0.5.

Uses the Ratio-of-Uniforms rejection method.

The computation time hardly depends on the parameters, except that it matters
a lot whether parameters are within the range where the LnFac function is 
tabulated.

Reference: E. Stadlober: "The ratio of uniforms approach for generating
discrete random variates". Journal of Computational and Applied Mathematics,
vol. 31, no. 1, 1990, pp. 181-189.
*/
int32_t StochasticLib1::BinomialRatioOfUniforms(int32_t n, double p) {
    double u;                           // uniform random
    double q1;                          // 1-p
    double np;                          // n*p
    double var;                         // variance
    double lf;                          // ln(f(x))
    double x;                           // real sample
    int32_t k;                          // integer sample

    if(bino_n_last != n || bino_p_last != p) {    // Set_up
        bino_n_last = n;
        bino_p_last = p;
        q1 = 1.0 - p;
        np = n * p;
        bino_mode = (int32_t)(np + p);             // mode
        bino_a = np + 0.5;                         // hat center
        bino_r1 = log(p / q1);
        bino_g = LnFac(bino_mode) + LnFac(n - bino_mode);
        var = np * q1;                             // variance
        bino_h = sqrt(SHAT1 * (var + 0.5)) + SHAT2; // hat width
        bino_bound = (int32_t)(bino_a + 6.0 * bino_h);// safety-bound
        if (bino_bound > n) bino_bound = n;        // safety-bound
    }

    while (1) {                                   // rejection loop
        u = Random();
        if (u == 0) continue;                      // avoid division by 0
        x = bino_a + bino_h * (Random() - 0.5) / u;
        if (x < 0. || x > bino_bound) continue;    // reject, avoid overflow
        k = (int32_t)x;                            // truncate
        lf = (k - bino_mode) * bino_r1 + bino_g - LnFac(k) - LnFac(n - k); // ln(f(k))
        if (u * (4.0 - u) - 3.0 <= lf) break;      // lower squeeze accept
        if (u * (u - lf) > 1.0) continue;          // upper squeeze reject
        if (2.0 * log(u) <= lf) break;             // final acceptance
    }
    return k;
}
void StochasticLib1::Multinomial (int32_t * destination, double * source, int32_t n, int colors) {
   /*
   This function generates a vector of random variates, each with the
   multinomial distribution.

   The multinomial distribution is obtained from the probability vector
   'source' representing probabilities or fractions of each color in an
   urn with 'colors' different colors.

   Parameters:
   destination:    An output array to receive the count of each color.
                   Must have space for at least 'colors' elements.
   source:         An input array containing the probabilities or fractions
                   of each color in the urn. Must have 'colors' elements.
                   All elements must be non-negative. The sum does not have
                   to be 1, but it must be positive.
   n:              The number of balls drawn from the urn.
   colors:         The number of possible colors.
   */
   double s, sum;
   int32_t x;
   int i;
   if (n < 0 || colors < 0) FatalError("Parameter negative in multinomial function");
   if (colors == 0) return;

   // compute sum of probabilities
   for (i=0, sum=0; i<colors; i++) { 
      s = source[i];
      if (s < 0) FatalError("Parameter negative in multinomial function");
      sum += s;
   }
   if (sum == 0 && n > 0) FatalError("Zero sum in multinomial function");

   for (i=0; i<colors-1; i++) { 
      // generate output by calling binomial (colors-1) times
      s = source[i];
      if (sum <= s) {
         // ensure x equals n when sum is zero to avoid division by zero
         // and prevent s/sum from exceeding 1 due to rounding errors
         x = n;
      }
      else {    
         // calculate the number of balls of this color using the binomial distribution
         x = Binomial(n, s/sum);
      }
      n -= x; sum -= s;
      destination[i] = x;
   }
   // last element receives the remaining number of balls
   destination[i] = n;
}


void StochasticLib1::Multinomial (int32_t * destination, int32_t * source, int32_t n, int colors) {
   // same as above, but for integer probabilities in 'source'
   int32_t x, p, sum;
   int i;
   if (n < 0 || colors < 0) FatalError("Parameter negative in multinomial function");
   if (colors == 0) return;

   // compute sum of probabilities
   for (i=0, sum=0; i<colors; i++) { 
      p = source[i];
      if (p < 0) FatalError("Parameter negative in multinomial function");
      sum += p;
   }
   if (sum == 0 && n > 0) FatalError("Zero sum in multinomial function");

   for (i=0; i<colors-1; i++) { 
      // generate output by calling binomial (colors-1) times
      if (sum == 0) {
         // if sum of probabilities is zero, set destination to 0 for this color
         destination[i] = 0; continue;
      }
      p = source[i];
      // calculate the number of balls of this color using the binomial distribution
      x = Binomial(n, (double)p/sum);
      n -= x; sum -= p;
      destination[i] = x;
   }
   // last element receives the remaining number of balls
   destination[i] = n;
}
double StochasticLib1::Normal(double m, double s) {
   // normal distribution with mean m and standard deviation s

   double normal_x1;                   // first random coordinate (normal_x2 is member of class)
   double w;                           // radius

   if (normal_x2_valid) {              // check if we have a valid result from the last call
      normal_x2_valid = 0;
      return normal_x2 * s + m;        // return the cached result adjusted by s and m
   }    

   // generate two normally distributed variates using Box-Muller transformation
   do {
      normal_x1 = 2. * Random() - 1.;   // generate first variate
      normal_x2 = 2. * Random() - 1.;   // generate second variate
      w = normal_x1*normal_x1 + normal_x2*normal_x2;  // compute squared radius
   }
   while (w >= 1. || w < 1E-30);         // repeat if outside acceptable bounds

   w = sqrt(log(w)*(-2./w));            // calculate scaling factor
   normal_x1 *= w;  normal_x2 *= w;     // scale to make normal_x1 and normal_x2 independent normal variates
   normal_x2_valid = 1;                 // mark normal_x2 as valid for the next call
   return normal_x1 * s + m;            // return normal variate adjusted by s and m
}

double StochasticLib1::NormalTrunc(double m, double s, double limit) {
   // Truncated normal distribution
   // The tails are cut off so that the output
   // is in the interval from (m-limit) to (m+limit)

   if (limit < s) FatalError("limit out of range in NormalTrunc function");  // check if limit is valid

   double x;
   do {
      x = Normal(0., s);               // generate a normal variate with mean 0 and standard deviation s
   } while (fabs(x) > limit);          // repeat if the generated value exceeds the limit

   return x + m;                       // return truncated normal variate adjusted by m
}

/***********************************************************************
Bernoulli distribution
***********************************************************************/
int StochasticLib1::Bernoulli(double p) {
   // Bernoulli distribution with parameter p. This function returns 
   // 0 or 1 with probability (1-p) and p, respectively.

   if (p < 0 || p > 1) FatalError("Parameter out of range in Bernoulli function");  // check if p is within valid range

   return Random() < p;                // return 1 with probability p, else return 0
}

/***********************************************************************
Shuffle function
***********************************************************************/
void StochasticLib1::Shuffle(int * list, int min, int n) {
   /*
   This function makes a list of the n numbers from min to min+n-1
   in random order.

   The parameter 'list' must be an array with at least n elements.
   The array index goes from 0 to n-1.

   If you want to shuffle something else than integers then use the 
   integers in list as an index into a table of the items you want to shuffle.
   */

   int i, j, swap;

   // initialize list with numbers from min to min+n-1
   for (i=0, j=min; i<n; i++, j++) list[i] = j;

   // shuffle list using Fisher-Yates algorithm
   for (i=0; i<n-1; i++) {
      j = IRandom(i,n-1);              // generate random index in the remaining elements
      swap = list[j];  list[j] = list[i];  list[i] = swap;  // swap elements at i and j
   }
}

#endif  // ifndef R_BUILD
```