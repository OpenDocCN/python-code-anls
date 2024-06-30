# `D:\src\scipysrc\scipy\scipy\stats\biasedurn\stocc.h`

```
/*****************************   stocc.h   **********************************
* Author:        Agner Fog
* Date created:  2004-01-08
* Last modified: 2013-09-20
* Project:       randomc.h
* Source URL:    www.agner.org/random
*
* Description:
* This file contains function prototypes and class declarations for the C++ 
* library of non-uniform random number generators. Most functions are fast and 
* accurate, even for extreme values of the parameters.
*
*
* functions without classes:
* ==========================
*
* void EndOfProgram(void);
* System-specific exit code. You may modify this to make it fit your
* user interface.
*
* void FatalError(const char * ErrorText);
* Used for outputting error messages from the other functions and classes.
* You may have to modify this function to make it fit your user interface.
*
* double Erf (double x);
* Calculates the error function, which is the integral of the normal distribution.
*
* double LnFac(int32_t n);
* Calculates the natural logarithm of the factorial of n.
*
*
* class StochasticLib1:
* ====================
* This class can be derived from any of the uniform random number generators
* defined in randomc.h. StochasticLib1 provides the following non-uniform random 
* variate generators:
*
* int Bernoulli(double p);
* Bernoulli distribution. Gives 0 or 1 with probability 1-p and p.
*
* double Normal(double m, double s);
* Normal distribution with mean m and standard deviation s.
*
* double NormalTrunc(double m, double s, double limit);
* Truncated normal distribution with tails cut off at m +/- limit
*
* int32_t Poisson (double L);
* Poisson distribution with mean L.
*
* int32_t Binomial (int32_t n, double p);
* Binomial distribution. n trials with probability p.
*
* int32_t Hypergeometric (int32_t n, int32_t m, int32_t N);
* Hypergeometric distribution. Taking n items out N, m of which are colored.
*
* void Multinomial (int32_t * destination, double * source, int32_t n, int colors);
* void Multinomial (int32_t * destination, int32_t * source, int32_t n, int colors);
* Multivariate binomial distribution.
*
* void MultiHypergeometric (int32_t * destination, int32_t * source, int32_t n, int colors);
* Multivariate hypergeometric distribution.
*
* void Shuffle(int * list, int min, int n);
* Shuffle a list of integers.
*
*
* class StochasticLib2:
* =====================
* This class is derived from class StochasticLib1. It redefines the functions
* Poisson, Binomial and HyperGeometric.
* In StochasticLib1, these functions are optimized for being called with 
* parameters that vary. In StochasticLib2, the same functions are optimized
* for being called repeatedly with the same parameters. If your parameters
* seldom vary, then StochasticLib2 is faster. The two classes use different
* calculation methods, both of which are accurate.
*
*
* class StochasticLib3:
* =====================
* This class can be derived from either StochasticLib1 or StochasticLib2, 
*/

/*
* This header file provides function prototypes and class declarations for
* a C++ library of non-uniform random number generators, authored by Agner Fog.
* The functions cover various distributions and utilities for statistical computations.
*
* Function prototypes:
*
* void EndOfProgram(void);
*   - A system-specific function for handling program termination.
*
* void FatalError(const char * ErrorText);
*   - Outputs error messages for functions and classes.
*
* double Erf (double x);
*   - Computes the error function, integral of the normal distribution.
*
* double LnFac(int32_t n);
*   - Computes the natural logarithm of factorial of n.
*
* class StochasticLib1:
*   - Provides non-uniform random variate generators based on uniform random number generators:
*     - int Bernoulli(double p): Bernoulli distribution.
*     - double Normal(double m, double s): Normal distribution.
*     - double NormalTrunc(double m, double s, double limit): Truncated normal distribution.
*     - int32_t Poisson (double L): Poisson distribution.
*     - int32_t Binomial (int32_t n, double p): Binomial distribution.
*     - int32_t Hypergeometric (int32_t n, int32_t m, int32_t N): Hypergeometric distribution.
*     - void Multinomial (int32_t * destination, double * source, int32_t n, int colors): Multivariate binomial distribution.
*     - void Multinomial (int32_t * destination, int32_t * source, int32_t n, int colors): Multivariate binomial distribution.
*     - void MultiHypergeometric (int32_t * destination, int32_t * source, int32_t n, int colors): Multivariate hypergeometric distribution.
*     - void Shuffle(int * list, int min, int n): Shuffles a list of integers.
*
* class StochasticLib2:
*   - Derived from StochasticLib1, optimized for repeated calls with the same parameters:
*     - Optimized versions of Poisson, Binomial, and Hypergeometric distributions.
*
* class StochasticLib3:
*   - Can be derived from either StochasticLib1 or StochasticLib2.
*/
/*
* whichever is preferred. It contains functions for generating variates with
* the univariate and multivariate Wallenius' and Fisher's noncentral
* hypergeometric distributions.
*/

/*
* int32_t WalleniusNCHyp (int32_t n, int32_t m, int32_t N, double odds);
* Sampling from Wallenius' noncentral hypergeometric distribution, which is 
* what you get when taking n items out N, m of which are colored, without 
* replacement, with bias.
*/

/*
* int32_t FishersNCHyp (int32_t n, int32_t m, int32_t N, double odds);
* Sampling from Fisher's noncentral hypergeometric distribution which is the
* conditional distribution of independent binomial variates given their sum n.
*/

/*
* void MultiWalleniusNCHyp (int32_t * destination, int32_t * source, double * weights, int32_t n, int colors);
* Sampling from multivariate Wallenius' noncentral hypergeometric distribution.
*/

/*
* void MultiFishersNCHyp (int32_t * destination, int32_t * source, double * weights, int32_t n, int colors);
* Sampling from multivariate Fisher's noncentral hypergeometric distribution.
*/

/*
* Uniform random number generators (integer and float) are also available, as
* these are inherited from the random number generator class that is the base
* class of StochasticLib1.
*/

/*
* class CWalleniusNCHypergeometric
* ================================
* This class implements various methods for calculating the probability 
* function and the mean and variance of the univariate Wallenius' noncentral 
* hypergeometric distribution. It is used by StochasticLib3 and can also be 
* used independently.
*/

/*
* class CMultiWalleniusNCHypergeometric
* =====================================
* This class implements various methods for calculating the probability func-
* tion and the mean of the multivariate Wallenius' noncentral hypergeometric
* distribution. It is used by StochasticLib3 and can also be used independently.
*/

/*
* class CMultiWalleniusNCHypergeometricMoments
* ============================================
* This class calculates the exact mean and variance of the multivariate
* Wallenius' noncentral hypergeometric probability distribution.
*/

/*
* class CFishersNCHypergeometric
* ==============================
* This class calculates the probability function and the mean and variance 
* of Fisher's noncentral hypergeometric distribution.
*/

/*
* class CMultiFishersNCHypergeometric
* ===================================
* This class calculates the probability function and the mean and variance 
* of the multivariate Fisher's noncentral hypergeometric distribution.
*/

/*
* source code:
* ============
* The code for EndOfProgram and FatalError is found in the file userintf.cpp.
* The code for the functions in StochasticLib1 is found in the file stoc1.cpp.
* The code for the functions in StochasticLib2 is found in the file stoc2.cpp.
* The code for the functions in StochasticLib3 is found in the file stoc3.cpp.
* The code for the functions in CWalleniusNCHypergeometric,
* CMultiWalleniusNCHypergeometric, CMultiWalleniusNCHypergeometricMoments,
* CFishersNCHypergeometric, and CMultiFishersNCHypergeometric can be found
* in their respective class implementation files.
*/
/*
* CMultiWalleniusNCHypergeometric and CMultiWalleniusNCHypergeometricMoments
* is found in the file wnchyppr.cpp.
* The code for the functions in CFishersNCHypergeometric and 
* CMultiFishersNCHypergeometric is found in the file fnchyppr.cpp
* LnFac is found in stoc1.cpp.
* Erf is found in wnchyppr.cpp.
*/

/*
* Examples:
* =========
* The file ex-stoc.cpp contains an example of how to use this class library.
*
* The file ex-cards.cpp contains an example of how to shuffle a list of items.
*
* The file ex-lotto.cpp contains an example of how to generate a sequence of
* random integers where no number can occur more than once.
*
* The file testbino.cpp contains an example of sampling from the binomial distribution.
*
* The file testhype.cpp contains an example of sampling from the hypergeometric distribution.
*
* The file testpois.cpp contains an example of sampling from the poisson distribution.
*
* The file testwnch.cpp contains an example of sampling from Wallenius noncentral hypergeometric distribution.
*
* The file testfnch.cpp contains an example of sampling from Fisher's noncentral hypergeometric distribution.
*
* The file testmwnc.cpp contains an example of sampling from the multivariate Wallenius noncentral hypergeometric distribution.
*
* The file testmfnc.cpp contains an example of sampling from the multivariate Fisher's noncentral hypergeometric distribution.
*
* The file evolc.zip contains examples of how to simulate biological evolution using this class library.
*/

/*
* Documentation:
* ==============
* The file ran-instructions.pdf contains further documentation and 
* instructions for these random number generators.
*
* The file distrib.pdf contains definitions of the standard statistic distributions:
* Bernoulli, Normal, Poisson, Binomial, Hypergeometric, Multinomial, MultiHypergeometric.
*
* The file sampmet.pdf contains theoretical descriptions of the methods used
* for sampling from these distributions.
*
* The file nchyp.pdf, available from www.agner.org/random/, contains
* definitions of the univariate and multivariate Wallenius and Fisher's 
* noncentral hypergeometric distributions and theoretical explanations of 
* the methods for calculating and sampling from these.
*
* Copyright 2004-2013 by Agner Fog. 
* Released under SciPy's license with permission of Agner Fog; see license.txt
*******************************************************************************/

/*
* Header guard for the file stocc.h to prevent multiple inclusions
*/
#ifndef STOCC_H
#define STOCC_H

/*
* Include math.h for mathematical functions
*/
#include <math.h>

/*
* Include the custom random number generator header
*/
#include "randomc.h"

/*
* Conditional inclusion of stocR.h when building the R-language interface
*/
#ifdef R_BUILD
   #include "stocR.h"           // Include this when building R-language interface
#endif

/*
* Definition section to choose the base class for random number generators
* STOC_BASE defines which base class to use for the non-uniform
* random number generator classes StochasticLib1, 2, and 3.
*/

#endif  // End of header guard for STOCC_H
#ifndef STOC_BASE
   #ifdef R_BUILD
      // 当构建 R 语言接口时，继承自 StocRBase
      #define STOC_BASE StocRBase
   #else
      // 否则，选择使用 C++ 的 Mersenne Twister 随机数生成器基类
      #define STOC_BASE CRandomMersenne     // C++ Mersenne Twister
      // 或选择任何其他随机数生成器基类，例如：
      //#include "randoma.h"
      //#define STOC_BASE CRandomSFMTA      // 二进制库 SFMT 生成器
   #endif
#endif

/***********************************************************************
         Other simple functions
***********************************************************************/

// 计算对数阶乘（stoc1.cpp）
double LnFac(int32_t n);
// 计算非整数的对数阶乘（wnchyppr.cpp）
double LnFacr(double x);
// 计算下降阶乘（wnchyppr.cpp）
double FallingFactorial(double a, double b);
// 误差函数（wnchyppr.cpp）
double Erf (double x);
// 对于 x > 0，计算 floor(log2(x))（wnchyppr.cpp）
int32_t FloorLog2(float x);
// 内部用于确定求和间隔的函数，返回所需精度的数量级（accuracy）（wnchyppr.cpp）
int NumSD (double accuracy);


/***********************************************************************
         Constants and tables
***********************************************************************/

// 多变量分布中的最大颜色数
#ifndef MAXCOLORS
   #define MAXCOLORS 32                // 您可以更改此值
#endif

// LnFac 函数的常量：
static const int FAK_LEN = 1024;       // 阶乘表的长度

// 下面的表是某种误差函数展开的余数表，用于拉普拉斯方法计算 Wallenius' 非中心超几何分布。
// ERFRES_N 个表涵盖从 2^(-ERFRES_B) 到 2^(-ERFRES_E) 所需的精度。只使用与所需精度匹配的表。
// 这些表定义在 erfres.h 中，该文件包含在 wnchyppr.cpp 中。

// ErfRes 表的常量：
static const int ERFRES_B = 16;        // 开始：最低精度的 -log2
static const int ERFRES_E = 40;        // 结束：最高精度的 -log2
static const int ERFRES_S =  2;        // 从开始到结束的步长
static const int ERFRES_N = (ERFRES_E-ERFRES_B)/ERFRES_S+1; // 表的数量
static const int ERFRES_L = 48;        // 每个表的长度

// 误差函数余数表：
extern "C" double ErfRes [ERFRES_N][ERFRES_L];

// 包含在积分中以获得所需精度的标准偏差数量：
extern "C" double NumSDev[ERFRES_N];


/***********************************************************************
         Class StochasticLib1
***********************************************************************/

class StochasticLib1 : public STOC_BASE {
   // 这个类封装了随机变量生成函数。
   // 可以从任何随机数生成器派生。
# 定义一个公共类 StochasticLib1，包含多种随机分布生成方法

public:
   StochasticLib1 (int seed);          // 构造函数，初始化随机数生成器的种子

   int Bernoulli(double p);            // 生成一个服从参数为 p 的伯努利分布随机数，返回值为 0 或 1

   double Normal(double m, double s);  // 生成一个均值为 m，标准差为 s 的正态分布随机数

   double NormalTrunc(double m, double s, double limit); // 生成一个均值为 m，标准差为 s，截断于 limit 的截断正态分布随机数

   int32_t Poisson (double L);         // 生成一个参数为 L 的泊松分布随机数

   int32_t Binomial (int32_t n, double p); // 生成一个参数为 n 和 p 的二项分布随机数

   int32_t Hypergeometric (int32_t n, int32_t m, int32_t N); // 生成一个参数为 n, m 和 N 的超几何分布随机数

   void Multinomial (int32_t * destination, double * source, int32_t n, int colors); // 从参数为 source 的多项分布中抽取 n 次样本，结果存放在 destination 中

   void Multinomial (int32_t * destination, int32_t * source, int32_t n, int colors);// 从参数为 source 的多项分布中抽取 n 次样本，结果存放在 destination 中

   void MultiHypergeometric (int32_t * destination, int32_t * source, int32_t n, int colors); // 从参数为 source 的多元超几何分布中抽取 n 次样本，结果存放在 destination 中

   void Shuffle(int * list, int min, int n); // 对数组 list 中从 min 开始的前 n 个元素进行随机重排

   // 内部使用的函数
protected:
   // 定义了静态方法 fc_lnpk，用于计算超几何分布中的某些值
   static double fc_lnpk(int32_t k, int32_t N_Mn, int32_t M, int32_t n); // used by Hypergeometric

   // 不同逼近方法的子函数声明

   // Poisson 分布的逆方法
   int32_t PoissonInver(double L);                         // poisson by inversion

   // Poisson 分布的比率均匀分布方法
   int32_t PoissonRatioUniforms(double L);                 // poisson by ratio of uniforms

   // 对极低 L 值使用的 Poisson 分布方法
   int32_t PoissonLow(double L);                           // poisson for extremely low L

   // 二项分布的逆方法
   int32_t BinomialInver (int32_t n, double p);            // binomial by inversion

   // 二项分布的比率均匀分布方法
   int32_t BinomialRatioOfUniforms (int32_t n, double p);  // binomial by ratio of uniforms

   // 超几何分布的逆方法，从众数开始搜索
   int32_t HypInversionMod (int32_t n, int32_t M, int32_t N);  // hypergeometric by inversion searching from mode

   // 超几何分布的比率均匀分布方法
   int32_t HypRatioOfUnifoms (int32_t n, int32_t M, int32_t N);// hypergeometric by ratio of uniforms method

   // 每个分布特定的变量声明

   // 正态分布所使用的变量
   double normal_x2;  int normal_x2_valid;

   // 超几何分布所使用的变量
   int32_t  hyp_n_last, hyp_m_last, hyp_N_last;            // Last values of parameters
   int32_t  hyp_mode, hyp_mp;                              // Mode, mode+1
   int32_t  hyp_bound;                                     // Safety upper bound
   double hyp_a;                                           // hat center
   double hyp_h;                                           // hat width
   double hyp_fm;                                          // Value at mode

   // Poisson 分布所使用的变量
   double pois_L_last;                                     // previous value of L
   double pois_f0;                                         // value at x=0 or at mode
   double pois_a;                                          // hat center
   double pois_h;                                          // hat width
   double pois_g;                                          // ln(L)
   int32_t  pois_bound;                                    // upper bound

   // 二项分布所使用的变量
   int32_t bino_n_last;                                    // last n
   double bino_p_last;                                     // last p
   int32_t bino_mode;                                      // mode
   int32_t bino_bound;                                     // upper bound
   double bino_a;                                          // hat center
   double bino_h;                                          // hat width
   double bino_g;                                          // value at mode
   double bino_r1;                                         // p/(1-p) or ln(p/(1-p))
};


/***********************************************************************
Class StochasticLib2
***********************************************************************/

// StochasticLib2 类继承自 StochasticLib1 类，重新定义了一些方法
class StochasticLib2 : public StochasticLib1 {
   // derived class, redefining some functions
// Poisson distribution
int32_t Poisson(double L); // 声明一个返回整型的函数，参数为双精度浮点数 L，实现泊松分布

// Binomial distribution
int32_t Binomial(int32_t n, double p); // 声明一个返回整型的函数，参数为整型 n 和双精度浮点数 p，实现二项分布

// Hypergeometric distribution
int32_t Hypergeometric(int32_t n, int32_t M, int32_t N); // 声明一个返回整型的函数，参数为三个整型 n、M 和 N，实现超几何分布

// Constructor
StochasticLib2(int seed):StochasticLib1(seed){}; // StochasticLib2 类的构造函数，从 StochasticLib1 继承 seed 参数作为种子

// Subfunction for Poisson distribution approximation method (search from mode)
int32_t PoissonModeSearch(double L); // 实现泊松分布的近似方法之一：从众数开始搜索

// Subfunction for Poisson distribution approximation method (patchwork rejection)
int32_t PoissonPatchwork(double L); // 实现泊松分布的近似方法之一：拼接修正拒绝法

// Static function used by PoissonPatchwork for calculation
static double PoissonF(int32_t k, double l_nu, double c_pm); // PoissonPatchwork 使用的静态函数，用于计算

// Subfunction for Binomial distribution approximation method (search from mode)
int32_t BinomialModeSearch(int32_t n, double p); // 实现二项分布的近似方法之一：从众数开始搜索

// Subfunction for Binomial distribution approximation method (patchwork rejection)
int32_t BinomialPatchwork(int32_t n, double p); // 实现二项分布的近似方法之一：拼接修正拒绝法

// Function used by BinomialPatchwork for calculation
double BinomialF(int32_t k, int32_t n, double l_pq, double c_pm); // BinomialPatchwork 使用的函数，用于计算

// Subfunction for Hypergeometric distribution approximation method (patchwork rejection)
int32_t HypPatchwork(int32_t n, int32_t M, int32_t N); // 实现超几何分布的近似方法：拼接修正拒绝法

// Variables used by Binomial distribution
int32_t Bino_k1, Bino_k2, Bino_k4, Bino_k5;
double Bino_dl, Bino_dr, Bino_r1, Bino_r2, Bino_r4, Bino_r5,
    Bino_ll, Bino_lr, Bino_l_pq, Bino_c_pm,
    Bino_f1, Bino_f2, Bino_f4, Bino_f5,
    Bino_p1, Bino_p2, Bino_p3, Bino_p4, Bino_p5, Bino_p6; // 二项分布相关变量

// Variables used by Poisson distribution
int32_t Pois_k1, Pois_k2, Pois_k4, Pois_k5;
double Pois_dl, Pois_dr, Pois_r1, Pois_r2, Pois_r4, Pois_r5,
    Pois_ll, Pois_lr, Pois_l_my, Pois_c_pm,
    Pois_f1, Pois_f2, Pois_f4, Pois_f5,
    Pois_p1, Pois_p2, Pois_p3, Pois_p4, Pois_p5, Pois_p6; // 泊松分布相关变量

// Variables used by Hypergeometric distribution
int32_t Hyp_L, Hyp_k1, Hyp_k2, Hyp_k4, Hyp_k5;
double Hyp_dl, Hyp_dr,
    Hyp_r1, Hyp_r2, Hyp_r4, Hyp_r5,
    Hyp_ll, Hyp_lr, Hyp_c_pm,
    Hyp_f1, Hyp_f2, Hyp_f4, Hyp_f5,
    Hyp_p1, Hyp_p2, Hyp_p3, Hyp_p4, Hyp_p5, Hyp_p6; // 超几何分布相关变量
// StochasticLib3 类的公共部分

public:
   StochasticLib3(int seed);           // 构造函数，初始化随机数种子

   void SetAccuracy(double accur);     // 设置计算精度

   int32_t WalleniusNCHyp (int32_t n, int32_t m, int32_t N, double odds);
   // 计算 Wallenius 非中心超几何分布，参数 n, m 分别为样本中两种类别的数量，N 为总体大小，odds 为非中心参数

   int32_t FishersNCHyp (int32_t n, int32_t m, int32_t N, double odds);
   // 计算 Fisher's 非中心超几何分布，参数 n, m 分别为样本中两种类别的数量，N 为总体大小，odds 为非中心参数

   void MultiWalleniusNCHyp (int32_t * destination, int32_t * source, double * weights, int32_t n, int colors);
   // 计算多变量 Wallenius 非中心超几何分布，destination 为结果存储数组，source 为样本数据，weights 为权重数组，n 为样本大小，colors 为类别数

   void MultiComplWalleniusNCHyp (int32_t * destination, int32_t * source, double * weights, int32_t n, int colors);
   // 计算多变量补充 Wallenius 非中心超几何分布，destination 为结果存储数组，source 为样本数据，weights 为权重数组，n 为样本大小，colors 为类别数

   void MultiFishersNCHyp (int32_t * destination, int32_t * source, double * weights, int32_t n, int colors);
   // 计算多变量 Fisher's 非中心超几何分布，destination 为结果存储数组，source 为样本数据，weights 为权重数组，n 为样本大小，colors 为类别数

   // 各种近似方法的子函数
protected:
   // 声明使用WalleniusNCHypUrn方法计算非中心超几何分布的值，参数为n，m，N，odds
   int32_t WalleniusNCHypUrn (int32_t n, int32_t m, int32_t N, double odds); // WalleniusNCHyp by urn model

   // 声明使用WalleniusNCHypInversion方法计算非中心超几何分布的值，参数为n，m，N，odds
   int32_t WalleniusNCHypInversion (int32_t n, int32_t m, int32_t N, double odds); // WalleniusNCHyp by inversion method

   // 声明使用WalleniusNCHypTable方法计算非中心超几何分布的值，参数为n，m，N，odds
   int32_t WalleniusNCHypTable (int32_t n, int32_t m, int32_t N, double odds); // WalleniusNCHyp by table method

   // 声明使用WalleniusNCHypRatioOfUnifoms方法计算非中心超几何分布的值，参数为n，m，N，odds
   int32_t WalleniusNCHypRatioOfUnifoms (int32_t n, int32_t m, int32_t N, double odds); // WalleniusNCHyp by ratio-of-uniforms

   // 声明使用FishersNCHypInversion方法计算Fisher非中心超几何分布的值，参数为n，m，N，odds
   int32_t FishersNCHypInversion (int32_t n, int32_t m, int32_t N, double odds); // FishersNCHyp by inversion

   // 声明使用FishersNCHypRatioOfUnifoms方法计算Fisher非中心超几何分布的值，参数为n，m，N，odds
   int32_t FishersNCHypRatioOfUnifoms (int32_t n, int32_t m, int32_t N, double odds); // FishersNCHyp by ratio-of-uniforms

   // desired accuracy of calculations，设置计算精度
   double accuracy;

   // Variables for Fisher，Fisher分布的变量
   int32_t fnc_n_last, fnc_m_last, fnc_N_last; // 上一次的参数值
   int32_t fnc_bound; // 上限
   double fnc_o_last; // 上一次的odds值
   double fnc_f0, fnc_scale;
   double fnc_a; // 中心值
   double fnc_h; // 宽度
   double fnc_lfm; // ln(f(mode))
   double fnc_logb; // ln(odds)

   // variables for Wallenius，Wallenius分布的变量
   int32_t wnc_n_last, wnc_m_last, wnc_N_last; // 上一次的参数值
   double wnc_o_last; // 上一次的odds值
   int32_t wnc_bound1, wnc_bound2; // 下限和上限
   int32_t wnc_mode; // mode
   double wnc_a; // 中心值
   double wnc_h; // 宽度
   double wnc_k; // mode处的概率值
   int UseChopDown; // 使用Chop Down Inversion方法的标志
   #define WALL_TABLELENGTH  512 // 表的最大长度
   double wall_ytable[WALL_TABLELENGTH]; // 概率值的表
   int32_t wall_tablen; // 表的长度
   int32_t wall_x1; // 表的下限
};

/***********************************************************************
Class CWalleniusNCHypergeometric
***********************************************************************/

class CWalleniusNCHypergeometric {
   // This class contains methods for calculating the univariate
   // Wallenius' noncentral hypergeometric probability function
// 定义公共成员函数和私有成员变量的声明
public:
   // 构造函数，初始化参数 n, m, N, odds，默认精度为 1.E-8
   CWalleniusNCHypergeometric(int32_t n, int32_t m, int32_t N, double odds, double accuracy=1.E-8);
   // 设置参数的方法
   void SetParameters(int32_t n, int32_t m, int32_t N, double odds);
   // 计算概率函数，给定 x
   double probability(int32_t x);
   // 创建概率表格，返回表格长度，更新 xfirst 和 xlast，可选截止值为 cutoff
   int32_t MakeTable(double * table, int32_t MaxLength, int32_t * xfirst, int32_t * xlast, double cutoff = 0.);
   // 近似计算均值
   double mean(void);
   // 近似计算方差（粗略估计）
   double variance(void);
   // 计算众数
   int32_t mode(void);
   // 计算精确的均值和方差
   double moments(double * mean, double * var);
   // 拒绝方法使用的 BernouilliH 函数
   int BernouilliH(int32_t x, double h, double rh, StochasticLib1 *sto);

protected:
   // 递归计算方法
   double recursive(void);
   // 积分方法的二项展开
   double binoexpand(void);
   // Laplace 方法，窄积分区间
   double laplace(void);
   // 数值积分方法
   double integrate(void);

   // 其他子函数
   // 计算自然对数的二项式系数
   double lnbico(void);
   // 计算 r, w, E 的方法
   void findpars(void);
   // integrate 方法使用的积分步长计算
   double integrate_step(double a, double b);
   // integrate 方法使用的寻找拐点函数
   double search_inflect(double t_from, double t_to);

   // 参数
   double omega;           // Odds，赔率
   int32_t n, m, N, x;     // 参数 n, m, N, x
   int32_t xmin, xmax;     // x 的最小和最大值
   double accuracy;        // 所需精度

   // lnbico 函数使用的参数
   int32_t xLastBico;      // 上一次使用的 x
   double bico, mFac, xFac; // 二项式系数、m 的阶乘、x 的阶乘

   // findpars 计算得到并由 probability, laplace, integrate 使用的参数
   double r, rd, w, wr, E, phi2d; // r, rd, w, wr, E, phi2d
   int32_t xLastFindpars;  // 上一次使用的 x
};



/***********************************************************************
Class CMultiWalleniusNCHypergeometric
***********************************************************************/

class CMultiWalleniusNCHypergeometric {
   // 该类封装了计算多元 Wallenius 非中心超几何概率函数的不同方法
// 定义公共接口和方法声明
public:
   // 构造函数，初始化 CMultiWalleniusNCHypergeometric 对象
   CMultiWalleniusNCHypergeometric(int32_t n, int32_t * m, double * odds, int colors, double accuracy=1.E-8);
   // 修改参数方法，更新采样参数和颜色信息
   void SetParameters(int32_t n, int32_t * m, double * odds, int colors);
   // 计算概率函数，根据给定的采样数目 x
   double probability(int32_t * x);
   // 计算近似均值方法，返回结果存储在 mu 数组中
   void mean(double * mu);

// 不同计算方法的实现，受保护的方法
protected:
   // 积分被积函数的二项展开
   double binoexpand(void);
   // 使用拉普拉斯方法和窄积分区间计算
   double laplace(void);
   // 数值积分方法
   double integrate(void);

   // 其他子函数
   // 计算二项系数的自然对数
   double lnbico(void);
   // 计算 r, w, E 参数值
   void findpars(void);
   // 积分函数的步骤，用于 integrate() 方法
   double integrate_step(double a, double b);
   // 在积分中使用的寻找拐点的方法，用于 integrate() 方法
   double search_inflect(double t_from, double t_to);

   // 参数定义
   double * omega;       // 各颜色的赔率
   double accuracy;      // 所需精度
   int32_t n;            // 样本大小
   int32_t N;            // 球罐中的总物品数
   int32_t * m;          // 球罐中每种颜色的物品数
   int32_t * x;          // 采样中每种颜色的物品数
   int colors;           // 不同颜色的数量
   int Dummy_align;      // 填充字段

   // 由 findpars 生成并由 probability、laplace、integrate 方法使用的参数
   double r, rd, w, wr, E, phi2d;
   // 由 lnbico 生成的参数
   double bico;
};


/***********************************************************************
Class CMultiWalleniusNCHypergeometricMoments
***********************************************************************/

// 继承自 CMultiWalleniusNCHypergeometric 的 CMultiWalleniusNCHypergeometricMoments 类
class CMultiWalleniusNCHypergeometricMoments: public CMultiWalleniusNCHypergeometric {
   // 该类通过计算所有可能的 x 组合来精确计算多元 Wallenius 非中心超几何分布的均值和方差，
   // 其中概率小于设定的精度值 accuracy
public:
   // 构造函数，初始化 CMultiWalleniusNCHypergeometricMoments 对象
   CMultiWalleniusNCHypergeometricMoments(int32_t n, int32_t * m, double * odds, int colors, double accuracy=1.E-8)
      : CMultiWalleniusNCHypergeometric(n, m, odds, colors, accuracy) {};
   // 计算均值和标准差的方法，将结果存储在 mean 和 stddev 数组中，可选地返回组合数 combinations
   double moments(double * mean, double * stddev, int32_t * combinations = 0);
protected:
   // 递归循环函数，计算非中心超几何分布的概率
   double loop(int32_t n, int c);                          // recursive loops

   // 数据
   int32_t xi[MAXCOLORS];                                  // 计算概率时使用的 x 向量
   int32_t xm[MAXCOLORS];                                  // x[i] 的四舍五入近似均值
   int32_t remaining[MAXCOLORS];                           // 球的颜色大于 c 的数量
   double sx[MAXCOLORS];                                   // x*f(x) 的和
   double sxx[MAXCOLORS];                                  // x^2*f(x) 的和
   int32_t sn;                                             // 组合的数量
};


/***********************************************************************
Class CFishersNCHypergeometric
***********************************************************************/

class CFishersNCHypergeometric {
   // 该类包含计算单变量 Fisher's 非中心超几何概率函数的方法
public:
   CFishersNCHypergeometric(int32_t n, int32_t m, int32_t N, double odds, double accuracy = 1E-8); // 构造函数
   double probability(int32_t x);                          // 计算概率函数
   double probabilityRatio(int32_t x, int32_t x0);         // 计算概率 f(x)/f(x0)
   double MakeTable(double * table, int32_t MaxLength, int32_t * xfirst, int32_t * xlast, double cutoff = 0.); // 制作概率表
   double mean(void);                                      // 计算近似均值
   double variance(void);                                  // 近似方差
   int32_t mode(void);                                     // 计算模式（精确）
   double moments(double * mean, double * var);            // 计算精确均值和方差

protected:
   double lng(int32_t x);                                  // 比例函数的自然对数

   // 参数
   double odds;                                            // 赔率比
   double logodds;                                         // 赔率比的自然对数
   double accuracy;                                        // 精度
   int32_t n, m, N;                                        // 参数
   int32_t xmin, xmax;                                     // x 的最小和最大值

   // 子函数使用的参数
   int32_t xLast;
   double mFac, xFac;                                      // 对数阶乘
   double scale;                                           // 应用于 lng 函数的比例
   double rsum;                                            // 比例函数的倒数和
   int ParametersChanged;
};


/***********************************************************************
Class CMultiFishersNCHypergeometric
***********************************************************************/
class CMultiFishersNCHypergeometric {
   // This class contains functions for calculating the multivariate
   // Fisher's noncentral hypergeometric probability function and its mean and 
   // variance. Warning: the time consumption for first call to 
   // probability or moments is proportional to the total number of
   // possible x combinations, which may be extreme!

public:
   // Constructor for initializing the CMultiFishersNCHypergeometric object.
   // Parameters:
   //   n: Total number of items in the sample
   //   m: Array of integers representing the number of items in each category
   //   odds: Array of doubles representing the odds for each category
   //   colors: Total number of categories
   //   accuracy: Optional parameter to specify the accuracy of calculations (default is 1E-9)
   CMultiFishersNCHypergeometric(int32_t n, int32_t * m, double * odds, int colors, double accuracy = 1E-9); // constructor
   
   // Calculates the probability function for a given vector x.
   // Parameters:
   //   x: Array of integers representing the vector for which probability is calculated
   // Returns:
   //   Probability value for the given x vector
   double probability(int32_t * x);                        // calculate probability function
   
   // Calculates the approximate mean of the probability distribution.
   // Parameters:
   //   mu: Array where the mean values will be stored
   void mean(double * mu);                                 // calculate approximate mean
   
   // Calculates the approximate variance of the probability distribution.
   // Parameters:
   //   var: Array where the variance values will be stored
   void variance(double * var);                            // calculate approximate variance
   
   // Calculates the exact mean and variance of the probability distribution.
   // Parameters:
   //   mean: Array where the mean values will be stored
   //   stddev: Array where the standard deviation values will be stored
   //   combinations: Optional array representing combinations of x (default is 0)
   // Returns:
   //   Moments value
   double moments(double * mean, double * stddev, int32_t * combinations = 0); // calculate exact mean and variance

protected:
   // Calculates the natural logarithm of the proportional function for a given vector x.
   // Parameters:
   //   x: Array of integers representing the vector for which logarithm is calculated
   // Returns:
   //   Natural logarithm of the proportional function for the given x vector
   double lng(int32_t * x);                                // natural log of proportional function
   
   // Calculates the sum of the proportional function for all possible x combinations.
   void SumOfAll(void);                                    // calculates sum of proportional function for all x combinations
   
   // Recursive function used by SumOfAll to calculate sums.
   // Parameters:
   //   n: Total number of items
   //   c: Current color index
   // Returns:
   //   Recursive sum value
   double loop(int32_t n, int c);                          // recursive loops used by SumOfAll

   // Member variables
   int32_t n, N;                                           // copy of parameters
   int32_t * m;                                             // array of counts
   double * odds;                                            // array of odds
   int colors;                                              // number of categories
   double logodds[MAXCOLORS];                               // log odds
   double mFac;                                             // sum of log m[i]!
   double scale;                                            // scale to apply to lng function
   double rsum;                                             // reciprocal sum of proportional function
   double accuracy;                                         // accuracy of calculation

   // Data used by SumOfAll
   int32_t xi[MAXCOLORS];                                   // x vector to calculate probability of
   int32_t xm[MAXCOLORS];                                   // rounded approximate mean of x[i]
   int32_t remaining[MAXCOLORS];                            // number of balls of color > c in urn
   double sx[MAXCOLORS];                                    // sum of x*f(x) or mean
   double sxx[MAXCOLORS];                                   // sum of x^2*f(x) or variance
   int32_t sn;                                              // number of possible combinations of x
};
```