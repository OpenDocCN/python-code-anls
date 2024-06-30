# `D:\src\scipysrc\scipy\scipy\stats\biasedurn\wnchyppr.cpp`

```
/*************************** wnchyppr.cpp **********************************
* Author:        Agner Fog
* Date created:  2002-10-20
* Last modified: 2013-12-20
* Project:       stocc.zip
* Source URL:    www.agner.org/random
*
* Description:
* Calculation of univariate and multivariate Wallenius noncentral 
* hypergeometric probability distribution.
*
* This file contains source code for the class CWalleniusNCHypergeometric 
* and CMultiWalleniusNCHypergeometricMoments defined in stocc.h.
*
* Documentation:
* ==============
* The file stocc.h contains class definitions.
* The file nchyp.pdf, available from www.agner.org/random/theory 
* describes the theory of the calculation methods.
* The file ran-instructions.pdf contains further documentation and 
* instructions.
*
* Copyright 2002-2008 by Agner Fog. 
* Released under SciPy's license with permission of Agner Fog; see license.txt
*****************************************************************************/

#include <string.h>                    // memcpy function
#include "stocc.h"                     // class definition
#include "erfres.cpp"                  // table of error function residues (Don't precompile this as a header file)

/***********************************************************************
constants
***********************************************************************/
static const double LN2 = 0.693147180559945309417; // log(2)


/***********************************************************************
Log and Exp functions with special care for small x
***********************************************************************/
// These are functions that involve expressions of the types log(1+x)
// and exp(x)-1. These functions need special care when x is small to
// avoid loss of precision. There are three versions of these functions:
// (1) Assembly version in library randomaXX.lib
// (2) Use library functions log1p and expm1 if available
// (3) Use Taylor expansion if none of the above are available

#if defined(__GNUC__) || defined (__INTEL_COMPILER) || defined(HAVE_EXPM1)
// Functions log1p(x) = log(1+x) and expm1(x) = exp(x)-1 are available
// in the math libraries of Gnu and Intel compilers
// and in R.DLL (www.r-project.org).

double pow2_1(double q, double * y0 = 0) {
   // calculate 2^q and (1-2^q) without loss of precision.
   // return value is (1-2^q). 2^q is returned in *y0
   double y, y1;
   q *= LN2;
   if (fabs(q) > 0.1) {
      y = exp(q);                      // 2^q
      y1 = 1. - y;                     // 1-2^q
   }
   else { // Use expm1
      y1 = expm1(q);                   // 2^q-1
      y = y1 + 1;                      // 2^q
      y1 = -y1;                        // 1-2^q
   }
   if (y0) *y0 = y;                    // Return y if not void pointer
   return y1;                          // Return y1
}
// (3)
// Functions log1p and expm1 are not available in MS and Borland compiler
// libraries. Use explicit Taylor expansion when needed.

double pow2_1(double q, double * y0 = 0) {
   // calculate 2^q and (1-2^q) without loss of precision.
   // return value is (1-2^q). 2^q is returned in *y0
   double y, y1, y2, qn, i, ifac;
   q *= LN2;
   // Check if q is large enough to use direct exponential calculation
   if (fabs(q) > 0.1) {
      y = exp(q);                   // Calculate e^q
      y1 = 1. - y;                  // Calculate 1 - e^q
   }
   else { // expand 1-e^q = -summa(q^n/n!) to avoid loss of precision
      y1 = 0;                       // Initialize y1 for Taylor series
      qn = i = ifac = 1;            // Initialize variables for Taylor series
      do {
         y2 = y1;                   // Store previous value of y1
         qn *= q;                   // q^n
         ifac *= i++;               // n!
         y1 -= qn / ifac;           // Sum term of Taylor series
      }
      while (y1 != y2);            // Continue until convergence of Taylor series
      y = 1. - y1;                 // Calculate 1 - e^q
   }
   if (y0) *y0 = y;                // Store 2^q in *y0 if y0 is provided
   return y1;                      // Return (1 - 2^q)
}

double log1mx(double x, double x1) {
   // Calculate log(1-x) without loss of precision when x is small.
   // Parameter x1 must be = 1-x.
   if (fabs(x) > 0.03) {
      return log(x1);             // Use direct logarithm calculation for larger x
   }
   else { // expand ln(1-x) = -summa(x^n/n) for small x
      double y, z1, z2, i;
      y = i = 1.;                  // Initialize variables for Taylor series
      z1 = 0;                      // Initialize result variable
      do {
         z2 = z1;                  // Store previous value of z1
         y *= x;                   // x^n
         z1 -= y / i++;            // Sum term of Taylor series
      }
      while (z1 != z2);            // Continue until convergence of Taylor series
      return z1;                   // Return calculated logarithm
   }
}

double log1pow(double q, double x) {
   // calculate log((1-e^q)^x) without loss of precision
   // Uses various Taylor expansions to avoid loss of precision
   double y, y1, y2, z1, z2, qn, i, ifac;

   // Check if q is large enough to use direct exponential calculation
   if (fabs(q) > 0.1) {
      y = exp(q);                  // Calculate e^q
      y1 = 1. - y;                 // Calculate 1 - e^q
   }
   else { // expand 1-e^q = -summa(q^n/n!) to avoid loss of precision
      y1 = 0;                      // Initialize y1 for Taylor series
      qn = i = ifac = 1;           // Initialize variables for Taylor series
      do {
         y2 = y1;                  // Store previous value of y1
         qn *= q;                  // q^n
         ifac *= i++;              // n!
         y1 -= qn / ifac;          // Sum term of Taylor series
      }
      while (y1 != y2);           // Continue until convergence of Taylor series
      y = 1. - y1;                // Calculate 1 - e^q
   }

   // Choose method based on magnitude of (1 - e^q)
   if (y > 0.1) {                 // (1 - e^q) is large enough for direct logarithm
      return x * log(y1);         // Calculate log((1 - e^q)^x)
   }
   else { // expand ln(1-y) = -summa(y^n/n) for small (1 - e^q)
      y1 = i = 1.;                // Initialize variables for Taylor series
      z1 = 0;                     // Initialize result variable
      do {
         z2 = z1;                 // Store previous value of z1
         y1 *= y;                 // y^n
         z1 -= y1 / i++;          // Sum term of Taylor series
      }
      while (z1 != z2);           // Continue until convergence of Taylor series
      return x * z1;              // Calculate log((1 - e^q)^x)
   }
}
double LnFacr(double x) {
   // log factorial of non-integer x
   int32_t ix = (int32_t)(x);
   // 如果 x 是整数，则调用 LnFac 函数计算其对数阶乘
   if (x == ix) return LnFac(ix);      // x is integer
   double r, r2, D = 1., f;
   static const double             
      C0 =  0.918938533204672722,      // ln(sqrt(2*pi))
      C1 =  1./12.,
      C3 = -1./360.,
      C5 =  1./1260.,
      C7 = -1./1680.;
   if (x < 6.) {
      if (x == 0 || x == 1) return 0;
      // 计算 x < 6 时的阶乘 D
      while (x < 6) D *= ++x;
   }
   r  = 1. / x;  r2 = r*r;
   // 计算非整数 x 的对数阶乘的近似值 f
   f = (x + 0.5)*log(x) - x + C0 + r*(C1 + r2*(C3 + r2*(C5 + r2*C7)));
   if (D != 1.) f -= log(D);
   return f;
}


double FallingFactorial(double a, double b) {
   // calculates ln(a*(a-1)*(a-2)* ... * (a-b+1))

   if (b < 30 && int(b) == b && a < 1E10) {
      // 直接计算，适用于 b 较小且为整数，a 较小的情况
      double f = 1.;
      for (int i = 0; i < b; i++) f *= a--;
      return log(f);
   }

   if (a > 100.*b && b > 1.) {
      // 结合 Stirling 公式来计算，避免精度损失，适用于 a 和 (a-b) 的比值较大的情况
      double ar = 1./a;
      double cr = 1./(a-b);
      // 通过 Taylor 展开计算 -log(1-b/a)
      double s = 0., lasts, n = 1., ba = b*ar, f = ba;
      do {
         lasts = s;
         s += f/n;
         f *= ba;
         n++;
      }
      while (s != lasts);
      return (a+0.5)*s + b*log(a-b) - b + (1./12.)*(ar-cr);
   }
   // 使用 LnFacr 函数计算 ln(a*(a-1)*...*(a-b+1))
   return LnFacr(a)-LnFacr(a-b);
}

double Erf (double x) {
   // Calculates the error function erf(x) as a series expansion or
   // continued fraction expansion.
   // This function may be available in math libraries as erf(x)
   static const double rsqrtpi  = 0.564189583547756286948; // 1/sqrt(pi)
   static const double rsqrtpi2 = 1.12837916709551257390;  // 2/sqrt(pi)
   if (x < 0.) return -Erf(-x);
   if (x > 6.) return 1.;
   if (x < 2.4) {
      // 使用级数展开计算误差函数 erf(x)，适用于 x 较小的情况
      double term;                     // term in summation
      double j21;                      // 2j+1
      double sum = 0;                  // summation
      double xx2 = x*x*2.;             // 2x^2
      int j;  
      term = x;  j21 = 1.;
      for (j=0; j<=50; j++) {          // summation loop
         sum += term;
         if (term <= 1.E-13) break;
         j21 += 2.;
         term *= xx2 / j21;
      }
      return exp(-x*x) * sum * rsqrtpi2;
   }
   else {
      // 使用连分数展开计算误差函数 erf(x)，适用于 x 较大的情况
      double a, f;
      int n = int(2.25f*x*x - 23.4f*x + 60.84f); // predict expansion degree
      if (n < 1) n = 1;
      a = 0.5 * n;  f = x;
      for (; n > 0; n--) {             // continued fraction loop
         f = x + a / f;
         a -= 0.5;
      }
      return 1. - exp(-x*x) * rsqrtpi / f;
   }
}


int32_t FloorLog2(float x) {
   // This function calculates floor(log2(x)) for positive x.
   // The return value is <= -127 for x <= 0.

   union UfloatInt {  // Union for extracting bits from a float
      float   f;
      int32_t i;
      UfloatInt(float ff) {f = ff;}  // constructor
   };
#if defined(_M_IX86) || defined(__INTEL__) || defined(_M_X64) || defined(__IA64__) || defined(__POWERPC__)
   // 如果定义了这些宏，表明代码运行在使用 IEEE-754 浮点格式的平台上
   int32_t n = UfloatInt(x).i;
   // 将浮点数 x 转换为整数 n，提取出浮点数的指数部分
   return (n >> 23) - 0x7F;
#else
   // 如果不确定浮点数格式是否为 IEEE-754
   static const UfloatInt check(1.0f);
   if (check.i == 0x3F800000) {
      // 如果浮点数格式是标准的 IEEE 浮点格式
      int32_t n = UfloatInt(x).i;
      // 将浮点数 x 转换为整数 n，提取出浮点数的指数部分
      return (n >> 23) - 0x7F;
   }
   else {
      // 如果浮点数格式未知
      if (x <= 0.f) return -127;
      // 返回 x 的对数除以 ln(2) 的结果，向下取整，作为浮点数的指数部分
      return (int32_t)floor(log(x)*(1./LN2));
   }
#endif
}


int NumSD (double accuracy) {
   // 计算积分间隔长度，以达到所需的精度，用于积分/求和概率函数，相对于标准偏差
   // 返回一个整数近似值，对应于 2*NormalDistrFractile(accuracy/2)
   static const double fract[] = {
      2.699796e-03, 4.652582e-04, 6.334248e-05, 6.795346e-06, 5.733031e-07,
      3.797912e-08, 1.973175e-09, 8.032001e-11, 2.559625e-12, 6.381783e-14
   };
   int i;
   for (i = 0; i < (int)(sizeof(fract)/sizeof(*fract)); i++) {
      if (accuracy >= fract[i]) break;
   }
   // 返回使得 accuracy/2 大于 fract[i] 的最小整数 i + 6
   return i + 6;
}


/***********************************************************************
Methods for class CWalleniusNCHypergeometric
***********************************************************************/

CWalleniusNCHypergeometric::CWalleniusNCHypergeometric(int32_t n_, int32_t m_, int32_t N_, double odds_, double accuracy_) {
   // 构造函数
   accuracy = accuracy_;
   // 设置精度
   SetParameters(n_, m_, N_, odds_);
   // 调用 SetParameters 设置其他参数
}


void CWalleniusNCHypergeometric::SetParameters(int32_t n_, int32_t m_, int32_t N_, double odds) {
   // 更改参数
   if (n_ < 0 || n_ > N_ || m_ < 0 || m_ > N_ || odds < 0) FatalError("Parameter out of range in CWalleniusNCHypergeometric");
   // 检查参数范围，若超出范围则报错
   n = n_; m = m_; N = N_; omega = odds;          // 设置参数
   // 设置 n, m, N, omega 参数
   xmin = m + n - N;  if (xmin < 0) xmin = 0;     // 计算 xmin
   // 计算 xmin
   xmax = n;  if (xmax > m) xmax = m;             // 计算 xmax
   // 计算 xmax
   xLastBico = xLastFindpars = -99;               // 指示上一个 x 是无效的
   // 初始化变量 xLastBico 和 xLastFindpars
   r = 1.;                                        // 初始化 r
   // 初始化 r
}
double CWalleniusNCHypergeometric::mean(void) {
   // find approximate mean
   int iter;                            // 迭代次数
   double a, b;                         // 计算第一个猜测时的临时变量
   double mean, mean1;                  // 均值的迭代值
   double m1r, m2r;                     // 1/m, 1/(N-m)
   double e1, e2;                       // 临时变量
   double g;                            // 根的函数
   double gd;                           // g 的导数
   double omegar;                       // 1/omega

   if (omega == 1.) { // 简单超几何分布
      return (double)(m)*n/N;
   }

   if (omega == 0.) {
      if (n > N-m) FatalError("Not enough items with nonzero weight in CWalleniusNCHypergeometric::mean");
      return 0.;
   }

   if (xmin == xmax) return xmin;

   // 计算作为第一个猜测的 Fisher 非中心超几何分布的 Cornfield 均值
   a = (m+n)*omega + (N-m-n); 
   b = a*a - 4.*omega*(omega-1.)*m*n;
   b = b > 0. ? sqrt(b) : 0.;
   mean = (a-b)/(2.*(omega-1.));
   if (mean < xmin) mean = xmin;
   if (mean > xmax) mean = xmax;

   m1r = 1./m;  m2r = 1./(N-m);
   iter = 0;

   if (omega > 1.) {
      do { // 牛顿-拉夫逊迭代
         mean1 = mean;
         e1 = 1.-(n-mean)*m2r;
         if (e1 < 1E-14) {
            e2 = 0.;     // 避免下溢
         }
         else {
            e2 = pow(e1,omega-1.);
         }
         g = e2*e1 + (mean-m)*m1r;
         gd = e2*omega*m2r + m1r;
         mean -= g / gd;
         if (mean < xmin) mean = xmin;
         if (mean > xmax) mean = xmax;
         if (++iter > 40) {
            FatalError("Search for mean failed in function CWalleniusNCHypergeometric::mean");
         }
      }
      while (fabs(mean1 - mean) > 2E-6);
   }
   else { // omega < 1
      omegar = 1./omega;
      do { // 牛顿-拉夫逊迭代
         mean1 = mean;
         e1 = 1.-mean*m1r;
         if (e1 < 1E-14) {
            e2 = 0.;   // 避免下溢
         }
         else {
            e2 = pow(e1,omegar-1.);
         }
         g = 1.-(n-mean)*m2r-e2*e1;
         gd = e2*omegar*m1r + m2r;
         mean -= g / gd;
         if (mean < xmin) mean = xmin;
         if (mean > xmax) mean = xmax;
         if (++iter > 40) {
            FatalError("Search for mean failed in function CWalleniusNCHypergeometric::mean");
         }
      }
      while (fabs(mean1 - mean) > 2E-6);
   }
   return mean;
}


double CWalleniusNCHypergeometric::variance(void) {
   // find approximate variance (poor approximation)    
   double my = mean(); // 计算近似均值
   // 使用 Fisher 非中心超几何分布的近似值计算近似方差
   double r1 = my * (m-my); double r2 = (n-my)*(my+N-n-m);
   if (r1 <= 0. || r2 <= 0.) return 0.;
   double var = N*r1*r2/((N-1)*(m*r2+(N-m)*r1));
   if (var < 0.) var = 0.;
   return var;
}
// 计算准确的均值和方差
// 返回值是 f(x) 的总和，期望值为 1.
double CWalleniusNCHypergeometric::moments(double * mean_, double * var_) {
   double y, sy=0, sxy=0, sxxy=0, me1;  // 定义变量 y, sy, sxy, sxxy, me1
   int32_t x, xm, x1;                    // 定义变量 x, xm, x1
   const double accuracy = 1E-10f;       // 计算精度设定为 1E-10
   xm = (int32_t)mean();                 // 计算均值的近似值
   for (x = xm; x <= xmax; x++) {        // 循环计算从 xm 到 xmax 的概率值
      y = probability(x);
      x1 = x - xm;                      // 减去近似均值以避免在求和中丢失精度
      sy += y; sxy += x1 * y; sxxy += x1 * x1 * y;
      if (y < accuracy && x != xm) break;  // 如果概率小于精度并且 x 不等于 xm，则结束循环
   }
   for (x = xm-1; x >= xmin; x--) {      // 循环计算从 xm-1 到 xmin 的概率值
      y = probability(x);
      x1 = x - xm;                      // 减去近似均值以避免在求和中丢失精度
      sy += y; sxy += x1 * y; sxxy += x1 * x1 * y;
      if (y < accuracy) break;          // 如果概率小于精度，则结束循环
   }

   me1 = sxy / sy;                      // 计算均值的第一部分
   *mean_ = me1 + xm;                   // 更新均值
   y = sxxy / sy - me1 * me1;           // 计算方差
   if (y < 0) y=0;                      // 如果方差小于 0，则设置为 0
   *var_ = y;                           // 更新方差
   return sy;                           // 返回 f(x) 的总和
}


int32_t CWalleniusNCHypergeometric::mode(void) {
   int32_t Mode;                        // 定义 mode

   if (omega == 1.) { 
      // 简单超几何分布
      int32_t L  = m + n - N;
      int32_t m1 = m + 1, n1 = n + 1;
      Mode = int32_t((double)m1*n1*omega/((m1+n1)*omega-L));  // 计算模式
   }
   else {
      // 查找模式
      double f, f2 = 0.;                 // 定义变量 f, f2
      int32_t xi, x2;
      int32_t xmin = m + n - N;          // 计算最小值 xmin
      if (xmin < 0) xmin = 0;
      int32_t xmax = n;                  // 计算最大值 xmax
      if (xmax > m) xmax = m;

      Mode = (int32_t)mean();            // 计算均值的下取整
      if (omega < 1.) {
        if (Mode < xmax) Mode++;         // 如果均值小于 xmax，则上取整
        x2 = xmin;                       // 设置下限
        if (omega > 0.294 && N <= 10000000) {
          x2 = Mode - 1;                 // 可以限制搜索模式的范围
        }
        for (xi = Mode; xi >= x2; xi--) {
          f = probability(xi);
          if (f <= f2) break;
          Mode = xi; f2 = f;
        }
      }
      else {
        if (Mode < xmin) Mode++;
        x2 = xmax;
        if (omega < 3.4 && N <= 10000000) {
          x2 = Mode + 1;
        }
        for (xi = Mode; xi <= x2; xi++) {
          f = probability(xi);
          if (f <= f2) break;
          Mode = xi; f2 = f;
        }
      }
   }
   return Mode;                         // 返回模式
}
double CWalleniusNCHypergeometric::lnbico() {
   // 计算二项式系数的自然对数
   // 返回 lambda = log(m!*x!/(m-x)!*m2!*x2!/(m2-x2)!)
   int32_t x2 = n-x, m2 = N-m;
   if (xLastBico < 0) { // 如果 m, n, N 发生变化
      // 计算 m! 和 m2! 的对数阶乘之和
      mFac = LnFac(m) + LnFac(m2);
   }
   // 当 m 和 m2 小于 FAK_LEN 时跳转到 DEFLT 处理
   if (m < FAK_LEN && m2 < FAK_LEN)  goto DEFLT;
   switch (x - xLastBico) {
  case 0: // x 未改变
     break;
  case 1: // x 增加了，从之前的值计算
     // 更新 xFac，计算新的二项式系数的对数
     xFac += log (double(x) * (m2-x2) / (double(x2+1)*(m-x+1)));
     break;
  case -1: // x 减少了，从之前的值计算
     // 更新 xFac，计算新的二项式系数的对数
     xFac += log (double(x2) * (m-x) / (double(x+1)*(m2-x2+1)));
     break;
  default: DEFLT: // 计算所有情况
     // 计算所有相关的对数阶乘
     xFac = LnFac(x) + LnFac(x2) + LnFac(m-x) + LnFac(m2-x2);
   }
   // 更新 xLastBico 的值为当前的 x
   xLastBico = x;
   // 返回计算得到的二项式系数的自然对数
   return bico = mFac - xFac;
}
void CWalleniusNCHypergeometric::findpars() {
   // 计算 d, E, r, w
   if (x == xLastFindpars) {
      return;    // 自上次调用以来所有值均未更改
   }

   // 查找 r 以使积分峰值位于 0.5
   double dd, d1, z, zd, rr, lastr, rrc, rt, r2, r21, a, b, dummy;
   double oo[2];
   double xx[2] = {static_cast<double>(x), static_cast<double>(n-x)};
   int i, j = 0;
   if (omega > 1.) { // 使两个 omegas 都 <= 1 以避免溢出
      oo[0] = 1.;  oo[1] = 1./omega;
   }
   else {
      oo[0] = omega;  oo[1] = 1.;
   }
   dd = oo[0]*(m-x) + oo[1]*(N-m-xx[1]);
   d1 = 1./dd;
   E = (oo[0]*m + oo[1]*(N-m)) * d1;
   rr = r;
   if (rr <= d1) rr = 1.2*d1;           // 初始猜测
   // 使用牛顿-拉夫逊迭代法查找 r
   do {
      lastr = rr;
      rrc = 1. / rr;
      z = dd - rrc;
      zd = rrc * rrc;
      for (i=0; i<2; i++) {
         rt = rr * oo[i];
         if (rt < 100.) {                  // 避免 rt 太大导致溢出
            r21 = pow2_1(rt, &r2);         // r2=2^r, r21=1.-2^r
            a = oo[i] / r21;               // omegai/(1.-2^r)
            b = xx[i] * a;                 // x*omegai/(1.-2^r)
            z  += b;
            zd += b * a * LN2 * r2;
         }
      }
      if (zd == 0) FatalError("can't find r in function CWalleniusNCHypergeometric::findpars");
      rr -= z / zd;
      if (rr <= d1) rr = lastr * 0.125 + d1*0.875;
      if (++j == 70) FatalError("convergence problem searching for r in function CWalleniusNCHypergeometric::findpars");
   }
   while (fabs(rr-lastr) > rr * 1.E-6);
   if (omega > 1) {
      dd *= omega;  rr *= oo[1];
   }
   r = rr;  rd = rr * dd;

   // 查找峰值宽度
   double ro, k1, k2;
   ro = r * omega;
   if (ro < 300) {                      // 避免溢出
      k1 = pow2_1(ro, &dummy);
      k1 = -1. / k1;
      k1 = omega*omega*(k1+k1*k1);
   }
   else k1 = 0.;
   if (r < 300) {                       // 避免溢出
      k2 = pow2_1(r, &dummy);
      k2 = -1. / k2;
      k2 = (k2+k2*k2);
   }
   else k2 = 0.;
   phi2d = -4.*r*r*(x*k1 + (n-x)*k2);
   if (phi2d >= 0.) {
      FatalError("peak width undefined in function CWalleniusNCHypergeometric::findpars");
      /* wr = r = 0.; */ 
   }
   else {
      wr = sqrt(-phi2d); w = 1./wr;
   }
   xLastFindpars = x;
}
// 定义 CWalleniusNCHypergeometric 类的 recursive 方法，返回双精度浮点数
double CWalleniusNCHypergeometric::recursive() {
   // 递归计算
   // 使用 Wallenius 非中心超几何分布的递归公式
   // 通过忽略精度低于 accuracy 的概率来近似，并最小化存储需求
   const int BUFSIZE = 512;            // 缓冲区大小
   double p[BUFSIZE+2];                // 概率数组
   double * p1, * p2;                  // p 数组的偏移指针
   double mxo;                         // (m-x)*omega
   double Nmnx;                        // N-m-nu+x
   double y, y1;                       // 在覆盖前保存旧的 p[x]
   double d1, d2, dcom;                // 概率公式中的除数
   double accuracya;                   // 绝对精度
   int32_t xi, nu;                     // xi, nu = x, n 的递归值
   int32_t x1, x2;                     // xi_min, xi_max

   accuracya = 0.005f * accuracy;      // 绝对精度
   p1 = p2 = p + 1;                    // 为 p1[-1] 留出空间
   p1[-1] = 0.;  p1[0]  = 1.;          // 初始化递归的起始条件
   x1 = x2 = 0;
   for (nu=1; nu<=n; nu++) {
      if (n - nu < x - x1 || p1[x1] < accuracya) {
         x1++;                        // 当超过断点或概率可以忽略时增加下限
         p2--;                        // 减少缓冲区偏移量以减少存储空间
      }
      if (x2 < x && p1[x2] >= accuracya) {
         x2++;  y1 = 0.;               // 增加上限直到达到 x
      }
      else {
         y1 = p1[x2];
      }
      if (x1 > x2) return 0.;
      if (p2+x2-p > BUFSIZE) FatalError("函数 CWalleniusNCHypergeometric::recursive 中的缓冲区溢出");

      mxo = (m-x2)*omega;
      Nmnx = N-m-nu+x2+1;
      for (xi = x2; xi >= x1; xi--) {  // 反向循环
         d2 = mxo + Nmnx;
         mxo += omega; Nmnx--;
         d1 = mxo + Nmnx;
         dcom = 1. / (d1 * d2);        // 通过共同的除数来节省一次除法
         y  = p1[xi-1]*mxo*d2*dcom + y1*(Nmnx+1)*d1*dcom;
         y1 = p1[xi-1];                // （警告：指针别名，不能交换指令顺序）
         p2[xi] = y;
      }
      p1 = p2;
   }

   if (x < x1 || x > x2) return 0.;
   return p1[x];
}
double CWalleniusNCHypergeometric::binoexpand() {
   // calculate by binomial expansion of integrand
   // only for x < 2 or n-x < 2 (not implemented for higher x because of loss of precision)
   int32_t x1, m1, m2;
   double o;
   if (x > n/2) { // invert
      x1 = n-x; m1 = N-m; m2 = m; o = 1./omega;
   }
   else {
      x1 = x; m1 = m; m2 = N-m; o = omega;
   }
   if (x1 == 0) {
      return exp(FallingFactorial(m2,n) - FallingFactorial(m2+o*m1,n));
   }    
   if (x1 == 1) {
      double d, e, q, q0, q1;
      q = FallingFactorial(m2,n-1);
      e = o*m1+m2;
      q1 = q - FallingFactorial(e,n);
      e -= o;
      q0 = q - FallingFactorial(e,n);
      d = e - (n-1);
      return m1*d*(exp(q0) - exp(q1));
   }

   FatalError("x > 1 not supported by function CWalleniusNCHypergeometric::binoexpand");
   return 0;
}

double CWalleniusNCHypergeometric::integrate_step(double ta, double tb) {
   // integration subprocedure used by integrate()
   // makes one integration step from ta to tb using Gauss-Legendre method.
   // result is scaled by multiplication with exp(bico)
   double ab, delta, tau, ltau, y, sum, taur, rdm1;
   int i;

   // define constants for Gauss-Legendre integration with IPOINTS points
#define IPOINTS  8  // number of points in each integration step

#if   IPOINTS == 3
   static const double xval[3]    = {-.774596669241,0,0.774596668241};
   static const double weights[3] = {.5555555555555555,.88888888888888888,.55555555555555};
#elif IPOINTS == 4
   static const double xval[4]    = {-0.861136311594,-0.339981043585,0.339981043585,0.861136311594};
   static const double weights[4] = {0.347854845137,0.652145154863,0.652145154863,0.347854845137};
#elif IPOINTS == 5
   static const double xval[5]    = {-0.906179845939,-0.538469310106,0,0.538469310106,0.906179845939};
   static const double weights[5] = {0.236926885056,0.478628670499,0.568888888889,0.478628670499,0.236926885056};
#elif IPOINTS == 6
   static const double xval[6]    = {-0.932469514203,-0.661209386466,-0.238619186083,0.238619186083,0.661209386466,0.932469514203};
   static const double weights[6] = {0.171324492379,0.360761573048,0.467913934573,0.467913934573,0.360761573048,0.171324492379};
#elif IPOINTS == 8
   static const double xval[8]    = {-0.960289856498,-0.796666477414,-0.525532409916,-0.183434642496,0.183434642496,0.525532409916,0.796666477414,0.960289856498};
   static const double weights[8] = {0.10122853629,0.222381034453,0.313706645878,0.362683783378,0.362683783378,0.313706645878,0.222381034453,0.10122853629};
#endif
#elif IPOINTS == 12
   static const double xval[12]   = {-0.981560634247,-0.90411725637,-0.769902674194,-0.587317954287,-0.367831498998,-0.125233408511,0.125233408511,0.367831498998,0.587317954287,0.769902674194,0.90411725637,0.981560634247};
   static const double weights[12]= {0.0471753363866,0.106939325995,0.160078328543,0.203167426723,0.233492536538,0.249147045813,0.249147045813,0.233492536538,0.203167426723,0.160078328543,0.106939325995,0.0471753363866};
#elif IPOINTS == 16
   static const double xval[16]   = {-0.989400934992,-0.944575023073,-0.865631202388,-0.755404408355,-0.617876244403,-0.458016777657,-0.281603550779,-0.0950125098376,0.0950125098376,0.281603550779,0.458016777657,0.617876244403,0.755404408355,0.865631202388,0.944575023073,0.989400934992};
   static const double weights[16]= {0.027152459411,0.0622535239372,0.0951585116838,0.124628971256,0.149595988817,0.169156519395,0.182603415045,0.189450610455,0.189450610455,0.182603415045,0.169156519395,0.149595988817,0.124628971256,0.0951585116838,0.0622535239372,0.027152459411};
#else
#error // IPOINTS must be a value for which the tables are defined
#endif

   delta = 0.5 * (tb - ta);  // 计算区间的一半长度
   ab = 0.5 * (ta + tb);     // 计算区间的中点
   rdm1 = rd - 1.;           // rd 减去 1 的结果
   sum = 0;                  // 初始化 sum 为 0

   for (i = 0; i < IPOINTS; i++) {  // 遍历 IPOINTS 次数
      tau = ab + delta * xval[i];   // 计算 tau 值
      ltau = log(tau);              // 计算 tau 的自然对数
      taur = r * ltau;              // 计算 r 乘以 tau 的自然对数
      // 可能因为这里的减法而导致精度损失：
      y = log1pow(taur*omega,x) + log1pow(taur,n-x) + rdm1*ltau + bico;  // 计算 y 值
      if (y > -50.) sum += weights[i] * exp(y);  // 如果 y 大于 -50，则累加权重乘以 exp(y)
   }
   return delta * sum;  // 返回 delta 乘以 sum 的结果作为最终计算值
}


}


double CWalleniusNCHypergeometric::probability(int32_t x_) {
   // 计算概率函数，选择最佳方法
   x = x_;  // 将输入的 x 值保存到类成员变量 x 中
   if (x < xmin || x > xmax) return 0.;  // 如果 x 小于最小值 xmin 或者大于最大值 xmax，则返回 0
   if (xmin == xmax) return 1.;  // 如果 xmin 等于 xmax，则返回 1

   if (omega == 1.) { // 如果 omega 等于 1，使用超几何分布计算
      return exp(lnbico() + LnFac(n) + LnFac(N-n) - LnFac(N));  // 返回超几何分布的概率值
   }

   if (omega == 0.) {  // 如果 omega 等于 0
      if (n > N-m) FatalError("Not enough items with nonzero weight in CWalleniusNCHypergeometric::probability");  // 如果 n 大于 N-m，则抛出致命错误
      return x == 0;  // 如果 x 等于 0，则返回 1；否则返回 0
   }

   int32_t x2 = n - x;  // 计算 n - x
   int32_t x0 = x < x2 ? x : x2;  // 取较小值 x 或者 x2
   int em = (x == m || x2 == N-m);  // 判断是否满足条件 em

   if (x0 == 0 && n > 500) {  // 如果 x0 等于 0 且 n 大于 500
      return binoexpand();  // 使用二项式展开计算概率
   }

   if (double(n)*x0 < 1000 || (double(n)*x0 < 10000 && (N > 1000.*n || em))) {  // 如果条件满足
      return recursive();  // 使用递归计算概率
   }

   if (x0 <= 1 && N-n <= 1) {  // 如果条件满足
      return binoexpand();  // 使用二项式展开计算概率
   }

   findpars();  // 寻找参数

   if (w < 0.04 && E < 10 && (!em || w > 0.004)) {  // 如果条件满足
      return laplace();  // 使用拉普拉斯方法计算概率
   }

   return integrate();  // 使用积分计算概率
}


}


/***********************************************************************
calculation methods in class CMultiWalleniusNCHypergeometric
***********************************************************************/

CMultiWalleniusNCHypergeometric::CMultiWalleniusNCHypergeometric(int32_t n_, int32_t * m_, double * odds_, int colors_, double accuracy_) {
   // 构造函数
   accuracy = accuracy_;  // 设置精度
   SetParameters(n_, m_, odds_, colors_);  // 调用 SetParameters 方法设置参数
}
// 设置多变量Wallenius非中心超几何分布的参数
void CMultiWalleniusNCHypergeometric::SetParameters(int32_t n_, int32_t * m_, double * odds_, int colors_) {
   // 改变参数
   int32_t N1;                              // 声明变量N1
   int i;                                    // 声明循环变量i
   n = n_;  m = m_;  omega = odds_;  colors = colors_;  // 设置成员变量值
   r = 1.;                                    // 初始化r为1.0
   for (N = N1 = 0, i = 0; i < colors; i++) {
      if (m[i] < 0 || omega[i] < 0) FatalError("Parameter negative in constructor for CMultiWalleniusNCHypergeometric");
      N += m[i];                             // 计算总数N
      if (omega[i]) N1 += m[i];               // 如果omega[i]非零，计算N1
   }
   if (N < n) FatalError("Not enough items in constructor for CMultiWalleniusNCHypergeometric");  // 若N小于n，报错
   if (N1 < n) FatalError("Not enough items with nonzero weight in constructor for CMultiWalleniusNCHypergeometric");  // 若N1小于n，报错
}


// 计算多变量Wallenius非中心超几何分布的近似均值，结果存放在mu[0..colors-1]中
void CMultiWalleniusNCHypergeometric::mean(double * mu) {
   double omeg[MAXCOLORS];                    // 缩放后的权重
   double omr;                                // 均值倒数的权重
   double t, t1;                              // 迭代中的独立变量
   double To, To1;                            // exp(t*omega[i])和1-exp(t*omega[i])
   double H;                                  // 用于求根的函数
   double HD;                                 // H的导数
   double dummy;                              // 未使用的返回值
   int i;                                     // 颜色索引
   int iter;                                  // 迭代次数

   if (n == 0) {
      // 需要特殊处理的情况
      for (i = 0; i < colors; i++) {
         mu[i] = 0.;
      }
      return;
   }

   // 计算均值倒数
   for (omr = 0., i = 0; i < colors; i++) omr += omega[i] * m[i];
   omr = N / omr;
   
   // 缩放权重使得均值等于1
   for (i = 0; i < colors; i++) omeg[i] = omega[i] * omr;
   
   // 牛顿-拉弗森迭代法
   iter = 0;  t = -1.;                        // 初次猜测
   do {
      t1 = t;
      H = HD = 0.;
      // 计算H和HD
      for (i = 0; i < colors; i++) {
         if (omeg[i] != 0.) {
            To1 = pow2_1(t * (1. / LN2) * omeg[i], &To);
            H += m[i] * To1;
            HD -= m[i] * omeg[i] * To;
         }
      }
      t -= (H - n) / HD;
      if (t >= 0) t = 0.5 * t1;
      if (++iter > 20) {
         FatalError("Search for mean failed in function CMultiWalleniusNCHypergeometric::mean");
      }
   } while (fabs(H - n) > 1E-3);

   // 完成迭代。获取所有mu[i]
   for (i = 0; i < colors; i++) {
      if (omeg[i] != 0.) {
         To1 = pow2_1(t * (1. / LN2) * omeg[i], &dummy);
         mu[i] = m[i] * To1;
      }
      else {
         mu[i] = 0.;
      }
   }
}
/*
void CMultiWalleniusNCHypergeometric::variance(double * var, double * mean_) {
   // calculates approximate variance and mean of multivariate 
   // Wallenius' noncentral hypergeometric distribution 
   // (accuracy is not too good).
   // Variance is returned in variance[0..colors-1].
   // Mean is returned in mean_[0..colors-1] if not NULL.
   // The calculation is reasonably fast.
   double r1, r2;
   double mu[MAXCOLORS];
   int i;

   // Store mean in array mu if mean_ is NULL
   if (mean_ == 0) mean_ = mu;

   // Calculate mean
   mean(mean_);

   // Calculate variance
   for (i = 0; i < colors; i++) {
      r1 = mean_[i] * (m[i]-mean_[i]);
      r2 = (n-mean_[i])*(mean_[i]+N-n-m[i]);
      if (r1 <= 0. || r2 <= 0.) {
         var[i] = 0.;
      }
      else {
         var[i] = N*r1*r2/((N-1)*(m[i]*r2+(N-m[i])*r1));
      }
   }
}
*/

/*
// implementations of different calculation methods
*/

double CMultiWalleniusNCHypergeometric::binoexpand(void) {
   // binomial expansion of integrand
   // only implemented for x[i] = 0 for all but one i
   int i, j, k;
   double W = 0.;                       // total weight
   for (i=j=k=0; i<colors; i++) {
      W += omega[i] * m[i];
      if (x[i]) {
         j=i; k++;                      // find the nonzero x[i]
      }
   }
   if (k > 1) FatalError("More than one x[i] nonzero in CMultiWalleniusNCHypergeometric::binoexpand");
   return exp(FallingFactorial(m[j],n) - FallingFactorial(W/omega[j],n));
}


double CMultiWalleniusNCHypergeometric::lnbico(void) {
   // natural log of binomial coefficients
   bico = 0.;
   int i;
   for (i=0; i<colors; i++) {
      if (x[i] < m[i] && omega[i]) {
         bico += LnFac(m[i]) - LnFac(x[i]) - LnFac(m[i]-x[i]);
      }
   }
   return bico;
}
// 计算 r, w, E
// 计算 d, E, r, w
void CMultiWalleniusNCHypergeometric::findpars(void) {
   // 寻找 r 以使积分的峰值位于 0.5 处

   // 定义变量
   double dd;                           // 缩放后的 d
   double dr;                           // 1/d

   double z, zd, rr, lastr, rrc, rt, r2, r21, a, b, ro, k1, dummy;
   double omax;                         // 最大的 omega
   double omaxr;                        // 1/omax
   double omeg[MAXCOLORS];              // 缩放后的权重
   int i, j = 0;

   // 寻找最大的 omega
   for (omax = 0., i = 0; i < colors; i++) {
      if (omega[i] > omax) omax = omega[i];
   }
   omaxr = 1. / omax;
   dd = E = 0.;
   for (i = 0; i < colors; i++) {
      // 缩放权重使最大值为 1
      omeg[i] = omega[i] * omaxr;
      // 计算 d 和 E
      dd += omeg[i] * (m[i] - x[i]);
      E  += omeg[i] * m[i];
   }
   dr = 1. / dd;
   E *= dr;
   rr = r * omax;
   if (rr <= dr) rr = 1.2 * dr;  // 初始猜测
   // 使用牛顿-拉夫逊法迭代寻找 r
   do {
      lastr = rr;
      rrc = 1. / rr;
      z = dd - rrc;                    // z(r)
      zd = rrc * rrc;                  // z'(r)
      for (i = 0; i < colors; i++) {
         rt = rr * omeg[i];
         if (rt < 100. && rt > 0.) {   // 避免溢出和除以 0
            r21 = pow2_1(rt, &r2);     // r2=2^r, r21=1.-2^r
            a = omeg[i] / r21;         // omegai/(1.-2^r)
            b = x[i] * a;              // x*omegai/(1.-2^r)
            z  += b;
            zd += b * a * r2 * LN2;
         }
      }
      if (zd == 0) FatalError("can't find r in function CMultiWalleniusNCHypergeometric::findpars");
      rr -= z / zd;                    // 下一个 r
      if (rr <= dr) rr = lastr * 0.125 + dr * 0.875;
      if (++j == 70) FatalError("convergence problem searching for r in function CMultiWalleniusNCHypergeometric::findpars");
   }
   while (fabs(rr - lastr) > rr * 1.E-5);
   rd = rr * dd;
   r = rr * omaxr;

   // 计算峰宽度
   phi2d = 0.;
   for (i = 0; i < colors; i++) {
      ro = rr * omeg[i];
      if (ro < 300 && ro > 0.) {       // 避免溢出和除以 0
         k1 = pow2_1(ro, &dummy);
         k1 = -1. / k1;
         k1 = omeg[i] * omeg[i] * (k1 + k1*k1);
      }
      else k1 = 0.;
      phi2d += x[i] * k1;
   }
   phi2d *= -4. * rr * rr;
   if (phi2d > 0.) FatalError("peak width undefined in function CMultiWalleniusNCHypergeometric::findpars");
   wr = sqrt(-phi2d);  w = 1. / wr;
}



// integrate() 使用的积分子过程
// 使用 Gauss-Legendre 方法从 ta 到 tb 执行一次积分步骤
// 结果通过乘以 exp(bico) 进行缩放
double CMultiWalleniusNCHypergeometric::integrate_step(double ta, double tb) {
   // 定义 Gauss-Legendre 积分的常数和点数
#define IPOINTS  8  // 每个积分步骤中的点数

   // 定义变量
   double ab, delta, tau, ltau, y, sum, taur, rdm1;
   int i, j;
#if IPOINTS == 3
   // 定义具有3个节点的Gauss-Legendre积分的节点值和权重
   static const double xval[3]    = {-.774596669241, 0, 0.774596668241};
   static const double weights[3] = {.5555555555555555, .88888888888888888, .55555555555555};
#elif IPOINTS == 4
   // 定义具有4个节点的Gauss-Legendre积分的节点值和权重
   static const double xval[4]    = {-0.861136311594, -0.339981043585, 0.339981043585, 0.861136311594};
   static const double weights[4] = {0.347854845137, 0.652145154863, 0.652145154863, 0.347854845137};
#elif IPOINTS == 5
   // 定义具有5个节点的Gauss-Legendre积分的节点值和权重
   static const double xval[5]    = {-0.906179845939, -0.538469310106, 0, 0.538469310106, 0.906179845939};
   static const double weights[5] = {0.236926885056, 0.478628670499, 0.568888888889, 0.478628670499, 0.236926885056};
#elif IPOINTS == 6
   // 定义具有6个节点的Gauss-Legendre积分的节点值和权重
   static const double xval[6]    = {-0.932469514203, -0.661209386466, -0.238619186083, 0.238619186083, 0.661209386466, 0.932469514203};
   static const double weights[6] = {0.171324492379, 0.360761573048, 0.467913934573, 0.467913934573, 0.360761573048, 0.171324492379};
#elif IPOINTS == 8
   // 定义具有8个节点的Gauss-Legendre积分的节点值和权重
   static const double xval[8]    = {-0.960289856498, -0.796666477414, -0.525532409916, -0.183434642496, 0.183434642496, 0.525532409916, 0.796666477414, 0.960289856498};
   static const double weights[8] = {0.10122853629, 0.222381034453, 0.313706645878, 0.362683783378, 0.362683783378, 0.313706645878, 0.222381034453, 0.10122853629};
#elif IPOINTS == 12
   // 定义具有12个节点的Gauss-Legendre积分的节点值和权重
   static const double xval[12]   = {-0.981560634247, -0.90411725637, -0.769902674194, -0.587317954287, -0.367831498998, -0.125233408511, 0.125233408511, 0.367831498998, 0.587317954287, 0.769902674194, 0.90411725637, 0.981560634247};
   static const double weights[12]= {0.0471753363866, 0.106939325995, 0.160078328543, 0.203167426723, 0.233492536538, 0.249147045813, 0.249147045813, 0.233492536538, 0.203167426723, 0.160078328543, 0.106939325995, 0.0471753363866};
#elif IPOINTS == 16
   // 定义具有16个节点的Gauss-Legendre积分的节点值和权重
   static const double xval[16]   = {-0.989400934992, -0.944575023073, -0.865631202388, -0.755404408355, -0.617876244403, -0.458016777657, -0.281603550779, -0.0950125098376, 0.0950125098376, 0.281603550779, 0.458016777657, 0.617876244403, 0.755404408355, 0.865631202388, 0.944575023073, 0.989400934992};
   static const double weights[16]= {0.027152459411, 0.0622535239372, 0.0951585116838, 0.124628971256, 0.149595988817, 0.169156519395, 0.182603415045, 0.189450610455, 0.189450610455, 0.182603415045, 0.169156519395, 0.149595988817, 0.124628971256, 0.0951585116838, 0.0622535239372, 0.027152459411};
#else
#error // IPOINTS必须是定义了表格的值
#endif

   // 计算区间的一半
   delta = 0.5 * (tb - ta);
   // 计算区间的平均值
   ab = 0.5 * (ta + tb);
   // rd减去1
   rdm1 = rd - 1.;
   // 初始化sum为0
   sum = 0;

   // 开始Gauss-Legendre积分的循环
   for (j = 0; j < IPOINTS; j++) {
      // 计算新的tau值
      tau = ab + delta * xval[j];
      // 计算tau的自然对数
      ltau = log(tau);
      // 计算taur值
      taur = r * ltau;
      // 初始化y为0
      y = 0.;
      // 对于每一种颜色，计算y值
      for (i = 0; i < colors; i++) {
         // 如果omega[i]不为零，计算log1pow函数值
         if (omega[i]) {
            y += log1pow(taur * omega[i], x[i]);   // ln((1-e^taur*omegai)^xi)
         }
      }
      // 加上rdm1 * ltau和bico的值
      y += rdm1 * ltau + bico;
      // 如果y大于-50，累加权重乘以exp(y)到sum
      if (y > -50.) sum += weights[j] * exp(y);
   }
   // 返回计算结果
   return delta * sum;
}
// 计算概率函数，选择最佳方法
double CMultiWalleniusNCHypergeometric::probability(int32_t * x_) {
   // 声明变量
   int i, j, em;
   // 中心化变量
   int central;
   // 计算 x 数组元素的总和
   int32_t xsum;
   // 将输入的 x 数组赋值给成员变量 x
   x = x_;

   // 计算 x 数组元素的总和
   for (xsum = i = 0; i < colors; i++)  xsum += x[i];
   // 如果 x 的总和不等于 n，则抛出致命错误
   if (xsum != n) {
      FatalError("sum of x values not equal to n in function CMultiWalleniusNCHypergeometric::probability");
   }

   // 如果颜色种类小于 3
   if (colors < 3) { 
      // 如果颜色种类小于等于 0，则返回 1
      if (colors <= 0) return 1.;
      // 如果颜色种类为 1，则返回 x[0] 是否等于 m[0]
      if (colors == 1) return x[0] == m[0];
      // 如果颜色种类为 2 且 omega[1] 等于 0，则返回 x[0] 是否等于 m[0]
      if (omega[1] == 0.) return x[0] == m[0];
      // 否则，调用单变量 WalleniusNCHypergeometric 分布的 probability 方法
      return CWalleniusNCHypergeometric(n,m[0],N,omega[0]/omega[1],accuracy).probability(x[0]);
   }

   // 初始化变量 central 为 1
   central = 1;
   // 遍历颜色种类
   for (i = j = em = 0; i < colors; i++) {
      // 如果 x[i] 大于 m[i] 或者小于 0，或者小于 n-N+m[i]，则返回 0
      if (x[i] > m[i] || x[i] < 0 || x[i] < n - N + m[i]) return 0.;
      // 如果 x[i] 大于 0，则 j 自增
      if (x[i] > 0) j++;
      // 如果 omega[i] 等于 0 且 x[i] 不等于 0，则返回 0
      if (omega[i] == 0. && x[i]) return 0.;
      // 如果 x[i] 等于 m[i] 或者 omega[i] 等于 0，则 em 自增
      if (x[i] == m[i] || omega[i] == 0.) em++;
      // 如果 i 大于 0 且 omega[i] 不等于 omega[i-1]，则 central 置为 0
      if (i > 0 && omega[i] != omega[i-1]) central = 0;
   }

   // 如果 n 等于 0 或者 em 等于 colors，则返回 1
   if (n == 0 || em == colors) return 1.;

   // 如果 central 等于 1
   if (central) {
      // 所有 omega 均相等，这是多变量中心超几何分布
      // 初始化局部变量 sx 和 sm，并设置 p 为 1
      int32_t sx = n,  sm = N;
      double p = 1.;
      // 遍历 colors - 1 次
      for (i = 0; i < colors - 1; i++) {
         // 使用单变量超几何分布的 probability 方法 (usedcolors-1) 次
         p *= CWalleniusNCHypergeometric(sx, m[i], sm, 1.).probability(x[i]);
         // 更新 sx 和 sm 的值
         sx -= x[i];  sm -= m[i];
      }
      // 返回 p
      return p;
   }

   // 如果 j 等于 1，则返回 binoexpand() 的结果
   if (j == 1) { 
      return binoexpand();
   }

   // 查找参数
   findpars();
   // 如果 w 小于 0.04 且 E 小于 10 且 ((!em 或者 w 大于 0.004))
   if (w < 0.04 && E < 10 && (!em || w > 0.004)) {
      // 返回 laplace() 的结果
      return laplace();
   }

   // 返回 integrate() 的结果
   return integrate();
}
// 计算多元Wallenius非中心超几何分布的均值和方差，通过计算所有x值的组合。
// 返回值为所有概率的总和。该值与1的偏差衡量了精度。
// 将均值返回给mu[0...colors-1]
// 将方差返回给variance[0...colors-1]
double CMultiWalleniusNCHypergeometricMoments::moments(double * mu, double * variance, int32_t * combinations) {
   double sumf;                        // 所有f(x)值的总和
   int32_t msum;                       // 临时求和
   int i;                              // 循环计数器

   // 获取近似均值
   mean(sx);

   // 将均值四舍五入为整数
   for (i=0; i < colors; i++) {
      xm[i] = (int32_t)(sx[i]+0.4999999);
   }

   // 设置递归循环的初始条件
   for (i=colors-1, msum=0; i >= 0; i--) {
      remaining[i] = msum;  msum += m[i];
   }
   for (i=0; i<colors; i++)  sx[i] = sxx[i] = 0.;
   sn = 0;

   // 递归循环计算总和
   sumf = loop(n, 0);

   // 计算均值和方差
   for (i = 0; i < colors; i++) {
      mu[i] = sx[i]/sumf;
      variance[i] = sxx[i]/sumf - sx[i]*sx[i]/(sumf*sumf);
   }

   // 返回组合数和总和
   if (combinations) *combinations = sn;
   return sumf;
}

// 递归函数，用于遍历所有x值的组合，被moments()函数调用
double CMultiWalleniusNCHypergeometricMoments::loop(int32_t n, int c) {
   int32_t x, x0;                      // 第c种颜色的x值
   int32_t xmin, xmax;                 // x[c]的最小值和最大值
   double s1, s2, sum = 0.;            // f(x)值的总和
   int i;                              // 循环计数器

   if (c < colors-1) {
      // 不是最后一种颜色
      // 计算给定x[0]..x[c-1]时x[c]的最小值和最大值
      xmin = n - remaining[c];  if (xmin < 0) xmin = 0;
      xmax = m[c];  if (xmax > n) xmax = n;
      x0 = xm[c];  if (x0 < xmin) x0 = xmin;  if (x0 > xmax) x0 = xmax;
      // 循环处理所有x[c]从均值开始的情况
      for (x = x0, s2 = 0.; x <= xmax; x++) {
         xi[c] = x;
         sum += s1 = loop(n-x, c+1);             // 递归调用处理剩余颜色
         if (s1 < accuracy && s1 < s2) break;    // 当值变得可忽略时停止
         s2 = s1;
      }
      // 循环处理所有x[c]从均值减小的情况
      for (x = x0-1; x >= xmin; x--) {
         xi[c] = x;
         sum += s1 = loop(n-x, c+1);             // 递归调用处理剩余颜色
         if (s1 < accuracy && s1 < s2) break;    // 当值变得可忽略时停止
         s2 = s1;
      }
   }
   else {
      // 最后一种颜色
      xi[c] = n;
      s1 = probability(xi);
      for (i=0; i < colors; i++) {
         sx[i]  += s1 * xi[i];
         sxx[i] += s1 * xi[i] * xi[i];
      }
      sn++;
      sum = s1;
   }
   return sum;
}
```