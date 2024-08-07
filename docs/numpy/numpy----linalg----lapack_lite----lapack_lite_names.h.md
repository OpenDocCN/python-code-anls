# `.\numpy\numpy\linalg\lapack_lite\lapack_lite_names.h`

```py
/*
 * NOTE: This is generated code. Look in numpy/linalg/lapack_lite for
 *       information on remaking this file.
 */
/*
 * This file renames all BLAS/LAPACK and f2c symbols to avoid
 * dynamic symbol name conflicts, in cases where e.g.
 * integer sizes do not match with 'standard' ABI.
 */

// 定义了一系列宏，用于重命名 BLAS/LAPACK 和 f2c 符号，避免动态符号名称冲突
#define caxpy_ BLAS_FUNC(caxpy)
#define ccopy_ BLAS_FUNC(ccopy)
#define cdotc_ BLAS_FUNC(cdotc)
#define cdotu_ BLAS_FUNC(cdotu)
#define cgebak_ BLAS_FUNC(cgebak)
#define cgebal_ BLAS_FUNC(cgebal)
#define cgebd2_ BLAS_FUNC(cgebd2)
#define cgebrd_ BLAS_FUNC(cgebrd)
#define cgeev_ BLAS_FUNC(cgeev)
#define cgehd2_ BLAS_FUNC(cgehd2)
#define cgehrd_ BLAS_FUNC(cgehrd)
#define cgelq2_ BLAS_FUNC(cgelq2)
#define cgelqf_ BLAS_FUNC(cgelqf)
#define cgelsd_ BLAS_FUNC(cgelsd)
#define cgemm_ BLAS_FUNC(cgemm)
#define cgemv_ BLAS_FUNC(cgemv)
#define cgeqr2_ BLAS_FUNC(cgeqr2)
#define cgeqrf_ BLAS_FUNC(cgeqrf)
#define cgerc_ BLAS_FUNC(cgerc)
#define cgeru_ BLAS_FUNC(cgeru)
#define cgesdd_ BLAS_FUNC(cgesdd)
#define cgesv_ BLAS_FUNC(cgesv)
#define cgetf2_ BLAS_FUNC(cgetf2)
#define cgetrf_ BLAS_FUNC(cgetrf)
#define cgetrs_ BLAS_FUNC(cgetrs)
#define cheevd_ BLAS_FUNC(cheevd)
#define chemv_ BLAS_FUNC(chemv)
#define cher2_ BLAS_FUNC(cher2)
#define cher2k_ BLAS_FUNC(cher2k)
#define cherk_ BLAS_FUNC(cherk)
#define chetd2_ BLAS_FUNC(chetd2)
#define chetrd_ BLAS_FUNC(chetrd)
#define chseqr_ BLAS_FUNC(chseqr)
#define clabrd_ BLAS_FUNC(clabrd)
#define clacgv_ BLAS_FUNC(clacgv)
#define clacp2_ BLAS_FUNC(clacp2)
#define clacpy_ BLAS_FUNC(clacpy)
#define clacrm_ BLAS_FUNC(clacrm)
#define cladiv_ BLAS_FUNC(cladiv)
#define claed0_ BLAS_FUNC(claed0)
#define claed7_ BLAS_FUNC(claed7)
#define claed8_ BLAS_FUNC(claed8)
#define clahqr_ BLAS_FUNC(clahqr)
#define clahr2_ BLAS_FUNC(clahr2)
#define clals0_ BLAS_FUNC(clals0)
#define clalsa_ BLAS_FUNC(clalsa)
#define clalsd_ BLAS_FUNC(clalsd)
#define clange_ BLAS_FUNC(clange)
#define clanhe_ BLAS_FUNC(clanhe)
#define claqr0_ BLAS_FUNC(claqr0)
#define claqr1_ BLAS_FUNC(claqr1)
#define claqr2_ BLAS_FUNC(claqr2)
#define claqr3_ BLAS_FUNC(claqr3)
#define claqr4_ BLAS_FUNC(claqr4)
#define claqr5_ BLAS_FUNC(claqr5)
#define clarcm_ BLAS_FUNC(clarcm)
#define clarf_ BLAS_FUNC(clarf)
#define clarfb_ BLAS_FUNC(clarfb)
#define clarfg_ BLAS_FUNC(clarfg)
#define clarft_ BLAS_FUNC(clarft)
#define clartg_ BLAS_FUNC(clartg)
#define clascl_ BLAS_FUNC(clascl)
#define claset_ BLAS_FUNC(claset)
#define clasr_ BLAS_FUNC(clasr)
#define classq_ BLAS_FUNC(classq)
#define claswp_ BLAS_FUNC(claswp)
#define clatrd_ BLAS_FUNC(clatrd)
#define clatrs_ BLAS_FUNC(clatrs)
#define clauu2_ BLAS_FUNC(clauu2)
#define clauum_ BLAS_FUNC(clauum)
#define cpotf2_ BLAS_FUNC(cpotf2)
#define cpotrf_ BLAS_FUNC(cpotrf)
#define cpotri_ BLAS_FUNC(cpotri)
#define cpotrs_ BLAS_FUNC(cpotrs)
#define crot_ BLAS_FUNC(crot)
#define cscal_ BLAS_FUNC(cscal)
#define csrot_ BLAS_FUNC(csrot)
#define csscal_ BLAS_FUNC(csscal)
#define cstedc_ BLAS_FUNC(cstedc)
#define csteqr_ BLAS_FUNC(csteqr)
// 定义宏，将函数名映射到 BLAS_FUNC 宏定义的函数名
#define cswap_ BLAS_FUNC(cswap)
#define ctrevc_ BLAS_FUNC(ctrevc)
#define ctrexc_ BLAS_FUNC(ctrexc)
#define ctrmm_ BLAS_FUNC(ctrmm)
#define ctrmv_ BLAS_FUNC(ctrmv)
#define ctrsm_ BLAS_FUNC(ctrsm)
#define ctrsv_ BLAS_FUNC(ctrsv)
#define ctrti2_ BLAS_FUNC(ctrti2)
#define ctrtri_ BLAS_FUNC(ctrtri)
#define cung2r_ BLAS_FUNC(cung2r)
#define cungbr_ BLAS_FUNC(cungbr)
#define cunghr_ BLAS_FUNC(cunghr)
#define cungl2_ BLAS_FUNC(cungl2)
#define cunglq_ BLAS_FUNC(cunglq)
#define cungqr_ BLAS_FUNC(cungqr)
#define cunm2l_ BLAS_FUNC(cunm2l)
#define cunm2r_ BLAS_FUNC(cunm2r)
#define cunmbr_ BLAS_FUNC(cunmbr)
#define cunmhr_ BLAS_FUNC(cunmhr)
#define cunml2_ BLAS_FUNC(cunml2)
#define cunmlq_ BLAS_FUNC(cunmlq)
#define cunmql_ BLAS_FUNC(cunmql)
#define cunmqr_ BLAS_FUNC(cunmqr)
#define cunmtr_ BLAS_FUNC(cunmtr)
#define daxpy_ BLAS_FUNC(daxpy)
#define dbdsdc_ BLAS_FUNC(dbdsdc)
#define dbdsqr_ BLAS_FUNC(dbdsqr)
#define dcabs1_ BLAS_FUNC(dcabs1)
#define dcopy_ BLAS_FUNC(dcopy)
#define ddot_ BLAS_FUNC(ddot)
#define dgebak_ BLAS_FUNC(dgebak)
#define dgebal_ BLAS_FUNC(dgebal)
#define dgebd2_ BLAS_FUNC(dgebd2)
#define dgebrd_ BLAS_FUNC(dgebrd)
#define dgeev_ BLAS_FUNC(dgeev)
#define dgehd2_ BLAS_FUNC(dgehd2)
#define dgehrd_ BLAS_FUNC(dgehrd)
#define dgelq2_ BLAS_FUNC(dgelq2)
#define dgelqf_ BLAS_FUNC(dgelqf)
#define dgelsd_ BLAS_FUNC(dgelsd)
#define dgemm_ BLAS_FUNC(dgemm)
#define dgemv_ BLAS_FUNC(dgemv)
#define dgeqr2_ BLAS_FUNC(dgeqr2)
#define dgeqrf_ BLAS_FUNC(dgeqrf)
#define dger_ BLAS_FUNC(dger)
#define dgesdd_ BLAS_FUNC(dgesdd)
#define dgesv_ BLAS_FUNC(dgesv)
#define dgetf2_ BLAS_FUNC(dgetf2)
#define dgetrf_ BLAS_FUNC(dgetrf)
#define dgetrs_ BLAS_FUNC(dgetrs)
#define dhseqr_ BLAS_FUNC(dhseqr)
#define disnan_ BLAS_FUNC(disnan)
#define dlabad_ BLAS_FUNC(dlabad)
#define dlabrd_ BLAS_FUNC(dlabrd)
#define dlacpy_ BLAS_FUNC(dlacpy)
#define dladiv_ BLAS_FUNC(dladiv)
#define dlae2_ BLAS_FUNC(dlae2)
#define dlaed0_ BLAS_FUNC(dlaed0)
#define dlaed1_ BLAS_FUNC(dlaed1)
#define dlaed2_ BLAS_FUNC(dlaed2)
#define dlaed3_ BLAS_FUNC(dlaed3)
#define dlaed4_ BLAS_FUNC(dlaed4)
#define dlaed5_ BLAS_FUNC(dlaed5)
#define dlaed6_ BLAS_FUNC(dlaed6)
#define dlaed7_ BLAS_FUNC(dlaed7)
#define dlaed8_ BLAS_FUNC(dlaed8)
#define dlaed9_ BLAS_FUNC(dlaed9)
#define dlaeda_ BLAS_FUNC(dlaeda)
#define dlaev2_ BLAS_FUNC(dlaev2)
#define dlaexc_ BLAS_FUNC(dlaexc)
#define dlahqr_ BLAS_FUNC(dlahqr)
#define dlahr2_ BLAS_FUNC(dlahr2)
#define dlaisnan_ BLAS_FUNC(dlaisnan)
#define dlaln2_ BLAS_FUNC(dlaln2)
#define dlals0_ BLAS_FUNC(dlals0)
#define dlalsa_ BLAS_FUNC(dlalsa)
#define dlalsd_ BLAS_FUNC(dlalsd)
#define dlamc1_ BLAS_FUNC(dlamc1)
#define dlamc2_ BLAS_FUNC(dlamc2)
#define dlamc3_ BLAS_FUNC(dlamc3)
#define dlamc4_ BLAS_FUNC(dlamc4)
#define dlamc5_ BLAS_FUNC(dlamc5)
#define dlamch_ BLAS_FUNC(dlamch)
#define dlamrg_ BLAS_FUNC(dlamrg)
#define dlange_ BLAS_FUNC(dlange)
#define dlanst_ BLAS_FUNC(dlanst)
#define dlansy_ BLAS_FUNC(dlansy)
#define dlanv2_ BLAS_FUNC(dlanv2)
#define dlapy2_ BLAS_FUNC(dlapy2)
// 定义各个 BLAS 函数的宏，它们是对应于 LAPACK 和 BLAS 数学库中特定函数的符号名称映射
#define dlapy3_ BLAS_FUNC(dlapy3)
#define dlaqr0_ BLAS_FUNC(dlaqr0)
#define dlaqr1_ BLAS_FUNC(dlaqr1)
#define dlaqr2_ BLAS_FUNC(dlaqr2)
#define dlaqr3_ BLAS_FUNC(dlaqr3)
#define dlaqr4_ BLAS_FUNC(dlaqr4)
#define dlaqr5_ BLAS_FUNC(dlaqr5)
#define dlarf_ BLAS_FUNC(dlarf)
#define dlarfb_ BLAS_FUNC(dlarfb)
#define dlarfg_ BLAS_FUNC(dlarfg)
#define dlarft_ BLAS_FUNC(dlarft)
#define dlarfx_ BLAS_FUNC(dlarfx)
#define dlartg_ BLAS_FUNC(dlartg)
#define dlas2_ BLAS_FUNC(dlas2)
#define dlascl_ BLAS_FUNC(dlascl)
#define dlasd0_ BLAS_FUNC(dlasd0)
#define dlasd1_ BLAS_FUNC(dlasd1)
#define dlasd2_ BLAS_FUNC(dlasd2)
#define dlasd3_ BLAS_FUNC(dlasd3)
#define dlasd4_ BLAS_FUNC(dlasd4)
#define dlasd5_ BLAS_FUNC(dlasd5)
#define dlasd6_ BLAS_FUNC(dlasd6)
#define dlasd7_ BLAS_FUNC(dlasd7)
#define dlasd8_ BLAS_FUNC(dlasd8)
#define dlasda_ BLAS_FUNC(dlasda)
#define dlasdq_ BLAS_FUNC(dlasdq)
#define dlasdt_ BLAS_FUNC(dlasdt)
#define dlaset_ BLAS_FUNC(dlaset)
#define dlasq1_ BLAS_FUNC(dlasq1)
#define dlasq2_ BLAS_FUNC(dlasq2)
#define dlasq3_ BLAS_FUNC(dlasq3)
#define dlasq4_ BLAS_FUNC(dlasq4)
#define dlasq5_ BLAS_FUNC(dlasq5)
#define dlasq6_ BLAS_FUNC(dlasq6)
#define dlasr_ BLAS_FUNC(dlasr)
#define dlasrt_ BLAS_FUNC(dlasrt)
#define dlassq_ BLAS_FUNC(dlassq)
#define dlasv2_ BLAS_FUNC(dlasv2)
#define dlaswp_ BLAS_FUNC(dlaswp)
#define dlasy2_ BLAS_FUNC(dlasy2)
#define dlatrd_ BLAS_FUNC(dlatrd)
#define dlauu2_ BLAS_FUNC(dlauu2)
#define dlauum_ BLAS_FUNC(dlauum)
#define dnrm2_ BLAS_FUNC(dnrm2)
#define dorg2r_ BLAS_FUNC(dorg2r)
#define dorgbr_ BLAS_FUNC(dorgbr)
#define dorghr_ BLAS_FUNC(dorghr)
#define dorgl2_ BLAS_FUNC(dorgl2)
#define dorglq_ BLAS_FUNC(dorglq)
#define dorgqr_ BLAS_FUNC(dorgqr)
#define dorm2l_ BLAS_FUNC(dorm2l)
#define dorm2r_ BLAS_FUNC(dorm2r)
#define dormbr_ BLAS_FUNC(dormbr)
#define dormhr_ BLAS_FUNC(dormhr)
#define dorml2_ BLAS_FUNC(dorml2)
#define dormlq_ BLAS_FUNC(dormlq)
#define dormql_ BLAS_FUNC(dormql)
#define dormqr_ BLAS_FUNC(dormqr)
#define dormtr_ BLAS_FUNC(dormtr)
#define dpotf2_ BLAS_FUNC(dpotf2)
#define dpotrf_ BLAS_FUNC(dpotrf)
#define dpotri_ BLAS_FUNC(dpotri)
#define dpotrs_ BLAS_FUNC(dpotrs)
#define drot_ BLAS_FUNC(drot)
#define dscal_ BLAS_FUNC(dscal)
#define dstedc_ BLAS_FUNC(dstedc)
#define dsteqr_ BLAS_FUNC(dsteqr)
#define dsterf_ BLAS_FUNC(dsterf)
#define dswap_ BLAS_FUNC(dswap)
#define dsyevd_ BLAS_FUNC(dsyevd)
#define dsymv_ BLAS_FUNC(dsymv)
#define dsyr2_ BLAS_FUNC(dsyr2)
#define dsyr2k_ BLAS_FUNC(dsyr2k)
#define dsyrk_ BLAS_FUNC(dsyrk)
#define dsytd2_ BLAS_FUNC(dsytd2)
#define dsytrd_ BLAS_FUNC(dsytrd)
#define dtrevc_ BLAS_FUNC(dtrevc)
#define dtrexc_ BLAS_FUNC(dtrexc)
#define dtrmm_ BLAS_FUNC(dtrmm)
#define dtrmv_ BLAS_FUNC(dtrmv)
#define dtrsm_ BLAS_FUNC(dtrsm)
#define dtrti2_ BLAS_FUNC(dtrti2)
#define dtrtri_ BLAS_FUNC(dtrtri)
#define dzasum_ BLAS_FUNC(dzasum)
#define dznrm2_ BLAS_FUNC(dznrm2)
#define icamax_ BLAS_FUNC(icamax)
#define idamax_ BLAS_FUNC(idamax)
#define ieeeck_ BLAS_FUNC(ieeeck)
#define ilaclc_ BLAS_FUNC(ilaclc)
#define ilaclr_ BLAS_FUNC(ilaclr)
// 定义宏 ilaclr_，展开为 BLAS_FUNC(ilaclr) 的内容

#define iladlc_ BLAS_FUNC(iladlc)
// 定义宏 iladlc_，展开为 BLAS_FUNC(iladlc) 的内容

#define iladlr_ BLAS_FUNC(iladlr)
// 定义宏 iladlr_，展开为 BLAS_FUNC(iladlr) 的内容

#define ilaenv_ BLAS_FUNC(ilaenv)
// 定义宏 ilaenv_，展开为 BLAS_FUNC(ilaenv) 的内容

#define ilaslc_ BLAS_FUNC(ilaslc)
// 定义宏 ilaslc_，展开为 BLAS_FUNC(ilaslc) 的内容

#define ilaslr_ BLAS_FUNC(ilaslr)
// 定义宏 ilaslr_，展开为 BLAS_FUNC(ilaslr) 的内容

#define ilazlc_ BLAS_FUNC(ilazlc)
// 定义宏 ilazlc_，展开为 BLAS_FUNC(ilazlc) 的内容

#define ilazlr_ BLAS_FUNC(ilazlr)
// 定义宏 ilazlr_，展开为 BLAS_FUNC(ilazlr) 的内容

#define iparmq_ BLAS_FUNC(iparmq)
// 定义宏 iparmq_，展开为 BLAS_FUNC(iparmq) 的内容

#define isamax_ BLAS_FUNC(isamax)
// 定义宏 isamax_，展开为 BLAS_FUNC(isamax) 的内容

#define izamax_ BLAS_FUNC(izamax)
// 定义宏 izamax_，展开为 BLAS_FUNC(izamax) 的内容

#define lsame_ BLAS_FUNC(lsame)
// 定义宏 lsame_，展开为 BLAS_FUNC(lsame) 的内容

#define saxpy_ BLAS_FUNC(saxpy)
// 定义宏 saxpy_，展开为 BLAS_FUNC(saxpy) 的内容

#define sbdsdc_ BLAS_FUNC(sbdsdc)
// 定义宏 sbdsdc_，展开为 BLAS_FUNC(sbdsdc) 的内容

#define sbdsqr_ BLAS_FUNC(sbdsqr)
// 定义宏 sbdsqr_，展开为 BLAS_FUNC(sbdsqr) 的内容

#define scabs1_ BLAS_FUNC(scabs1)
// 定义宏 scabs1_，展开为 BLAS_FUNC(scabs1) 的内容

#define scasum_ BLAS_FUNC(scasum)
// 定义宏 scasum_，展开为 BLAS_FUNC(scasum) 的内容

#define scnrm2_ BLAS_FUNC(scnrm2)
// 定义宏 scnrm2_，展开为 BLAS_FUNC(scnrm2) 的内容

#define scopy_ BLAS_FUNC(scopy)
// 定义宏 scopy_，展开为 BLAS_FUNC(scopy) 的内容

#define sdot_ BLAS_FUNC(sdot)
// 定义宏 sdot_，展开为 BLAS_FUNC(sdot) 的内容

#define sgebak_ BLAS_FUNC(sgebak)
// 定义宏 sgebak_，展开为 BLAS_FUNC(sgebak) 的内容

#define sgebal_ BLAS_FUNC(sgebal)
// 定义宏 sgebal_，展开为 BLAS_FUNC(sgebal) 的内容

#define sgebd2_ BLAS_FUNC(sgebd2)
// 定义宏 sgebd2_，展开为 BLAS_FUNC(sgebd2) 的内容

#define sgebrd_ BLAS_FUNC(sgebrd)
// 定义宏 sgebrd_，展开为 BLAS_FUNC(sgebrd) 的内容

#define sgeev_ BLAS_FUNC(sgeev)
// 定义宏 sgeev_，展开为 BLAS_FUNC(sgeev) 的内容

#define sgehd2_ BLAS_FUNC(sgehd2)
// 定义宏 sgehd2_，展开为 BLAS_FUNC(sgehd2) 的内容

#define sgehrd_ BLAS_FUNC(sgehrd)
// 定义宏 sgehrd_，展开为 BLAS_FUNC(sgehrd) 的内容

#define sgelq2_ BLAS_FUNC(sgelq2)
// 定义宏 sgelq2_，展开为 BLAS_FUNC(sgelq2) 的内容

#define sgelqf_ BLAS_FUNC(sgelqf)
// 定义宏 sgelqf_，展开为 BLAS_FUNC(sgelqf) 的内容

#define sgelsd_ BLAS_FUNC(sgelsd)
// 定义宏 sgelsd_，展开为 BLAS_FUNC(sgelsd) 的内容

#define sgemm_ BLAS_FUNC(sgemm)
// 定义宏 sgemm_，展开为 BLAS_FUNC(sgemm) 的内容

#define sgemv_ BLAS_FUNC(sgemv)
// 定义宏 sgemv_，展开为 BLAS_FUNC(sgemv) 的内容

#define sgeqr2_ BLAS_FUNC(sgeqr2)
// 定义宏 sgeqr2_，展开为 BLAS_FUNC(sgeqr2) 的内容

#define sgeqrf_ BLAS_FUNC(sgeqrf)
// 定义宏 sgeqrf_，展开为 BLAS_FUNC(sgeqrf) 的内容

#define sger_ BLAS_FUNC(sger)
// 定义宏 sger_，展开为 BLAS_FUNC(sger) 的内容

#define sgesdd_ BLAS_FUNC(sgesdd)
// 定义宏 sgesdd_，展开为 BLAS_FUNC(sgesdd) 的内容

#define sgesv_ BLAS_FUNC(sgesv)
// 定义宏 sgesv_，展开为 BLAS_FUNC(sgesv) 的内容

#define sgetf2_ BLAS_FUNC(sgetf2)
// 定义宏 sgetf2_，展开为 BLAS_FUNC(sgetf2) 的内容

#define sgetrf_ BLAS_FUNC(sgetrf)
// 定义宏 sgetrf_，展开为 BLAS_FUNC(sgetrf) 的内容

#define sgetrs_ BLAS_FUNC(sgetrs)
// 定义宏 sgetrs_，展开为 BLAS_FUNC(sgetrs) 的内容

#define shseqr_ BLAS_FUNC(shseqr)
// 定义宏 shseqr_，展开为 BLAS_FUNC(shseqr) 的内容

#define sisnan_ BLAS_FUNC(sisnan)
// 定义宏 sisnan_，展开为 BLAS_FUNC(sisnan) 的内容

#define slabad_ BLAS_FUNC(slabad)
// 定义宏 slabad_，展开为 BLAS_FUNC(slabad) 的内容

#define slabrd_ BLAS_FUNC(slabrd)
// 定义宏 slabrd_，展开为 BLAS_FUNC(slabrd) 的内容

#define slacpy_ BLAS_FUNC(slacpy)
// 定义宏 slacpy_，展开为 BLAS_FUNC(slacpy) 的内容

#define sladiv_ BLAS_FUNC(sladiv)
// 定义宏 sladiv_，展开为 BLAS_FUNC(sladiv) 的内容

#define slae2_ BLAS_FUNC(slae2)
// 定义宏 slae2_，展开为 BLAS_FUNC(slae2) 的内容

#define slaed0_ BLAS_FUNC(slaed0)
// 定义宏 slaed0_，展开为 BLAS_FUNC(slaed0) 的内容

#define slaed1_ BLAS_FUNC(slaed1)
// 定义宏 slaed1_，展开为 BLAS_FUNC(slaed1) 的内容

#define slaed2_ BLAS_FUNC(slaed2)
// 定义宏 slaed2_，展开为 BLAS_FUNC(slaed2) 的内容

#define slaed3_ BLAS_FUNC(slaed3)
// 定义宏 slaed3_，展开为 BLAS_FUNC(slaed3) 的内容

#define slaed4_ BLAS_FUNC(slaed4)
// 定义宏 slaed4_，展开为 BLAS_FUNC(slaed4) 的内容

#define slaed5_ BLAS_FUNC(slaed5)
// 定义宏 slaed5_，展开为 BLAS_FUNC(slaed5) 的内容

#define slaed6_ BLAS_FUNC(slaed6)
// 定义宏 slaed6_，展开为 BLAS_FUNC(slaed6) 的内容

#define slaed7_ BLAS_FUNC(slaed7)
// 定义宏 slaed7_，展开为 BLAS_FUNC(slaed7) 的内容

#define slaed8_ BLAS_FUNC(slaed8)
// 定义宏 slaed8_，展开为 BLAS_FUNC(slaed8) 的内容

#define slaed
// 定义各个 BLAS 函数的别名，使用 BLAS_FUNC 宏进行定义
#define slarfg_ BLAS_FUNC(slarfg)
#define slarft_ BLAS_FUNC(slarft)
#define slarfx_ BLAS_FUNC(slarfx)
#define slartg_ BLAS_FUNC(slartg)
#define slas2_ BLAS_FUNC(slas2)
#define slascl_ BLAS_FUNC(slascl)
#define slasd0_ BLAS_FUNC(slasd0)
#define slasd1_ BLAS_FUNC(slasd1)
#define slasd2_ BLAS_FUNC(slasd2)
#define slasd3_ BLAS_FUNC(slasd3)
#define slasd4_ BLAS_FUNC(slasd4)
#define slasd5_ BLAS_FUNC(slasd5)
#define slasd6_ BLAS_FUNC(slasd6)
#define slasd7_ BLAS_FUNC(slasd7)
#define slasd8_ BLAS_FUNC(slasd8)
#define slasda_ BLAS_FUNC(slasda)
#define slasdq_ BLAS_FUNC(slasdq)
#define slasdt_ BLAS_FUNC(slasdt)
#define slaset_ BLAS_FUNC(slaset)
#define slasq1_ BLAS_FUNC(slasq1)
#define slasq2_ BLAS_FUNC(slasq2)
#define slasq3_ BLAS_FUNC(slasq3)
#define slasq4_ BLAS_FUNC(slasq4)
#define slasq5_ BLAS_FUNC(slasq5)
#define slasq6_ BLAS_FUNC(slasq6)
#define slasr_ BLAS_FUNC(slasr)
#define slasrt_ BLAS_FUNC(slasrt)
#define slassq_ BLAS_FUNC(slassq)
#define slasv2_ BLAS_FUNC(slasv2)
#define slaswp_ BLAS_FUNC(slaswp)
#define slasy2_ BLAS_FUNC(slasy2)
#define slatrd_ BLAS_FUNC(slatrd)
#define slauu2_ BLAS_FUNC(slauu2)
#define slauum_ BLAS_FUNC(slauum)
#define snrm2_ BLAS_FUNC(snrm2)
#define sorg2r_ BLAS_FUNC(sorg2r)
#define sorgbr_ BLAS_FUNC(sorgbr)
#define sorghr_ BLAS_FUNC(sorghr)
#define sorgl2_ BLAS_FUNC(sorgl2)
#define sorglq_ BLAS_FUNC(sorglq)
#define sorgqr_ BLAS_FUNC(sorgqr)
#define sorm2l_ BLAS_FUNC(sorm2l)
#define sorm2r_ BLAS_FUNC(sorm2r)
#define sormbr_ BLAS_FUNC(sormbr)
#define sormhr_ BLAS_FUNC(sormhr)
#define sorml2_ BLAS_FUNC(sorml2)
#define sormlq_ BLAS_FUNC(sormlq)
#define sormql_ BLAS_FUNC(sormql)
#define sormqr_ BLAS_FUNC(sormqr)
#define sormtr_ BLAS_FUNC(sormtr)
#define spotf2_ BLAS_FUNC(spotf2)
#define spotrf_ BLAS_FUNC(spotrf)
#define spotri_ BLAS_FUNC(spotri)
#define spotrs_ BLAS_FUNC(spotrs)
#define srot_ BLAS_FUNC(srot)
#define sscal_ BLAS_FUNC(sscal)
#define sstedc_ BLAS_FUNC(sstedc)
#define ssteqr_ BLAS_FUNC(ssteqr)
#define ssterf_ BLAS_FUNC(ssterf)
#define sswap_ BLAS_FUNC(sswap)
#define ssyevd_ BLAS_FUNC(ssyevd)
#define ssymv_ BLAS_FUNC(ssymv)
#define ssyr2_ BLAS_FUNC(ssyr2)
#define ssyr2k_ BLAS_FUNC(ssyr2k)
#define ssyrk_ BLAS_FUNC(ssyrk)
#define ssytd2_ BLAS_FUNC(ssytd2)
#define ssytrd_ BLAS_FUNC(ssytrd)
#define strevc_ BLAS_FUNC(strevc)
#define strexc_ BLAS_FUNC(strexc)
#define strmm_ BLAS_FUNC(strmm)
#define strmv_ BLAS_FUNC(strmv)
#define strsm_ BLAS_FUNC(strsm)
#define strti2_ BLAS_FUNC(strti2)
#define strtri_ BLAS_FUNC(strtri)
#define xerbla_ BLAS_FUNC(xerbla)
#define zaxpy_ BLAS_FUNC(zaxpy)
#define zcopy_ BLAS_FUNC(zcopy)
#define zdotc_ BLAS_FUNC(zdotc)
#define zdotu_ BLAS_FUNC(zdotu)
#define zdrot_ BLAS_FUNC(zdrot)
#define zdscal_ BLAS_FUNC(zdscal)
#define zgebak_ BLAS_FUNC(zgebak)
#define zgebal_ BLAS_FUNC(zgebal)
#define zgebd2_ BLAS_FUNC(zgebd2)
#define zgebrd_ BLAS_FUNC(zgebrd)
#define zgeev_ BLAS_FUNC(zgeev)
#define zgehd2_ BLAS_FUNC(zgehd2)
#define zgehrd_ BLAS_FUNC(zgehrd)
#define zgelq2_ BLAS_FUNC(zgelq2)
// 定义宏，用于重命名 BLAS 函数名，以下是一系列宏定义
#define zgelqf_ BLAS_FUNC(zgelqf)
#define zgelsd_ BLAS_FUNC(zgelsd)
#define zgemm_ BLAS_FUNC(zgemm)
#define zgemv_ BLAS_FUNC(zgemv)
#define zgeqr2_ BLAS_FUNC(zgeqr2)
#define zgeqrf_ BLAS_FUNC(zgeqrf)
#define zgerc_ BLAS_FUNC(zgerc)
#define zgeru_ BLAS_FUNC(zgeru)
#define zgesdd_ BLAS_FUNC(zgesdd)
#define zgesv_ BLAS_FUNC(zgesv)
#define zgetf2_ BLAS_FUNC(zgetf2)
#define zgetrf_ BLAS_FUNC(zgetrf)
#define zgetrs_ BLAS_FUNC(zgetrs)
#define zheevd_ BLAS_FUNC(zheevd)
#define zhemv_ BLAS_FUNC(zhemv)
#define zher2_ BLAS_FUNC(zher2)
#define zher2k_ BLAS_FUNC(zher2k)
#define zherk_ BLAS_FUNC(zherk)
#define zhetd2_ BLAS_FUNC(zhetd2)
#define zhetrd_ BLAS_FUNC(zhetrd)
#define zhseqr_ BLAS_FUNC(zhseqr)
#define zlabrd_ BLAS_FUNC(zlabrd)
#define zlacgv_ BLAS_FUNC(zlacgv)
#define zlacp2_ BLAS_FUNC(zlacp2)
#define zlacpy_ BLAS_FUNC(zlacpy)
#define zlacrm_ BLAS_FUNC(zlacrm)
#define zladiv_ BLAS_FUNC(zladiv)
#define zlaed0_ BLAS_FUNC(zlaed0)
#define zlaed7_ BLAS_FUNC(zlaed7)
#define zlaed8_ BLAS_FUNC(zlaed8)
#define zlahqr_ BLAS_FUNC(zlahqr)
#define zlahr2_ BLAS_FUNC(zlahr2)
#define zlals0_ BLAS_FUNC(zlals0)
#define zlalsa_ BLAS_FUNC(zlalsa)
#define zlalsd_ BLAS_FUNC(zlalsd)
#define zlange_ BLAS_FUNC(zlange)
#define zlanhe_ BLAS_FUNC(zlanhe)
#define zlaqr0_ BLAS_FUNC(zlaqr0)
#define zlaqr1_ BLAS_FUNC(zlaqr1)
#define zlaqr2_ BLAS_FUNC(zlaqr2)
#define zlaqr3_ BLAS_FUNC(zlaqr3)
#define zlaqr4_ BLAS_FUNC(zlaqr4)
#define zlaqr5_ BLAS_FUNC(zlaqr5)
#define zlarcm_ BLAS_FUNC(zlarcm)
#define zlarf_ BLAS_FUNC(zlarf)
#define zlarfb_ BLAS_FUNC(zlarfb)
#define zlarfg_ BLAS_FUNC(zlarfg)
#define zlarft_ BLAS_FUNC(zlarft)
#define zlartg_ BLAS_FUNC(zlartg)
#define zlascl_ BLAS_FUNC(zlascl)
#define zlaset_ BLAS_FUNC(zlaset)
#define zlasr_ BLAS_FUNC(zlasr)
#define zlassq_ BLAS_FUNC(zlassq)
#define zlaswp_ BLAS_FUNC(zlaswp)
#define zlatrd_ BLAS_FUNC(zlatrd)
#define zlatrs_ BLAS_FUNC(zlatrs)
#define zlauu2_ BLAS_FUNC(zlauu2)
#define zlauum_ BLAS_FUNC(zlauum)
#define zpotf2_ BLAS_FUNC(zpotf2)
#define zpotrf_ BLAS_FUNC(zpotrf)
#define zpotri_ BLAS_FUNC(zpotri)
#define zpotrs_ BLAS_FUNC(zpotrs)
#define zrot_ BLAS_FUNC(zrot)
#define zscal_ BLAS_FUNC(zscal)
#define zstedc_ BLAS_FUNC(zstedc)
#define zsteqr_ BLAS_FUNC(zsteqr)
#define zswap_ BLAS_FUNC(zswap)
#define ztrevc_ BLAS_FUNC(ztrevc)
#define ztrexc_ BLAS_FUNC(ztrexc)
#define ztrmm_ BLAS_FUNC(ztrmm)
#define ztrmv_ BLAS_FUNC(ztrmv)
#define ztrsm_ BLAS_FUNC(ztrsm)
#define ztrsv_ BLAS_FUNC(ztrsv)
#define ztrti2_ BLAS_FUNC(ztrti2)
#define ztrtri_ BLAS_FUNC(ztrtri)
#define zung2r_ BLAS_FUNC(zung2r)
#define zungbr_ BLAS_FUNC(zungbr)
#define zunghr_ BLAS_FUNC(zunghr)
#define zungl2_ BLAS_FUNC(zungl2)
#define zunglq_ BLAS_FUNC(zunglq)
#define zungqr_ BLAS_FUNC(zungqr)
#define zunm2l_ BLAS_FUNC(zunm2l)
#define zunm2r_ BLAS_FUNC(zunm2r)
#define zunmbr_ BLAS_FUNC(zunmbr)
#define zunmhr_ BLAS_FUNC(zunmhr)
#define zunml2_ BLAS_FUNC(zunml2)
#define zunmlq_ BLAS_FUNC(zunmlq)
#define zunmql_ BLAS_FUNC(zunmql)
#define zunmqr_ BLAS_FUNC(zunmqr)
# 定义 zunmtr_ 符号，表示 BLAS_FUNC 函数的 zunmtr 实现
#define zunmtr_ BLAS_FUNC(zunmtr)

# 下面是一系列由 f2c.c 导出的符号名称重定义，这些符号名称都指向 numpy_lapack_lite 模块中对应的函数或变量。
# 例如，将 abort_ 重定义为 numpy_lapack_lite_abort_
#define abort_ numpy_lapack_lite_abort_
#define c_abs numpy_lapack_lite_c_abs
#define c_cos numpy_lapack_lite_c_cos
#define c_div numpy_lapack_lite_c_div
#define c_exp numpy_lapack_lite_c_exp
#define c_log numpy_lapack_lite_c_log
#define c_sin numpy_lapack_lite_c_sin
#define c_sqrt numpy_lapack_lite_c_sqrt
#define d_abs numpy_lapack_lite_d_abs
#define d_acos numpy_lapack_lite_d_acos
#define d_asin numpy_lapack_lite_d_asin
#define d_atan numpy_lapack_lite_d_atan
#define d_atn2 numpy_lapack_lite_d_atn2
#define d_cnjg numpy_lapack_lite_d_cnjg
#define d_cos numpy_lapack_lite_d_cos
#define d_cosh numpy_lapack_lite_d_cosh
#define d_dim numpy_lapack_lite_d_dim
#define d_exp numpy_lapack_lite_d_exp
#define d_imag numpy_lapack_lite_d_imag
#define d_int numpy_lapack_lite_d_int
#define d_lg10 numpy_lapack_lite_d_lg10
#define d_log numpy_lapack_lite_d_log
#define d_mod numpy_lapack_lite_d_mod
#define d_nint numpy_lapack_lite_d_nint
#define d_prod numpy_lapack_lite_d_prod
#define d_sign numpy_lapack_lite_d_sign
#define d_sin numpy_lapack_lite_d_sin
#define d_sinh numpy_lapack_lite_d_sinh
#define d_sqrt numpy_lapack_lite_d_sqrt
#define d_tan numpy_lapack_lite_d_tan
#define d_tanh numpy_lapack_lite_d_tanh
#define derf_ numpy_lapack_lite_derf_
#define derfc_ numpy_lapack_lite_derfc_
#define do_fio numpy_lapack_lite_do_fio
#define do_lio numpy_lapack_lite_do_lio
#define do_uio numpy_lapack_lite_do_uio
#define e_rdfe numpy_lapack_lite_e_rdfe
#define e_rdue numpy_lapack_lite_e_rdue
#define e_rsfe numpy_lapack_lite_e_rsfe
#define e_rsfi numpy_lapack_lite_e_rsfi
#define e_rsle numpy_lapack_lite_e_rsle
#define e_rsli numpy_lapack_lite_e_rsli
#define e_rsue numpy_lapack_lite_e_rsue
#define e_wdfe numpy_lapack_lite_e_wdfe
#define e_wdue numpy_lapack_lite_e_wdue
#define e_wsfe numpy_lapack_lite_e_wsfe
#define e_wsfi numpy_lapack_lite_e_wsfi
#define e_wsle numpy_lapack_lite_e_wsle
#define e_wsli numpy_lapack_lite_e_wsli
#define e_wsue numpy_lapack_lite_e_wsue
#define ef1asc_ numpy_lapack_lite_ef1asc_
#define ef1cmc_ numpy_lapack_lite_ef1cmc_
#define erf_ numpy_lapack_lite_erf_
#define erfc_ numpy_lapack_lite_erfc_
#define f__cabs numpy_lapack_lite_f__cabs
#define f__cabsf numpy_lapack_lite_f__cabsf
#define f_back numpy_lapack_lite_f_back
#define f_clos numpy_lapack_lite_f_clos
#define f_end numpy_lapack_lite_f_end
#define f_exit numpy_lapack_lite_f_exit
#define f_inqu numpy_lapack_lite_f_inqu
#define f_open numpy_lapack_lite_f_open
#define f_rew numpy_lapack_lite_f_rew
#define flush_ numpy_lapack_lite_flush_
#define getarg_ numpy_lapack_lite_getarg_
#define getenv_ numpy_lapack_lite_getenv_
#define h_abs numpy_lapack_lite_h_abs
#define h_dim numpy_lapack_lite_h_dim
#define h_dnnt numpy_lapack_lite_h_dnnt
#define h_indx numpy_lapack_lite_h_indx
#define h_len numpy_lapack_lite_h_len
#define h_mod numpy_lapack_lite_h_mod
#define h_nint numpy_lapack_lite_h_nint
#define h_sign numpy_lapack_lite_h_sign
# 定义宏指令，用于将给定名称重新映射为对应的numpy_lapack_lite模块中的函数或变量
#define hl_ge numpy_lapack_lite_hl_ge
#define hl_gt numpy_lapack_lite_hl_gt
#define hl_le numpy_lapack_lite_hl_le
#define hl_lt numpy_lapack_lite_hl_lt
#define i_abs numpy_lapack_lite_i_abs
#define i_dim numpy_lapack_lite_i_dim
#define i_dnnt numpy_lapack_lite_i_dnnt
#define i_indx numpy_lapack_lite_i_indx
#define i_len numpy_lapack_lite_i_len
#define i_mod numpy_lapack_lite_i_mod
#define i_nint numpy_lapack_lite_i_nint
#define i_sign numpy_lapack_lite_i_sign
#define iargc_ numpy_lapack_lite_iargc_
#define l_ge numpy_lapack_lite_l_ge
#define l_gt numpy_lapack_lite_l_gt
#define l_le numpy_lapack_lite_l_le
#define l_lt numpy_lapack_lite_l_lt
#define pow_ci numpy_lapack_lite_pow_ci
#define pow_dd numpy_lapack_lite_pow_dd
#define pow_di numpy_lapack_lite_pow_di
#define pow_hh numpy_lapack_lite_pow_hh
#define pow_ii numpy_lapack_lite_pow_ii
#define pow_ri numpy_lapack_lite_pow_ri
#define pow_zi numpy_lapack_lite_pow_zi
#define pow_zz numpy_lapack_lite_pow_zz
#define r_abs numpy_lapack_lite_r_abs
#define r_acos numpy_lapack_lite_r_acos
#define r_asin numpy_lapack_lite_r_asin
#define r_atan numpy_lapack_lite_r_atan
#define r_atn2 numpy_lapack_lite_r_atn2
#define r_cnjg numpy_lapack_lite_r_cnjg
#define r_cos numpy_lapack_lite_r_cos
#define r_cosh numpy_lapack_lite_r_cosh
#define r_dim numpy_lapack_lite_r_dim
#define r_exp numpy_lapack_lite_r_exp
#define r_imag numpy_lapack_lite_r_imag
#define r_int numpy_lapack_lite_r_int
#define r_lg10 numpy_lapack_lite_r_lg10
#define r_log numpy_lapack_lite_r_log
#define r_mod numpy_lapack_lite_r_mod
#define r_nint numpy_lapack_lite_r_nint
#define r_sign numpy_lapack_lite_r_sign
#define r_sin numpy_lapack_lite_r_sin
#define r_sinh numpy_lapack_lite_r_sinh
#define r_sqrt numpy_lapack_lite_r_sqrt
#define r_tan numpy_lapack_lite_r_tan
#define r_tanh numpy_lapack_lite_r_tanh
#define s_cat numpy_lapack_lite_s_cat
#define s_cmp numpy_lapack_lite_s_cmp
#define s_copy numpy_lapack_lite_s_copy
#define s_paus numpy_lapack_lite_s_paus
#define s_rdfe numpy_lapack_lite_s_rdfe
#define s_rdue numpy_lapack_lite_s_rdue
#define s_rnge numpy_lapack_lite_s_rnge
#define s_rsfe numpy_lapack_lite_s_rsfe
#define s_rsfi numpy_lapack_lite_s_rsfi
#define s_rsle numpy_lapack_lite_s_rsle
#define s_rsli numpy_lapack_lite_s_rsli
#define s_rsne numpy_lapack_lite_s_rsne
#define s_rsni numpy_lapack_lite_s_rsni
#define s_rsue numpy_lapack_lite_s_rsue
#define s_stop numpy_lapack_lite_s_stop
#define s_wdfe numpy_lapack_lite_s_wdfe
#define s_wdue numpy_lapack_lite_s_wdue
#define s_wsfe numpy_lapack_lite_s_wsfe
#define s_wsfi numpy_lapack_lite_s_wsfi
#define s_wsle numpy_lapack_lite_s_wsle
#define s_wsli numpy_lapack_lite_s_wsli
#define s_wsne numpy_lapack_lite_s_wsne
#define s_wsni numpy_lapack_lite_s_wsni
#define s_wsue numpy_lapack_lite_s_wsue
#define sig_die numpy_lapack_lite_sig_die
#define signal_ numpy_lapack_lite_signal_
#define system_ numpy_lapack_lite_system_
#define z_abs numpy_lapack_lite_z_abs
#define z_cos numpy_lapack_lite_z_cos
# 定义五个宏，分别将其映射到 numpy_lapack_lite 库中的对应函数
#define z_div numpy_lapack_lite_z_div
#define z_exp numpy_lapack_lite_z_exp
#define z_log numpy_lapack_lite_z_log
#define z_sin numpy_lapack_lite_z_sin
#define z_sqrt numpy_lapack_lite_z_sqrt
```