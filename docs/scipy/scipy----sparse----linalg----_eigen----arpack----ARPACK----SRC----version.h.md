# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\arpack\ARPACK\SRC\version.h`

```
/*
   在当前版本中，Kahan的正交性检验中的参数KAPPA设定为0.717，与Gragg和Reichel使用的相同。
   然而计算经验表明，这可能过于严格，会经常强制重新正交化，而实际上可能并不需要这样做。

   此外，在非对称代码中，“移动边界”思想目前未激活，因为尚未确定是否始终正确。需要进一步研究。

   自1993年2月1日起，Richard Lehoucq接管了代码的软件控制权，取代了Phuong Vu。
   在1993年3月1日，所有的*.F文件迁移到了SCCS。版本1.1的代码是从Phuong Vu那里接收到的。
   1992年7月8日的冻结版本现在被认为是版本1.1。

   版本2.1包含两个新的对称例程，sesrt和seupd。
   对版本1.1代码进行的更改以及bug修复，形成了版本1.2。这些1.2版本也将包含在版本2.1中。
   子程序[d,s]saupd现在需要略多一些的工作空间。详细信息请参见[d,s]saupd。

   SCCS信息: @(#) 
   文件: version.h   版本标识符: 2.3   版本标识日期: 11/16/95   发布版本: 2
*/

// 定义版本号常量
#define VERSION_NUMBER ' 2.1'
// 定义版本日期常量
#define VERSION_DATE   ' 11/15/95'
```