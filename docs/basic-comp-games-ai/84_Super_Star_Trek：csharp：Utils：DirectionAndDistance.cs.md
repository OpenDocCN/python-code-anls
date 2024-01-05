# `84_Super_Star_Trek\csharp\Utils\DirectionAndDistance.cs`

```
using System;  // 导入 System 命名空间
using SuperStarTrek.Space;  // 导入 SuperStarTrek.Space 命名空间

namespace SuperStarTrek.Utils  // 声明 SuperStarTrek.Utils 命名空间
{
    internal class DirectionAndDistance  // 声明 DirectionAndDistance 类
    {
        private readonly float _fromX;  // 声明私有只读字段 _fromX，用于存储 X 坐标
        private readonly float _fromY;  // 声明私有只读字段 _fromY，用于存储 Y 坐标

        private DirectionAndDistance(float fromX, float fromY)  // 声明私有构造函数，用于初始化 _fromX 和 _fromY
        {
            _fromX = fromX;  // 将参数 fromX 赋值给 _fromX
            _fromY = fromY;  // 将参数 fromY 赋值给 _fromY
        }

        internal static DirectionAndDistance From(Coordinates coordinates) => From(coordinates.X, coordinates.Y);  // 声明静态方法 From，接受 Coordinates 对象作为参数，并调用另一个 From 方法

        internal static DirectionAndDistance From(float x, float y) => new DirectionAndDistance(x, y);  // 声明静态方法 From，接受两个 float 类型的参数，并返回一个新的 DirectionAndDistance 对象
        // 定义一个内部方法，用于计算当前坐标到目标坐标的方向和距离
        internal (float Direction, float Distance) To(Coordinates coordinates) => To(coordinates.X, coordinates.Y);

        // 定义一个内部方法，用于计算当前坐标到指定坐标的方向和距离
        internal (float Direction, float Distance) To(float x, float y)
        {
            // 计算目标坐标与当前坐标的横向和纵向距离
            var deltaX = x - _fromX;
            var deltaY = y - _fromY;

            // 返回目标方向和距离
            return (GetDirection(deltaX, deltaY), GetDistance(deltaX, deltaY));
        }

        // 以下是对原始代码的数学等价算法的注释
        // 算法的数学等价代码如下，其中X为deltaY，A为deltaX
        //     8220 X=X-A:A=C1-W1:IFX<0THEN8350
        //     8250 IFA<0THEN8410
        //     8260 IFX>0THEN8280
        //     8270 IFA=0THENC1=5:GOTO8290
        //     8280 C1=1
        //     8290 IFABS(A)<=ABS(X)THEN8330
        //     8310 PRINT"DIRECTION =";C1+(((ABS(A)-ABS(X))+ABS(A))/ABS(A)):GOTO8460
        //     8330 PRINT"DIRECTION =";C1+(ABS(A)/ABS(X)):GOTO8460
//     8350 IFA>0THENC1=3:GOTO8420
// 如果 A 大于 0，则设置 C1 为 3，并跳转到 8420 行
// If A is greater than 0, then set C1 to 3 and go to line 8420

//     8360 IFX<>0THENC1=5:GOTO8290
// 如果 X 不等于 0，则设置 C1 为 5，并跳转到 8290 行
// If X is not equal to 0, then set C1 to 5 and go to line 8290

//     8410 C1=7
// 设置 C1 为 7
// Set C1 to 7

//     8420 IFABS(A)>=ABS(X)THEN8450
// 如果 A 的绝对值大于等于 X 的绝对值，则跳转到 8450 行
// If the absolute value of A is greater than or equal to the absolute value of X, then go to line 8450

//     8430 PRINT"DIRECTION =";C1+(((ABS(X)-ABS(A))+ABS(X))/ABS(X)):GOTO8460
// 打印 "DIRECTION =" 和 C1 加上 (((X 的绝对值减去 A 的绝对值) 加上 X 的绝对值) 除以 X 的绝对值)，然后跳转到 8460 行
// Print "DIRECTION =" and C1 plus (((absolute value of X minus absolute value of A) plus absolute value of X) divided by absolute value of X), then go to line 8460

//     8450 PRINT"DIRECTION =";C1+(ABS(X)/ABS(A))
// 打印 "DIRECTION =" 和 C1 加上 (X 的绝对值除以 A 的绝对值)
// Print "DIRECTION =" and C1 plus (absolute value of X divided by absolute value of A)

//     8460 PRINT"DISTANCE =";SQR(X^2+A^2):IFH8=1THEN1990
// 打印 "DISTANCE =" 和 (X 的平方加上 A 的平方) 的平方根；如果 H8 等于 1，则跳转到 1990 行
// Print "DISTANCE =" and the square root of (X squared plus A squared); if H8 equals 1, then go to line 1990
# 定义一个私有的静态方法，用于计算两点之间的距离
# 参数 deltaX: 两点在 x 轴上的距离
# 参数 deltaY: 两点在 y 轴上的距离
# 返回值: 两点之间的距离
private static float GetDistance(float deltaX, float deltaY) =>
    (float)Math.Sqrt(Math.Pow(deltaX, 2) + Math.Pow(deltaY, 2));
# 结束类定义
}
# 结束命名空间定义
}
```