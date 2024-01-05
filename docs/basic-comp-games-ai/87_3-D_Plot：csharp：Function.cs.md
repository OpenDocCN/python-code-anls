# `d:/src/tocomm/basic-computer-games\87_3-D_Plot\csharp\Function.cs`

```
using System;  // 导入 System 命名空间，包含常用的基本类和数据类型
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间，包含泛型集合类

namespace Plot  // 命名空间 Plot
{
    internal static class Function  // 内部静态类 Function
    {
        internal static IEnumerable<IEnumerable<int>> GetRows()  // 返回一个 IEnumerable 类型的嵌套集合
        {
            for (var x = -30f; x <= 30f; x += 1.5f)  // 循环，从 -30 到 30，每次增加 1.5
            {
                yield return GetValues(x);  // 返回 GetValues 方法的结果
            }
        }

        private static IEnumerable<int> GetValues(float x)  // 返回一个 IEnumerable 类型的整数集合，参数为浮点数 x
        {
            var zPrevious = 0;  // 声明并初始化变量 zPrevious 为 0
            var yLimit = 5 * (int)(Math.Sqrt(900 - x * x) / 5);  // 声明并初始化变量 yLimit 为一个计算结果
# 循环，从yLimit开始递减到-yLimit，每次递减5
for (var y = yLimit; y >= -yLimit; y -= 5)
{
    # 调用GetValue函数，传入x和y的值，将结果赋给z
    var z = GetValue(x, y);

    # 如果z大于zPrevious
    if (z > zPrevious)
    {
        # 将z赋给zPrevious
        zPrevious = z;
        # 返回z的值
        yield return z;
    }
}

# 定义一个名为GetValue的私有静态函数，接受两个float类型的参数x和y
private static int GetValue(float x, float y)
{
    # 计算r的值，使用Math.Sqrt计算x的平方加上y的平方的平方根
    var r = (float)Math.Sqrt(x * x + y * y);
    # 返回一个整数值，计算公式为25 + 30 * e^(-r^2 / 100) - 0.7 * y
    return (int)(25 + 30 * Math.Exp(-r * r / 100) - 0.7f * y);
}
```