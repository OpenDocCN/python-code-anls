# `basic-computer-games\87_3-D_Plot\csharp\Function.cs`

```py
using System;
using System.Collections.Generic;

namespace Plot
{
    internal static class Function
    {
        // 返回一个包含多个整数序列的序列
        internal static IEnumerable<IEnumerable<int>> GetRows()
        {
            // 从-30到30，以1.5为步长，生成x的值
            for (var x = -30f; x <= 30f; x += 1.5f)
            {
                // 返回x对应的整数序列
                yield return GetValues(x);
            }
        }

        // 返回一个整数序列
        private static IEnumerable<int> GetValues(float x)
        {
            // 初始化zPrevious为0
            var zPrevious = 0;
            // 计算y的上限值
            var yLimit = 5 * (int)(Math.Sqrt(900 - x * x) / 5);

            // 从yLimit开始，以-5为步长，生成y的值
            for (var y = yLimit; y >= -yLimit; y -= 5)
            {
                // 计算(x, y)对应的z值
                var z = GetValue(x, y);

                // 如果z大于zPrevious，则更新zPrevious并返回z
                if (z > zPrevious)
                {
                    zPrevious = z;
                    yield return z;
                }
            }
        }

        // 返回一个整数值
        private static int GetValue(float x, float y)
        {
            // 计算点(x, y)到原点的距离
            var r = (float)Math.Sqrt(x * x + y * y);
            // 根据距离r计算z值
            return (int)(25 + 30 * Math.Exp(-r * r / 100) - 0.7f * y);
        }
    }
}
```