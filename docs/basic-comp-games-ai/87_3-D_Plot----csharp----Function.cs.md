# `basic-computer-games\87_3-D_Plot\csharp\Function.cs`

```

// 引入系统和集合类库
using System;
using System.Collections.Generic;

// 定义名为 Plot 的命名空间
namespace Plot
{
    // 定义名为 Function 的静态类
    internal static class Function
    {
        // 返回一个包含整数集合的集合
        internal static IEnumerable<IEnumerable<int>> GetRows()
        {
            // 从 -30 到 30，以 1.5 为步长，生成 x 值，并返回对应的值集合
            for (var x = -30f; x <= 30f; x += 1.5f)
            {
                yield return GetValues(x);
            }
        }

        // 根据给定的 x 值，返回对应的整数集合
        private static IEnumerable<int> GetValues(float x)
        {
            // 初始化前一个 z 值为 0
            var zPrevious = 0;
            // 计算 y 的取值范围
            var yLimit = 5 * (int)(Math.Sqrt(900 - x * x) / 5);

            // 从 yLimit 开始，以 -5 为步长，生成 y 值，并返回对应的 z 值集合
            for (var y = yLimit; y >= -yLimit; y -= 5)
            {
                // 计算当前 x、y 对应的 z 值
                var z = GetValue(x, y);

                // 如果当前 z 值大于前一个 z 值，则更新前一个 z 值，并返回当前 z 值
                if (z > zPrevious)
                {
                    zPrevious = z;
                    yield return z;
                }
            }
        }

        // 根据给定的 x、y 值，计算并返回对应的整数值
        private static int GetValue(float x, float y)
        {
            // 计算 x、y 对应的极坐标半径
            var r = (float)Math.Sqrt(x * x + y * y);
            // 根据极坐标半径和 y 值计算 z 值
            return (int)(25 + 30 * Math.Exp(-r * r / 100) - 0.7f * y);
        }
    }
}

```