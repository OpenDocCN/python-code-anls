# `basic-computer-games\83_Stock_Market\csharp\Extensions\RandomExtensions.cs`

```py
// 命名空间 Game.Extensions 包含了 RandomExtensions 类
namespace Game.Extensions
{
    /// <summary>
    /// 为 Random 类提供额外的方法
    /// </summary>
    public static class RandomExtensions
    {
        /// <summary>
        /// 生成一个无限序列的随机数
        /// </summary>
        /// <param name="random">
        /// 随机数生成器
        /// </param>
        /// <param name="min">
        /// 要生成的范围的包含下限
        /// </param>
        /// <param name="max">
        /// 要生成的范围的排除上限
        /// </param>
        /// <returns>
        /// 在范围 [min, max) 内的无限随机整数序列
        /// </returns>
        /// <remarks>
        /// <para>
        /// 我们使用排除上限，尽管有点令人困惑，为了与 Random.Next 保持一致性。
        /// </para>
        /// <para>
        /// 由于序列是无限的，典型的用法是使用类似 Enumerable.Take 的函数来限制结果。例如，
        /// 要对三个六面骰子的结果求和，我们可以这样做：
        /// </para>
        /// <code>
        /// random.Integers(1, 7).Take(3).Sum()
        /// </code>
        /// </remarks>
        public static IEnumerable<int> Integers(this Random random, int min, int max)
        {
            // 无限循环，不断返回随机数
            while (true)
                yield return random.Next(min, max);
        }
    }
}
```