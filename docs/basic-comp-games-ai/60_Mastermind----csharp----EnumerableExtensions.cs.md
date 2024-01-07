# `basic-computer-games\60_Mastermind\csharp\EnumerableExtensions.cs`

```

// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;

// 创建静态类 EnumerableExtensions
namespace Game
{
    /// <summary>
    /// 为 IEnumerable{T} 接口提供额外的方法
    /// </summary>
    public static class EnumerableExtensions
    {
        /// <summary>
        /// 循环遍历范围在 [0, count) 内的整数值
        /// </summary>
        /// <param name="start">
        /// 要返回的第一个值
        /// </param>
        /// <param name="count">
        /// 要返回的值的数量
        /// </param>
        public static IEnumerable<int> Cycle(int start, int count)
        {
            if (count < 1)
                throw new ArgumentException("count must be at least 1");

            if (start < 0 || start >= count)
                throw new ArgumentException("start must be in the range [0, count)");

            for (var i = start; i < count; ++i)
                yield return i;

            for (var i = 0; i < start; ++i)
                yield return i;
        }

        /// <summary>
        /// 查找给定序列中满足给定条件的第一个元素的索引
        /// </summary>
        /// <typeparam name="T">
        /// 序列中元素的类型
        /// </typeparam>
        /// <param name="source">
        /// 源序列
        /// </param>
        /// <param name="predicate">
        /// 谓词函数
        /// </param>
        /// <returns>
        /// 源序列中满足条件 predicate(element) 的第一个元素的索引。如果没有这样的元素，则返回 null。
        /// </returns>
        public static int? FindFirstIndex<T>(this IEnumerable<T> source, Func<T, bool> predicate) =>
            source.Select((element, index) => predicate(element) ? index : default(int?))
                .FirstOrDefault(index => index.HasValue);

        /// <summary>
        /// 返回给定序列中与给定条件匹配的第一个元素
        /// </summary>
        /// <typeparam name="T">
        /// 序列中元素的类型
        /// </typeparam>
        /// <param name="source">
        /// 源序列
        /// </param>
        /// <param name="predicate">
        /// 检查每个元素的谓词
        /// </param>
        /// <param name="defaultValue">
        /// 如果没有元素与谓词匹配，则返回的值
        /// </param>
        /// <returns>
        /// 源序列中与给定谓词匹配的第一个元素，如果没有则返回提供的默认值
        /// </returns>
        public static T FirstOrDefault<T>(this IEnumerable<T> source, Func<T, bool> predicate, T defaultValue)
        {
            foreach (var element in source)
                if (predicate(element))
                    return element;

            return defaultValue;
        }
    }
}

```