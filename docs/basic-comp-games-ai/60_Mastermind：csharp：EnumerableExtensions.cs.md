# `d:/src/tocomm/basic-computer-games\60_Mastermind\csharp\EnumerableExtensions.cs`

```
        /// </param>
        /// <returns>
        /// An infinite sequence of integers starting from <paramref name="start"/>
        /// and cycling through the range [0, count).
        /// </returns>
        public static IEnumerable<int> Cycle(this int start, int count)
        {
            // 使用 LINQ 生成一个无限序列，循环返回指定范围内的整数值
            return Enumerable.Range(start, count).Cycle();
        }

        /// <summary>
        /// Cycles through the elements of the source sequence.
        /// </summary>
        /// <param name="source">
        /// The sequence to cycle through.
        /// </param>
        /// <returns>
        /// An infinite sequence of elements from the source sequence, cycling
        /// through its elements.
        /// </returns>
        public static IEnumerable<T> Cycle<T>(this IEnumerable<T> source)
        {
            // 使用 LINQ 生成一个无限序列，循环返回源序列中的元素
            while (true)
            {
                foreach (var element in source)
                {
                    yield return element;
                }
            }
        }
    }
}
        /// <param name="start">The starting value for the cycle</param>
        /// <param name="count">The number of values in the cycle</param>
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
        /// Finds the index of the first item in the given sequence that
        /// satisfies the given predicate.
        /// </summary>
# 定义一个扩展方法，用于在给定的序列中查找第一个满足条件的元素的索引
# <typeparam name="T">序列中元素的类型</typeparam>
# <param name="source">源序列</param>
# <param name="predicate">条件函数</param>
# <returns>源序列中第一个满足条件的元素的索引。如果没有这样的元素，则返回 null。</returns>
public static int? FindFirstIndex<T>(this IEnumerable<T> source, Func<T, bool> predicate) =>
    # 使用 Select 方法将序列中的元素和它们的索引映射为满足条件的元素的索引或者 null
    source.Select((element, index) => predicate(element) ? index : default(int?))
        # 使用 FirstOrDefault 方法找到第一个满足条件的元素的索引
        .FirstOrDefault(index => index.HasValue);

# 返回给定序列中第一个满足条件的元素
        /// <summary>
        /// 返回序列中满足给定条件的第一个元素；如果没有元素满足条件，则返回指定的默认值。
        /// </summary>
        /// <typeparam name="T">
        /// 序列中元素的类型。
        /// </typeparam>
        /// <param name="source">
        /// 源序列。
        /// </param>
        /// <param name="predicate">
        /// 检查每个元素的条件。
        /// </param>
        /// <param name="defaultValue">
        /// 如果没有元素满足条件，则返回的值。
        /// </param>
        /// <returns>
        /// 返回序列中满足给定条件的第一个元素，如果没有则返回指定的默认值。
        /// </returns>
        public static T FirstOrDefault<T>(this IEnumerable<T> source, Func<T, bool> predicate, T defaultValue)
        {
# 遍历源列表中的每个元素
foreach (var element in source)
    # 如果满足条件函数的条件，返回该元素
    if (predicate(element))
        return element;

# 如果没有满足条件的元素，返回默认值
return defaultValue;
```