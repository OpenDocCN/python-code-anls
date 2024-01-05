# `83_Stock_Market\csharp\Extensions\ImmutableArrayExtensions.cs`

```
        /// The source immutable array.
        /// </param>
        /// <param name="selector">
        /// A function to apply to each element in the array.
        /// </param>
        /// <returns>
        /// An immutable array that contains the result of applying the function to each element in the source array.
        /// </returns>
        public static ImmutableArray<TResult> Map<TSource, TResult>(this ImmutableArray<TSource> source, Func<TSource, TResult> selector)
        {
            // Create a new array to store the mapped elements
            var builder = ImmutableArray.CreateBuilder<TResult>(source.Length);
            
            // Iterate through each element in the source array and apply the selector function
            foreach (var item in source)
            {
                // Add the result of the selector function to the new array
                builder.Add(selector(item));
            }
            
            // Return the new immutable array
            return builder.ToImmutable();
        }
    }
}
# 定义一个静态扩展方法，用于对不可变数组进行映射操作
public static ImmutableArray<TResult> Map<TSource, TResult>(this ImmutableArray<TSource> source, Func<TSource, int, TResult> selector)
{
    # 创建一个不可变数组构建器，用于构建结果数组
    var builder = ImmutableArray.CreateBuilder<TResult>(source.Length);

    # 遍历源数组，对每个元素调用传入的 selector 函数，并将结果添加到构建器中
    for (var i = 0; i < source.Length; ++i)
        builder.Add(selector(source[i], i));

    # 将构建器中的内容转换为不可变数组并返回
    return builder.MoveToImmutable();
}
```