# `basic-computer-games\83_Stock_Market\csharp\Extensions\ImmutableArrayExtensions.cs`

```

// 命名空间 Game.Extensions 包含了 ImmutableArrayExtensions 类
namespace Game.Extensions
{
    /// <summary>
    /// 为 ImmutableArray<T> 类提供额外的方法
    /// </summary>
    public static class ImmutableArrayExtensions
    {
        /// <summary>
        /// 将不可变数组中的每个元素映射到一个新值
        /// </summary>
        /// <typeparam name="TSource">
        /// 源数组中元素的类型
        /// </typeparam>
        /// <typeparam name="TResult">
        /// 结果数组中元素的类型
        /// </typeparam>
        /// <param name="source">
        /// 源数组
        /// </param>
        /// <param name="selector">
        /// 接收源数组中的元素和其索引，并返回结果元素的函数
        /// </param>
        public static ImmutableArray<TResult> Map<TSource, TResult>(this ImmutableArray<TSource> source, Func<TSource, int, TResult> selector)
        {
            // 创建一个结果数组的构建器
            var builder = ImmutableArray.CreateBuilder<TResult>(source.Length);

            // 遍历源数组，将经过 selector 函数处理后的元素添加到结果数组中
            for (var i = 0; i < source.Length; ++i)
                builder.Add(selector(source[i], i));

            // 将构建器中的内容转换为不可变数组并返回
            return builder.MoveToImmutable();
        }
    }
}

```