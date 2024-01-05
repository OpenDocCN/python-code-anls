# `d:/src/tocomm/basic-computer-games\83_Stock_Market\csharp\Extensions\EnumerableExtensions.cs`

```
        /// The type of the result elements.
        /// </typeparam>
        /// <param name="source">
        /// The sequence to apply the projection to.
        /// </param>
        /// <param name="func">
        /// A function that projects each element of the sequence.
        /// </param>
        /// <returns>
        /// An <see cref="IEnumerable{T}"/> that contains the result of
        /// applying the projection to each element of the source sequence.
        /// </returns>
        public static IEnumerable<TResult> SelectMany<TSource, TResult>(
            this IEnumerable<TSource> source, Func<TSource, IEnumerable<TResult>> func)
        {
            // Create a new list to store the results
            List<TResult> results = new List<TResult>();
            
            // Iterate through each element in the source sequence
            foreach (var item in source)
            {
                // Apply the projection function to the element and add the result to the list
                results.AddRange(func(item));
            }
            
            // Return the list of results
            return results;
        }
    }
}
# The type of elements in the result sequence.
# </typeparam>
# <param name="source">
# The source sequence.
# </param>
# <param name="seed">
# The seed value for the aggregation component.  This value is
# passed to the first call to <paramref name="selector"/>.
# </param>
# <param name="selector">
# The projection function.  This function is supplied with a value
# from the source sequence and the result of the projection on the
# previous value in the source sequence.
# </param>
# <returns>
# The resulting sequence.
# </returns>
# This method extends the functionality of IEnumerable by adding a SelectAndAggregate method.
# It takes a source sequence, a seed value, and a projection function as input parameters.
# It returns a resulting sequence of type TResult.
# The projection function is used to perform an aggregation operation on the source sequence, using the seed value as the initial value.
# The result of the aggregation is then projected into the resulting sequence.
# The method is generic, allowing it to work with different types of sequences and result types.
        Func<TSource, TResult, TResult> selector)
        {
            // 遍历源序列中的每个元素
            foreach (var element in source)
            {
                // 使用选择器函数将当前元素和累加器进行组合
                seed = selector(element, seed);
                // 返回组合后的结果
                yield return seed;
            }
        }

        /// <summary>
        /// 将三个不同序列的结果合并成一个单一序列。
        /// </summary>
        /// <typeparam name="T1">
        /// 第一个序列的元素类型。
        /// </typeparam>
        /// <typeparam name="T2">
        /// 第二个序列的元素类型。
        /// </typeparam>
        /// <typeparam name="T3">
        /// The element type of the third sequence.
        /// </typeparam>
        /// <typeparam name="TResult">
        /// The element type of the resulting sequence.
        /// </typeparam>
        /// <param name="first">
        /// The first source sequence.
        /// </param>
        /// <param name="second">
        /// The second source sequence.
        /// </param>
        /// <param name="third">
        /// The third source sequence.
        /// </param>
        /// <param name="resultSelector">
        /// Function that combines results from each source sequence into a
        /// final result.
        /// </param>
        /// <returns>
        /// A sequence of combined values.
```

这段代码是一个函数的注释部分，用于说明函数的参数和返回值的含义。其中包括了泛型参数的说明，以及每个参数的作用和类型。这些注释可以帮助其他程序员理解函数的用法和功能。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
# 定义一个静态方法，用于将三个 IEnumerable 对象进行 ZIP 操作，返回一个新的 IEnumerable 对象
public static IEnumerable<TResult> Zip<T1, T2, T3, TResult>(
    this IEnumerable<T1> first,  # 第一个输入的 IEnumerable 对象
    IEnumerable<T2> second,      # 第二个输入的 IEnumerable 对象
    IEnumerable<T3> third,       # 第三个输入的 IEnumerable 对象
    Func<T1, T2, T3, TResult> resultSelector)  # 一个函数，用于将三个输入对象的元素组合成一个新的结果对象
{
    using var enumerator1 = first.GetEnumerator();  # 获取第一个输入对象的枚举器
    using var enumerator2 = second.GetEnumerator();  # 获取第二个输入对象的枚举器
    using var enumerator3 = third.GetEnumerator();  # 获取第三个输入对象的枚举器

    while (enumerator1.MoveNext() && enumerator2.MoveNext() && enumerator3.MoveNext())  # 遍历三个输入对象的元素
        yield return resultSelector(enumerator1.Current, enumerator2.Current, enumerator3.Current);  # 使用 resultSelector 函数将三个输入对象的当前元素组合成一个新的结果对象，并返回
}
```