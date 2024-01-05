# `00_Utilities\DotnetUtils\DotnetUtils\Extensions.cs`

```
// 命名空间 DotnetUtils
namespace DotnetUtils;
// 创建 Extensions 类
public static class Extensions {
    // 创建 SelectT 方法，用于将元组中的两个元素映射为另一种类型的元素
    public static IEnumerable<TResult> SelectT<T1, T2, TResult>(this IEnumerable<(T1, T2)> src, Func<T1, T2, TResult> selector) =>
        src.Select(x => selector(x.Item1, x.Item2));
    // 创建 SelectT 方法，用于将元组中的三个元素映射为另一种类型的元素
    public static IEnumerable<TResult> SelectT<T1, T2, T3, TResult>(this IEnumerable<(T1, T2, T3)> src, Func<T1, T2, T3, TResult> selector) =>
        src.Select(x => selector(x.Item1, x.Item2, x.Item3));
    // 创建 WithIndex 方法，用于给元组中的元素添加索引
    public static IEnumerable<(T1, T2, int)> WithIndex<T1, T2>(this IEnumerable<(T1, T2)> src) => src.Select((x, index) => (x.Item1, x.Item2, index));

    // 创建 None 方法，用于判断集合中的元素是否满足指定条件
    public static bool None<T>(this IEnumerable<T> src, Func<T, bool>? predicate = null) =>
        predicate is null ?
            !src.Any() :
            !src.Any(predicate);

    // 创建 IsNullOrWhitespace 方法，用于判断字符串是否为 null 或空白字符串
    public static bool IsNullOrWhitespace([NotNullWhen(false)] this string? s) => string.IsNullOrWhiteSpace(s);

    // 添加 NotNullIfNotNull 特性，用于指示当 path 不为 null 时返回值也不为 null
    [return: NotNullIfNotNull("path")]
```
# 定义一个静态方法，用于获取相对路径
public static string? RelativePath(this string? path, string? rootPath) {
    # 如果路径为空或者根路径为空，则直接返回路径
    if (
        path.IsNullOrWhitespace() ||
        rootPath.IsNullOrWhitespace()
    ) { return path; }

    # 如果路径以反斜杠结尾，则去除结尾的反斜杠
    path = path.TrimEnd('\\'); // remove trailing backslash, if present
    # 调用 GetRelativePath 方法获取相对路径
    return GetRelativePath(rootPath, path.TrimEnd('\\'));
}
```