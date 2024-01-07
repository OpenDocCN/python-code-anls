# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\Extensions.cs`

```

// 命名空间 DotnetUtils
namespace DotnetUtils;

// 创建一个静态类 Extensions
public static class Extensions {
    // 创建一个扩展方法，用于将元组类型的集合转换为另一种类型的集合
    public static IEnumerable<TResult> SelectT<T1, T2, TResult>(this IEnumerable<(T1, T2)> src, Func<T1, T2, TResult> selector) =>
        src.Select(x => selector(x.Item1, x.Item2));
    // 创建一个扩展方法，用于将三元组类型的集合转换为另一种类型的集合
    public static IEnumerable<TResult> SelectT<T1, T2, T3, TResult>(this IEnumerable<(T1, T2, T3)> src, Func<T1, T2, T3, TResult> selector) =>
        src.Select(x => selector(x.Item1, x.Item2, x.Item3));
    // 创建一个扩展方法，用于给元组类型的集合添加索引
    public static IEnumerable<(T1, T2, int)> WithIndex<T1, T2>(this IEnumerable<(T1, T2)> src) => src.Select((x, index) => (x.Item1, x.Item2, index));

    // 创建一个扩展方法，用于判断集合中的元素是否全部不满足给定条件
    public static bool None<T>(this IEnumerable<T> src, Func<T, bool>? predicate = null) =>
        predicate is null ?
            !src.Any() :
            !src.Any(predicate);

    // 创建一个扩展方法，用于判断字符串是否为 null 或空白
    public static bool IsNullOrWhitespace([NotNullWhen(false)] this string? s) => string.IsNullOrWhiteSpace(s);

    // 创建一个扩展方法，用于计算相对路径
    [return: NotNullIfNotNull("path")]
    public static string? RelativePath(this string? path, string? rootPath) {
        // 如果路径为空或者根路径为空，则返回路径本身
        if (
            path.IsNullOrWhitespace() ||
            rootPath.IsNullOrWhitespace()
        ) { return path; }

        // 去除路径末尾的反斜杠
        path = path.TrimEnd('\\'); // remove trailing backslash, if present
        // 返回相对路径
        return GetRelativePath(rootPath, path.TrimEnd('\\'));
    }
}

```