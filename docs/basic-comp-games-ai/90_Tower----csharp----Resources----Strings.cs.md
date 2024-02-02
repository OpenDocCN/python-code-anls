# `basic-computer-games\90_Tower\csharp\Resources\Strings.cs`

```py
// 命名空间 Tower.Resources，包含了静态类 Strings
namespace Tower.Resources
{
    // 内部静态类 Strings
    internal static class Strings
    {
        // 内部静态属性 Congratulations，返回资源文件中的内容
        internal static string Congratulations => GetResource();
        // 内部静态属性 DiskCountPrompt，返回资源文件中的内容
        internal static string DiskCountPrompt => GetResource();
        // 内部静态属性 DiskCountQuit，返回资源文件中的内容
        internal static string DiskCountQuit => GetResource();
        // 内部静态属性 DiskCountRetry，返回资源文件中的内容
        internal static string DiskCountRetry => GetResource();
        // 内部静态属性 DiskNotInPlay，返回资源文件中的内容
        internal static string DiskNotInPlay => GetResource();
        // 内部静态属性 DiskPrompt，返回资源文件中的内容
        internal static string DiskPrompt => GetResource();
        // 内部静态属性 DiskQuit，返回资源文件中的内容
        internal static string DiskQuit => GetResource();
        // 内部静态属性 DiskRetry，返回资源文件中的内容
        internal static string DiskRetry => GetResource();
        // 内部静态属性 DiskUnavailable，返回资源文件中的内容
        internal static string DiskUnavailable => GetResource();
        // 内部静态属性 IllegalMove，返回资源文件中的内容
        internal static string IllegalMove => GetResource();
        // 内部静态属性 Instructions，返回资源文件中的内容
        internal static string Instructions => GetResource();
        // 内部静态属性 Intro，返回资源文件中的内容
        internal static string Intro => GetResource();
        // 内部静态属性 NeedlePrompt，返回资源文件中的内容
        internal static string NeedlePrompt => GetResource();
        // 内部静态属性 NeedleQuit，返回资源文件中的内容
        internal static string NeedleQuit => GetResource();
        // 内部静态属性 NeedleRetry，返回资源文件中的内容
        internal static string NeedleRetry => GetResource();
        // 内部静态属性 PlayAgainPrompt，返回资源文件中的内容
        internal static string PlayAgainPrompt => GetResource();
        // 内部静态属性 TaskFinished，返回资源文件中的内容
        internal static string TaskFinished => GetResource();
        // 内部静态属性 Thanks，返回资源文件中的内容
        internal static string Thanks => GetResource();
        // 内部静态属性 Title，返回资源文件中的内容
        internal static string Title => GetResource();
        // 内部静态属性 TooManyMoves，返回资源文件中的内容
        internal static string TooManyMoves => GetResource();
        // 内部静态属性 YesNoPrompt，返回资源文件中的内容
        internal static string YesNoPrompt => GetResource();

        // 私有静态方法 GetResource，根据调用者的名称获取资源文件中的内容
        private static string GetResource([CallerMemberName] string name = "")
        {
            // 构建资源文件的名称
            var streamName = $"Tower.Resources.{name}.txt";
            // 使用当前程序集获取资源文件的流
            using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(streamName);
            // 使用流创建读取器
            using var reader = new StreamReader(stream);

            // 返回读取器中的所有内容
            return reader.ReadToEnd();
        }
    }
}
```