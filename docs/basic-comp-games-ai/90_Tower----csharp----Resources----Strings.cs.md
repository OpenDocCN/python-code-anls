# `basic-computer-games\90_Tower\csharp\Resources\Strings.cs`

```

// 引入所需的命名空间
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

// 定义一个静态类 Strings
namespace Tower.Resources
{
    internal static class Strings
    {
        // 定义静态属性 Congratulations，返回资源文件中的内容
        internal static string Congratulations => GetResource();
        // 定义静态属性 DiskCountPrompt，返回资源文件中的内容
        internal static string DiskCountPrompt => GetResource();
        // 定义静态属性 DiskCountQuit，返回资源文件中的内容
        internal static string DiskCountQuit => GetResource();
        // 定义静态属性 DiskCountRetry，返回资源文件中的内容
        internal static string DiskCountRetry => GetResource();
        // 定义静态属性 DiskNotInPlay，返回资源文件中的内容
        internal static string DiskNotInPlay => GetResource();
        // 定义静态属性 DiskPrompt，返回资源文件中的内容
        internal static string DiskPrompt => GetResource();
        // 定义静态属性 DiskQuit，返回资源文件中的内容
        internal static string DiskQuit => GetResource();
        // 定义静态属性 DiskRetry，返回资源文件中的内容
        internal static string DiskRetry => GetResource();
        // 定义静态属性 DiskUnavailable，返回资源文件中的内容
        internal static string DiskUnavailable => GetResource();
        // 定义静态属性 IllegalMove，返回资源文件中的内容
        internal static string IllegalMove => GetResource();
        // 定义静态属性 Instructions，返回资源文件中的内容
        internal static string Instructions => GetResource();
        // 定义静态属性 Intro，返回资源文件中的内容
        internal static string Intro => GetResource();
        // 定义静态属性 NeedlePrompt，返回资源文件中的内容
        internal static string NeedlePrompt => GetResource();
        // 定义静态属性 NeedleQuit，返回资源文件中的内容
        internal static string NeedleQuit => GetResource();
        // 定义静态属性 NeedleRetry，返回资源文件中的内容
        internal static string NeedleRetry => GetResource();
        // 定义静态属性 PlayAgainPrompt，返回资源文件中的内容
        internal static string PlayAgainPrompt => GetResource();
        // 定义静态属性 TaskFinished，返回资源文件中的内容
        internal static string TaskFinished => GetResource();
        // 定义静态属性 Thanks，返回资源文件中的内容
        internal static string Thanks => GetResource();
        // 定义静态属性 Title，返回资源文件中的内容
        internal static string Title => GetResource();
        // 定义静态属性 TooManyMoves，返回资源文件中的内容
        internal static string TooManyMoves => GetResource();
        // 定义静态属性 YesNoPrompt，返回资源文件中的内容
        internal static string YesNoPrompt => GetResource();

        // 定义一个私有的静态方法 GetResource，用于获取资源文件中的内容
        private static string GetResource([CallerMemberName] string name = "")
        {
            // 构建资源文件的路径
            var streamName = $"Tower.Resources.{name}.txt";
            // 使用 Assembly.GetExecutingAssembly().GetManifestResourceStream 方法获取资源文件的流
            using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(streamName);
            // 使用 StreamReader 读取资源文件的内容
            using var reader = new StreamReader(stream);

            // 返回资源文件的内容
            return reader.ReadToEnd();
        }
    }
}

```