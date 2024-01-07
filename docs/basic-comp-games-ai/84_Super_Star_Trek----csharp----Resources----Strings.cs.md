# `basic-computer-games\84_Super_Star_Trek\csharp\Resources\Strings.cs`

```

// 引入所需的命名空间
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

// 定义一个静态类 Strings
namespace SuperStarTrek.Resources
{
    internal static class Strings
    {
        // 定义一个静态属性 CombatArea，返回资源文件中的内容
        internal static string CombatArea => GetResource();

        // 定义其他静态属性，每个都返回资源文件中的内容
        internal static string Congratulations => GetResource();
        internal static string CourtMartial => GetResource();
        // ... 其他属性

        // 定义一个私有方法 GetResource，用于获取资源文件中的内容
        private static string GetResource([CallerMemberName] string name = "")
        {
            // 根据属性名构建资源文件的路径
            var streamName = $"SuperStarTrek.Resources.{name}.txt";
            // 使用 Assembly.GetExecutingAssembly().GetManifestResourceStream 方法获取资源文件流
            using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(streamName);
            // 使用 StreamReader 读取资源文件流，并返回其中的内容
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }
    }
}

```