# `d:/src/tocomm/basic-computer-games\90_Tower\csharp\Resources\Strings.cs`

```
// 引入命名空间
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明内部静态类 Strings
namespace Tower.Resources
{
    internal static class Strings
    {
        // 声明内部静态属性 Congratulations，返回资源字符串
        internal static string Congratulations => GetResource();
        // 声明内部静态属性 DiskCountPrompt，返回资源字符串
        internal static string DiskCountPrompt => GetResource();
        // 声明内部静态属性 DiskCountQuit，返回资源字符串
        internal static string DiskCountQuit => GetResource();
        // 声明内部静态属性 DiskCountRetry，返回资源字符串
        internal static string DiskCountRetry => GetResource();
        // 声明内部静态属性 DiskNotInPlay，返回资源字符串
        internal static string DiskNotInPlay => GetResource();
        // 声明内部静态属性 DiskPrompt，返回资源字符串
        internal static string DiskPrompt => GetResource();
        // 声明内部静态属性 DiskQuit，返回资源字符串
        internal static string DiskQuit => GetResource();
        // 声明内部静态属性 DiskRetry，返回资源字符串
        internal static string DiskRetry => GetResource();
        // 声明内部静态属性 DiskUnavailable，返回资源字符串
        internal static string DiskUnavailable => GetResource();
        // 声明内部静态属性 IllegalMove，返回资源字符串
        internal static string IllegalMove => GetResource();
        // 声明内部静态属性 Instructions，返回资源字符串
        internal static string Instructions => GetResource();
        // 声明内部静态属性 Intro，返回资源字符串
        internal static string Intro => GetResource();
```
```csharp
        // 获取资源字符串的方法
        private static string GetResource([CallerMemberName] string name = "")
        {
            // 获取当前程序集
            Assembly assembly = Assembly.GetExecutingAssembly();
            // 构建资源文件名
            string resourceName = "Tower.Resources.Strings." + name + ".txt";
            // 从程序集中获取资源流
            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            {
                // 读取资源流中的内容
                using (StreamReader reader = new StreamReader(stream))
                {
                    return reader.ReadToEnd();
                }
            }
        }
    }
}
        // 定义静态属性 NeedlePrompt，获取资源文件中的内容
        internal static string NeedlePrompt => GetResource();
        // 定义静态属性 NeedleQuit，获取资源文件中的内容
        internal static string NeedleQuit => GetResource();
        // 定义静态属性 NeedleRetry，获取资源文件中的内容
        internal static string NeedleRetry => GetResource();
        // 定义静态属性 PlayAgainPrompt，获取资源文件中的内容
        internal static string PlayAgainPrompt => GetResource();
        // 定义静态属性 TaskFinished，获取资源文件中的内容
        internal static string TaskFinished => GetResource();
        // 定义静态属性 Thanks，获取资源文件中的内容
        internal static string Thanks => GetResource();
        // 定义静态属性 Title，获取资源文件中的内容
        internal static string Title => GetResource();
        // 定义静态属性 TooManyMoves，获取资源文件中的内容
        internal static string TooManyMoves => GetResource();
        // 定义静态属性 YesNoPrompt，获取资源文件中的内容
        internal static string YesNoPrompt => GetResource();

        // 定义私有方法 GetResource，用于获取资源文件中指定名称的内容
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
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```