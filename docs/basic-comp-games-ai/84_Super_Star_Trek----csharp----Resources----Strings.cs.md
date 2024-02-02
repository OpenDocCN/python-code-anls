# `basic-computer-games\84_Super_Star_Trek\csharp\Resources\Strings.cs`

```py
# 引入命名空间，用于处理文件和目录路径
using System.IO;
# 引入命名空间，用于获取有关程序集的信息
using System.Reflection;
# 引入命名空间，用于指示编译器在生成的代码中是否保留成员的名称
using System.Runtime.CompilerServices;

# 定义内部静态类，用于存储资源字符串
namespace SuperStarTrek.Resources
{
    internal static class Strings
    {
        // 获取内部资源 CombatArea 的字符串值
        internal static string CombatArea => GetResource();
    
        // 获取内部资源 Congratulations 的字符串值
        internal static string Congratulations => GetResource();
    
        // 获取内部资源 CourtMartial 的字符串值
        internal static string CourtMartial => GetResource();
    
        // 获取内部资源 Destroyed 的字符串值
        internal static string Destroyed => GetResource();
    
        // 获取内部资源 EndOfMission 的字符串值
        internal static string EndOfMission => GetResource();
    
        // 获取内部资源 Enterprise 的字符串值
        internal static string Enterprise => GetResource();
    
        // 获取内部资源 Instructions 的字符串值
        internal static string Instructions => GetResource();
    
        // 获取内部资源 LowShields 的字符串值
        internal static string LowShields => GetResource();
    
        // 获取内部资源 NoEnemyShips 的字符串值
        internal static string NoEnemyShips => GetResource();
    
        // 获取内部资源 NoStarbase 的字符串值
        internal static string NoStarbase => GetResource();
    
        // 获取内部资源 NowEntering 的字符串值
        internal static string NowEntering => GetResource();
    
        // 获取内部资源 Orders 的字符串值
        internal static string Orders => GetResource();
    
        // 获取内部资源 PermissionDenied 的字符串值
        internal static string PermissionDenied => GetResource();
    
        // 获取内部资源 Protected 的字符串值
        internal static string Protected => GetResource();
    
        // 获取内部资源 RegionNames 的字符串值
        internal static string RegionNames => GetResource();
    
        // 获取内部资源 RelievedOfCommand 的字符串值
        internal static string RelievedOfCommand => GetResource();
    
        // 获取内部资源 RepairEstimate 的字符串值
        internal static string RepairEstimate => GetResource();
    
        // 获取内部资源 RepairPrompt 的字符串值
        internal static string RepairPrompt => GetResource();
    
        // 获取内部资源 ReplayPrompt 的字符串值
        internal static string ReplayPrompt => GetResource();
    
        // 获取内部资源 ShieldsDropped 的字符串值
        internal static string ShieldsDropped => GetResource();
    
        // 获取内部资源 ShieldsSet 的字符串值
        internal static string ShieldsSet => GetResource();
    
        // 获取内部资源 ShortRangeSensorsOut 的字符串值
        internal static string ShortRangeSensorsOut => GetResource();
    
        // 获取内部资源 StartText 的字符串值
        internal static string StartText => GetResource();
    
        // 获取内部资源 Stranded 的字符串值
        internal static string Stranded => GetResource();
    
        // 获取内部资源 Title 的字符串值
        internal static string Title => GetResource();
    
        // 从资源文件中获取字符串值
        private static string GetResource([CallerMemberName] string name = "")
        {
            // 构建资源文件的路径
            var streamName = $"SuperStarTrek.Resources.{name}.txt";
            // 获取当前程序集的嵌入资源流
            using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(streamName);
            // 使用流创建读取器
            using var reader = new StreamReader(stream);
    
            // 返回读取器中的所有内容作为字符串值
            return reader.ReadToEnd();
        }
    }
# 闭合前面的函数定义
```