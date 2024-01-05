# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Resources\Strings.cs`

```
# 使用 System.IO 命名空间中的类来操作文件和目录
# 使用 System.Reflection 命名空间中的类来获取有关程序集的信息
# 使用 System.Runtime.CompilerServices 命名空间中的类来控制编译器的行为

# 定义一个静态类 Strings，用于存储程序中使用的字符串资源
# 定义静态属性 CombatArea，返回资源文件中的字符串
# 定义静态属性 Congratulations，返回资源文件中的字符串
# 定义静态属性 CourtMartial，返回资源文件中的字符串
# 定义静态属性 Destroyed，返回资源文件中的字符串
# 定义静态属性 EndOfMission，返回资源文件中的字符串
# 定义静态属性 Enterprise，返回资源文件中的字符串
        // 获取 Instructions 资源的字符串
        internal static string Instructions => GetResource();

        // 获取 LowShields 资源的字符串
        internal static string LowShields => GetResource();

        // 获取 NoEnemyShips 资源的字符串
        internal static string NoEnemyShips => GetResource();

        // 获取 NoStarbase 资源的字符串
        internal static string NoStarbase => GetResource();

        // 获取 NowEntering 资源的字符串
        internal static string NowEntering => GetResource();

        // 获取 Orders 资源的字符串
        internal static string Orders => GetResource();

        // 获取 PermissionDenied 资源的字符串
        internal static string PermissionDenied => GetResource();

        // 获取 Protected 资源的字符串
        internal static string Protected => GetResource();

        // 获取 RegionNames 资源的字符串
        internal static string RegionNames => GetResource();

        // 获取 RelievedOfCommand 资源的字符串
        internal static string RelievedOfCommand => GetResource();
        // 定义一个内部静态属性，用于获取资源中的修复估计
        internal static string RepairEstimate => GetResource();

        // 定义一个内部静态属性，用于获取资源中的修复提示
        internal static string RepairPrompt => GetResource();

        // 定义一个内部静态属性，用于获取资源中的重播提示
        internal static string ReplayPrompt => GetResource();

        // 定义一个内部静态属性，用于获取资源中的护盾已关闭
        internal static string ShieldsDropped => GetResource();

        // 定义一个内部静态属性，用于获取资源中的护盾已设置
        internal static string ShieldsSet => GetResource();

        // 定义一个内部静态属性，用于获取资源中的短程传感器故障
        internal static string ShortRangeSensorsOut => GetResource();

        // 定义一个内部静态属性，用于获取资源中的开始文本
        internal static string StartText => GetResource();

        // 定义一个内部静态属性，用于获取资源中的受困
        internal static string Stranded => GetResource();

        // 定义一个内部静态属性，用于获取资源中的标题
        internal static string Title => GetResource();

        // 定义一个私有的静态方法，用于获取资源中的字符串
        private static string GetResource([CallerMemberName] string name = "")
        {
# 使用给定的文件名构建资源文件的路径
var streamName = $"SuperStarTrek.Resources.{name}.txt";
# 使用程序集获取执行中的资源文件流
using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(streamName);
# 使用流创建一个读取器
using var reader = new StreamReader(stream);
# 读取并返回读取器中的所有内容
return reader.ReadToEnd();
```