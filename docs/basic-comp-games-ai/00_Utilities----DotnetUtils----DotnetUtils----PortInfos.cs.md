# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\PortInfos.cs`

```py
// 声明命名空间 DotnetUtils
namespace DotnetUtils
{
    // 声明公共静态类 PortInfos
    public static class PortInfos
    {
        // 声明只读的根目录路径
        public static readonly string Root;

        // 静态构造函数
        static PortInfos()
        {
            // 获取程序集的父目录的完整路径并赋值给 Root
            Root = GetParent(Assembly.GetEntryAssembly()!.Location)!.FullName;
            // 截取 Root 目录路径，去掉 \00_Utilities 部分
            Root = Root[..Root.IndexOf(@"\00_Utilities")];

            // 获取所有子目录的路径，并根据 LangData 的关键字创建 PortInfo 对象数组
            Get = GetDirectories(Root)
                .SelectMany(gamePath => LangData.Keys.Select(keyword => (gamePath, keyword)))
                .SelectT((gamePath, keyword) => PortInfo.Create(gamePath, keyword))
                .Where(x => x is not null)
                .ToArray()!;
        }

        // 声明只读的 PortInfo 数组
        public static readonly PortInfo[] Get;
    }
}
```