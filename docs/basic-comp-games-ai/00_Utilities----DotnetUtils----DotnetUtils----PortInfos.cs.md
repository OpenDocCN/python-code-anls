# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\PortInfos.cs`

```

// 引入命名空间和静态类
using System.Reflection;
using static System.IO.Directory;
using static DotnetUtils.Globals;

namespace DotnetUtils;

// 定义端口信息类
public static class PortInfos {
    // 定义根目录
    public static readonly string Root;

    // 静态构造函数
    static PortInfos() {
        // 获取程序集的父目录的全名作为根目录
        Root = GetParent(Assembly.GetEntryAssembly()!.Location)!.FullName;
        // 截取根目录直到"\00_Utilities"之前的部分
        Root = Root[..Root.IndexOf(@"\00_Utilities")];

        // 获取根目录下的子目录，根据语言数据的关键字创建端口信息对象
        Get = GetDirectories(Root)
            .SelectMany(gamePath => LangData.Keys.Select(keyword => (gamePath, keyword)))
            .SelectT((gamePath, keyword) => PortInfo.Create(gamePath, keyword))
            .Where(x => x is not null)
            .ToArray()!;
    }

    // 获取端口信息数组
    public static readonly PortInfo[] Get;
}

```