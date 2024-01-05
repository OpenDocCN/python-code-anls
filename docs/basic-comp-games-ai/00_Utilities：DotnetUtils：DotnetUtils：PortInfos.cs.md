# `d:/src/tocomm/basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\PortInfos.cs`

```
# 获取程序入口的程序集，获取其父目录的完整路径
Root = GetParent(Assembly.GetEntryAssembly()!.Location)!.FullName
# 截取路径，去掉"\00_Utilities"部分
Root = Root[..Root.IndexOf(@"\00_Utilities")]

# 获取Root目录下的所有子目录
Get = GetDirectories(Root)
    # 遍历每个子目录，以及LangData字典的关键字
    .SelectMany(gamePath => LangData.Keys.Select(keyword => (gamePath, keyword)))
    # 调用PortInfo.Create方法创建PortInfo对象
    .SelectT((gamePath, keyword) => PortInfo.Create(gamePath, keyword))
    # 过滤掉空对象
    .Where(x => x is not null)
    # 转换为数组
    .ToArray()!
# 定义一个公共的静态只读的属性 Get，类型为 PortInfo 数组
public static readonly PortInfo[] Get;
```