# `53_King\csharp\Game.cs`

```
namespace King;  // 命名空间声明

internal class Game  // 内部类声明
{
    const int TermOfOffice = 8;  // 声明常量

    private readonly IReadWrite _io;  // 声明私有只读字段
    private readonly IRandom _random;  // 声明私有只读字段

    public Game(IReadWrite io, IRandom random)  // 构造函数，接受 IReadWrite 和 IRandom 接口类型的参数
    {
        _io = io;  // 初始化私有字段
        _random = random;  // 初始化私有字段
    }

    public void Play()  // 公共方法声明
    {
        _io.Write(Title);  // 调用 _io 对象的 Write 方法，传入 Title 参数

        var reign = SetUpReign();  // 调用 SetUpReign 方法，将返回值赋给变量 reign
```
```csharp
        // 其他代码...
    }
}
        # 如果reign不为空
        if (reign != null)
        {
            # 当reign的PlayYear方法返回true时循环执行
            while (reign.PlayYear());
        }

        # 输出空行
        _io.WriteLine();
        # 输出空行
        _io.WriteLine();
    }

    # 设置Reign对象的方法
    private Reign? SetUpReign()
    {
        # 从用户输入中读取字符串并转换为大写
        var response = _io.ReadString(InstructionsPrompt).ToUpper();

        # 如果用户输入是"Again"，不区分大小写
        if (response.Equals("Again", StringComparison.InvariantCultureIgnoreCase))
        {
            # 尝试从随机数和输入输出对象中读取游戏数据，如果成功则返回reign，否则返回null
            return _io.TryReadGameData(_random, out var reign) ? reign : null;
        }
        
        # 如果用户输入不是以"N"开头，不区分大小写
        if (!response.StartsWith("N", StringComparison.InvariantCultureIgnoreCase))
        {
# 写入任期指令文本到 _io 流中
_io.Write(InstructionsText(TermOfOffice));

# 在 _io 流中写入换行符
_io.WriteLine();

# 返回一个新的 Reign 对象，传入 _io 流和 _random 对象
return new Reign(_io, _random);
```