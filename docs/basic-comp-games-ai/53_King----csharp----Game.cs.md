# `basic-computer-games\53_King\csharp\Game.cs`

```py
namespace King;

internal class Game
{
    // 定义国王任期的长度为8年
    const int TermOfOffice = 8;

    // 声明私有只读字段_io，用于输入输出操作
    private readonly IReadWrite _io;
    // 声明私有只读字段_random，用于生成随机数
    private readonly IRandom _random;

    // 构造函数，接受io和random作为参数
    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    // 游戏进行方法
    public void Play()
    {
        // 输出游戏标题
        _io.Write(Title);

        // 设置国王任期
        var reign = SetUpReign();
        // 如果设置成功，则进行游戏循环
        if (reign != null)
        {
            while (reign.PlayYear());
        }

        // 输出空行
        _io.WriteLine();
        // 输出空行
        _io.WriteLine();
    }

    // 设置国王任期方法
    private Reign? SetUpReign()
    {
        // 读取用户输入的响应，并转换为大写
        var response = _io.ReadString(InstructionsPrompt).ToUpper();

        // 如果用户输入为"Again"，则尝试读取游戏数据，如果成功则返回国王任期对象，否则返回null
        if (response.Equals("Again", StringComparison.InvariantCultureIgnoreCase))
        {
            return _io.TryReadGameData(_random, out var reign) ? reign : null;
        }
        
        // 如果用户输入不是以"N"开头，则输出指令文本
        if (!response.StartsWith("N", StringComparison.InvariantCultureIgnoreCase))
        {
            _io.Write(InstructionsText(TermOfOffice));
        }

        // 输出空行
        _io.WriteLine();
        // 返回新的国王任期对象
        return new Reign(_io, _random);
    }
}
```