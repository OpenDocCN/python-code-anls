# `basic-computer-games\53_King\csharp\Game.cs`

```

namespace King;

// 内部游戏类
internal class Game
{
    // 在任期内的年限
    const int TermOfOffice = 8;

    // 读写接口
    private readonly IReadWrite _io;
    // 随机数接口
    private readonly IRandom _random;

    // 游戏构造函数
    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    // 游戏开始
    public void Play()
    {
        // 输出游戏标题
        _io.Write(Title);

        // 设置统治
        var reign = SetUpReign();
        // 如果设置成功，则进行游戏循环
        if (reign != null)
        {
            while (reign.PlayYear());
        }

        // 输出空行
        _io.WriteLine();
        _io.WriteLine();
    }

    // 设置统治
    private Reign? SetUpReign()
    {
        // 读取用户输入并转换为大写
        var response = _io.ReadString(InstructionsPrompt).ToUpper();

        // 如果用户输入为"Again"，则尝试读取游戏数据，否则返回空
        if (response.Equals("Again", StringComparison.InvariantCultureIgnoreCase))
        {
            return _io.TryReadGameData(_random, out var reign) ? reign : null;
        }
        
        // 如果用户输入不以"N"开头，则输出任期说明
        if (!response.StartsWith("N", StringComparison.InvariantCultureIgnoreCase))
        {
            _io.Write(InstructionsText(TermOfOffice));
        }

        // 输出空行
        _io.WriteLine();
        return new Reign(_io, _random);
    }
}

```