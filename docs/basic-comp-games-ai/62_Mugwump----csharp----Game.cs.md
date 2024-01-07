# `basic-computer-games\62_Mugwump\csharp\Game.cs`

```

// 使用 System.Reflection 命名空间
using System.Reflection;

// 定义 Mugwump 命名空间
namespace Mugwump;

// 定义 Game 类
internal class Game
{
    // 私有字段，用于输入输出
    private readonly TextIO _io;
    // 私有字段，用于生成随机数
    private readonly IRandom _random;

    // 构造函数，接受输入输出和随机数生成器
    internal Game(TextIO io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    // 游戏主循环，接受一个可选的再玩一次的函数
    internal void Play(Func<bool> playAgain = null)
    {
        // 显示游戏介绍
        DisplayIntro();

        // 当再玩一次的函数返回 true 时，继续游戏
        while (playAgain?.Invoke() ?? true)
        {
            // 开始游戏
            Play(new Grid(_io, _random));

            // 输出提示信息
            _io.WriteLine();
            _io.WriteLine("That was fun! Let's play again.......");
            _io.WriteLine("Four more mugwumps are now in hiding.");
        }
    }

    // 显示游戏介绍
    private void DisplayIntro()
    {
        // 使用 Assembly.GetExecutingAssembly() 获取当前程序集，然后获取资源文件流
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("Mugwump.Strings.Intro.txt");

        // 输出资源文件流内容
        _io.Write(stream);
    }

    // 开始游戏
    private void Play(Grid grid)
    {
        // 进行 10 轮游戏
        for (int turn = 1; turn <= 10; turn++)
        {
            // 读取玩家猜测
            var guess = _io.ReadGuess($"Turn no. {turn} -- what is your guess");

            // 检查玩家猜测是否正确
            if (grid.Check(guess))
            {
                // 输出猜中信息并结束游戏
                _io.WriteLine();
                _io.WriteLine($"You got them all in {turn} turns!");
                return;
            }
        }

        // 输出未猜中信息并显示隐藏位置
        _io.WriteLine();
        _io.WriteLine("Sorry, that's 10 tries.  Here is where they're hiding:");
        grid.Reveal();
    }
}

```