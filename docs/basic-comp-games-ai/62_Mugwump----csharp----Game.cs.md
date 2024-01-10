# `basic-computer-games\62_Mugwump\csharp\Game.cs`

```
// 使用 System.Reflection 命名空间
using System.Reflection;

// 定义 Mugwump 命名空间
namespace Mugwump;

// 定义 Game 类
internal class Game
{
    // 声明私有的 TextIO 类型变量 _io
    private readonly TextIO _io;
    // 声明私有的 IRandom 类型变量 _random
    private readonly IRandom _random;

    // Game 类的构造函数，接受 TextIO 和 IRandom 类型的参数
    internal Game(TextIO io, IRandom random)
    {
        // 将传入的 io 赋值给 _io
        _io = io;
        // 将传入的 random 赋值给 _random
        _random = random;
    }

    // Play 方法，接受一个返回布尔值的委托 playAgain
    internal void Play(Func<bool> playAgain = null)
    {
        // 显示游戏介绍
        DisplayIntro();

        // 当 playAgain 不为空且返回值为 true 时循环执行以下代码块
        while (playAgain?.Invoke() ?? true)
        {
            // 调用 Play 方法，传入一个新的 Grid 对象
            Play(new Grid(_io, _random));

            // 输出空行
            _io.WriteLine();
            // 输出提示信息
            _io.WriteLine("That was fun! Let's play again.......");
            _io.WriteLine("Four more mugwumps are now in hiding.");
        }
    }

    // DisplayIntro 方法，用于显示游戏介绍
    private void DisplayIntro()
    {
        // 使用 Assembly.GetExecutingAssembly() 获取当前执行的程序集，再使用 GetManifestResourceStream 方法获取资源文件流
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("Mugwump.Strings.Intro.txt");

        // 将资源文件流输出到 _io
        _io.Write(stream);
    }

    // Play 方法，接受一个 Grid 类型的参数 grid
    private void Play(Grid grid)
    {
        // 循环执行 10 次
        for (int turn = 1; turn <= 10; turn++)
        {
            // 从 _io 读取玩家的猜测
            var guess = _io.ReadGuess($"Turn no. {turn} -- what is your guess");

            // 检查玩家的猜测是否正确
            if (grid.Check(guess))
            {
                // 输出空行
                _io.WriteLine();
                // 输出玩家猜中的信息
                _io.WriteLine($"You got them all in {turn} turns!");
                // 结束方法
                return;
            }
        }

        // 输出空行
        _io.WriteLine();
        // 输出未猜中的信息
        _io.WriteLine("Sorry, that's 10 tries.  Here is where they're hiding:");
        // 显示所有 mugwumps 的位置
        grid.Reveal();
    }
}
```