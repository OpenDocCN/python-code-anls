# `62_Mugwump\csharp\Game.cs`

```
        # 显示游戏介绍
        DisplayIntro()

        # 当playAgain函数存在且返回值为True时，继续游戏
        while (playAgain?.Invoke() ?? true)
        {
            // 调用 Play 方法，传入一个 Grid 对象和一个 Random 对象
            Play(new Grid(_io, _random));

            // 输出空行
            _io.WriteLine();
            // 输出提示信息
            _io.WriteLine("That was fun! Let's play again.......");
            // 输出提示信息
            _io.WriteLine("Four more mugwumps are now in hiding.");
        }
    }

    private void DisplayIntro()
    {
        // 使用 Assembly.GetExecutingAssembly().GetManifestResourceStream 方法获取资源文件 "Mugwump.Strings.Intro.txt" 的流
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("Mugwump.Strings.Intro.txt");

        // 将流内容输出到控制台
        _io.Write(stream);
    }

    private void Play(Grid grid)
    {
        // 循环进行游戏，共进行 10 次
        for (int turn = 1; turn <= 10; turn++)
        {
            var guess = _io.ReadGuess($"Turn no. {turn} -- what is your guess");  // 从输入流中读取玩家的猜测，并将其存储在变量 guess 中

            if (grid.Check(guess))  // 检查玩家的猜测是否正确
            {
                _io.WriteLine();  // 在输出流中打印空行
                _io.WriteLine($"You got them all in {turn} turns!");  // 在输出流中打印玩家猜中所有位置所用的轮次
                return;  // 结束当前函数的执行
            }
        }

        _io.WriteLine();  // 在输出流中打印空行
        _io.WriteLine("Sorry, that's 10 tries.  Here is where they're hiding:");  // 在输出流中打印消息，表示玩家已经用完了10次猜测机会
        grid.Reveal();  // 调用 grid 对象的 Reveal 方法，显示所有位置的真实值
    }
}
```