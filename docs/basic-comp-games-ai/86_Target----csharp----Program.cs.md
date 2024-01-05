# `86_Target\csharp\Program.cs`

```
# 导入所需的模块
import System
import import Games.Common.IO
import import Games.Common.Randomness

# 创建 Program 类
class Program:
    # 创建 Main 方法
    def Main():
        # 创建 ConsoleIO 实例
        io = ConsoleIO()
        # 创建 Game 实例，传入 ConsoleIO 实例和 RandomNumberGenerator 实例
        game = Game(io, FiringRange(RandomNumberGenerator()))

        # 调用 Play 方法，传入 Game 实例、ConsoleIO 实例和一个始终返回 True 的函数
        Play(game, io, lambda: True)

    # 创建 Play 方法，接受 Game 实例、TextIO 实例和一个返回布尔值的函数
    def Play(game, io, playAgain):
        # 调用 DisplayTitleAndInstructions 方法，传入 TextIO 实例
        DisplayTitleAndInstructions(io)
# 当playAgain()返回True时，执行循环体内的代码
while (playAgain())
{
    # 调用game对象的Play方法
    game.Play();

    # 输出空行
    io.WriteLine();
    io.WriteLine();
    io.WriteLine();
    io.WriteLine();
    io.WriteLine();
    # 输出"Next target..."
    io.WriteLine("Next target...");
    # 输出空行
    io.WriteLine();
}

# 定义一个私有方法DisplayTitleAndInstructions，接受一个TextIO对象作为参数
private static void DisplayTitleAndInstructions(TextIO io)
{
    # 使用Assembly.GetExecutingAssembly()获取当前执行的程序集
    # 调用GetManifestResourceStream方法获取嵌入资源"Target.Strings.TitleAndInstructions.txt"的流
    using var stream = Assembly.GetExecutingAssembly()
        .GetManifestResourceStream("Target.Strings.TitleAndInstructions.txt");
    # 将流中的内容写入到io对象中
    io.Write(stream);
}
抱歉，给定的代码片段不完整，缺少了必要的上下文和变量定义，无法为其添加注释。如果您有完整的代码片段需要解释和注释，请提供完整的代码片段。谢谢！
```