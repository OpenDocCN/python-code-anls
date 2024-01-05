# `33_Dice\csharp\Game.cs`

```
# 导入必要的命名空间
using System;
using System.Linq;

namespace BasicComputerGames.Dice
{
    public class Game
    {
        # 创建一个RollGenerator对象作为私有成员变量
        private readonly RollGenerator _roller = new RollGenerator();

        # 游戏循环
        public void GameLoop()
        {
            # 显示游戏介绍文本
            DisplayIntroText();

            # 使用do-while循环执行以下代码块
            do
            {
                # 获取用户输入的掷骰子次数
                int numRolls = GetInput();
                # 计算掷骰子结果的次数
                var counter = CountRolls(numRolls);
                # 显示掷骰子结果的次数
                DisplayCounts(counter);
		} while (TryAgain());  // 使用do-while循环来实现用户选择是否重新开始游戏

		private void DisplayIntroText()  // 显示游戏介绍文本的方法
		{
			Console.ForegroundColor = ConsoleColor.Yellow;  // 设置控制台前景色为黄色
			Console.WriteLine("Dice");  // 输出"Dice"
			Console.WriteLine("Creating Computing, Morristown, New Jersey."); Console.WriteLine();  // 输出"Creating Computing, Morristown, New Jersey."并换行

			Console.ForegroundColor = ConsoleColor.DarkGreen;  // 设置控制台前景色为深绿色
			Console.WriteLine("Original code by Danny Freidus.");  // 输出"Original code by Danny Freidus."
			Console.WriteLine("Originally published in 1978 in the book 'Basic Computer Games' by David Ahl.");  // 输出"Originally published in 1978 in the book 'Basic Computer Games' by David Ahl."
			Console.WriteLine("Modernized and converted to C# in 2021 by James Curran (noveltheory.com).");  // 输出"Modernized and converted to C# in 2021 by James Curran (noveltheory.com)."
			Console.WriteLine();

			Console.ForegroundColor = ConsoleColor.Gray;  // 设置控制台前景色为灰色
			Console.WriteLine("This program simulates the rolling of a pair of dice.");  // 输出"This program simulates the rolling of a pair of dice."
			Console.WriteLine("You enter the number of times you want the computer to");  // 输出"You enter the number of times you want the computer to"
			Console.WriteLine("'roll' the dice. Watch out, very large numbers take");  // 输出"'roll' the dice. Watch out, very large numbers take"
			Console.WriteLine("a long time. In particular, numbers over 10 million.");  // 输出"a long time. In particular, numbers over 10 million."
# 输出一个空行
Console.WriteLine();

# 设置控制台前景色为黄色，并输出提示信息
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("Press any key start the game.");

# 等待用户按下任意键
Console.ReadKey(true);
```

```
# 获取用户输入的整数
private int GetInput()
{
    # 初始化变量 num 为 -1
    int num = -1;
    
    # 输出一个空行
    Console.WriteLine();
    
    # 使用 do-while 循环，提示用户输入掷骰子的次数，直到用户输入一个整数为止
    do
    {
        Console.WriteLine();
        Console.Write("How many rolls? ");
    } while (!Int32.TryParse(Console.ReadLine(), out num));

    # 返回用户输入的整数
    return num;
}
		private  void DisplayCounts(int[] counter)
		{
			// 打印空行
			Console.WriteLine();
			// 打印表头
			Console.WriteLine($"\tTotal\tTotal Number");
			Console.WriteLine($"\tSpots\tof Times");
			Console.WriteLine($"\t===\t=========");
			// 遍历计数器数组，打印每个点数和其出现的次数
			for (var n = 1; n < counter.Length; ++n)
			{
				Console.WriteLine($"\t{n + 1,2}\t{counter[n],9:#,0}");
			}
			// 打印空行
			Console.WriteLine();
		}

		private  int[] CountRolls(int x)
		{
			// 使用 Roller 对象生成 x 次掷骰子结果，然后对结果进行计数
			var counter = _roller.Rolls().Take(x).Aggregate(new int[12], (cntr, r) =>
			{
				// 根据掷骰子结果更新计数器数组
				cntr[r.die1 + r.die2 - 1]++;
				return cntr;
			});
			return counter;
		}
		/// <summary>
		/// Prompt the player to try again, and wait for them to press Y or N.
		/// </summary>
		/// <returns>Returns true if the player wants to try again, false if they have finished playing.</returns>
		private bool TryAgain()
		{
			// 设置控制台前景色为白色
			Console.ForegroundColor = ConsoleColor.White;
			// 输出提示信息
			Console.WriteLine("Would you like to try again? (Press 'Y' for yes or 'N' for no)");

			// 设置控制台前景色为黄色
			Console.ForegroundColor = ConsoleColor.Yellow;
			// 输出提示符
			Console.Write("> ");

			// 定义变量保存用户按下的键
			char pressedKey;
			// 循环直到获取到有效的输入
			do
			{
				// 读取一个键，不在屏幕上显示
				ConsoleKeyInfo key = Console.ReadKey(true);
// 将按键转换为大写，这样我们就不需要关心大小写
pressedKey = Char.ToUpper(key.KeyChar);
// 这是我们认识的按键吗？如果不是，就继续循环
} while (pressedKey != 'Y' && pressedKey != 'N');
// 在屏幕上显示结果
Console.WriteLine(pressedKey);

// 如果玩家按下了'Y'，则返回true，否则返回false
return (pressedKey == 'Y');
```