# `d:/src/tocomm/basic-computer-games\05_Bagels\csharp\Game.cs`

```
# 导入所需的模块
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace BasicComputerGames.Bagels
{
    # 定义一个名为Game的类，继承自GameBase类
    public class Game : GameBase
    {
        # 定义一个名为GameLoop的方法
        public void GameLoop()
        {
            # 调用DisplayIntroText方法，显示游戏介绍文本
            DisplayIntroText();
            # 初始化变量points，用于记录玩家得分
            int points = 0;
            # 开始游戏循环
            do
            {
                # 调用PlayRound方法，进行一轮游戏，并将结果存储在result变量中
                var result = PlayRound();
                # 如果result为真（玩家猜对了），则增加points
                if (result)
                    ++points;
            } while (TryAgain());  # 继续游戏循环，直到玩家选择不再尝试

            # 在控制台输出空行
            Console.WriteLine();
			Console.WriteLine($"A {points} point Bagels buff!!");  // 打印玩家获得的积分
			Console.WriteLine("Hope you had fun. Bye.");  // 打印结束游戏的提示信息
		}

		private const int Length = 3;  // 定义秘密数字的长度为3
		private const int MaxGuesses = 20;  // 定义最大猜测次数为20次

		private bool  PlayRound()  // 定义一个名为PlayRound的方法，返回布尔值
		{
			var secret = BagelNumber.CreateSecretNumber(Length);  // 生成一个秘密数字
			Console.WriteLine("O.K. I have a number in mind.");  // 打印提示信息
			for (int guessNo = 1; guessNo <= MaxGuesses; ++guessNo)  // 循环进行猜测，最多20次
			{
				string strGuess;  // 定义一个字符串变量用于存储玩家的猜测
				BagelValidation isValid;  // 定义一个BagelValidation类型的变量用于存储猜测的有效性
				do
				{
					Console.WriteLine($"Guess #{guessNo}");  // 打印提示信息，让玩家进行猜测
					strGuess = Console.ReadLine();  // 读取玩家输入的猜测
					isValid = BagelNumber.IsValid(strGuess, Length);  // 判断玩家猜测的有效性
				PrintError(isValid);  # 调用PrintError函数，将isValid作为参数传入
				} while (isValid != BagelValidation.Valid);  # 当isValid不等于BagelValidation.Valid时，执行循环

				var guess = new BagelNumber(strGuess);  # 创建一个新的BagelNumber对象，使用strGuess作为参数
				var fermi = 0;  # 初始化fermi变量为0
				var pico = 0;  # 初始化pico变量为0
				(pico, fermi) = secret.CompareTo(guess);  # 调用secret对象的CompareTo方法，将返回的pico和fermi值分别赋给pico和fermi变量
				if(pico + fermi == 0)  # 如果pico和fermi的和等于0
					Console.Write("BAGELS!");  # 输出"BAGELS!"
				else if (fermi == Length)  # 否则，如果fermi等于Length
				{
					Console.WriteLine("You got it!");  # 输出"You got it!"
					return true;  # 返回true
				}
				else  # 否则
				{
					PrintList("Pico ", pico);  # 调用PrintList函数，将"Pico "和pico作为参数传入
					PrintList("Fermi ", fermi);  # 调用PrintList函数，将"Fermi "和fermi作为参数传入
				}
				Console.WriteLine();  # 输出空行
		}

		// 打印错误消息
		private void PrintError(BagelValidation isValid)
		{
			// 根据验证结果进行不同的错误消息打印
			switch (isValid)
			{
				case BagelValidation.NonDigit:
					Console.WriteLine("What?");
					break;

				case BagelValidation.NotUnique:
					Console.WriteLine("Oh, I forgot to tell you that the number I have in mind has no two digits the same.");
					break;
# 打印提示信息，告诉玩家猜测一个特定长度的数字
case BagelValidation.WrongLength:
    Console.WriteLine($"Try guessing a {Length}-digit number.");
    break;

# 如果猜测正确，不做任何操作
case BagelValidation.Valid:
    break;
}

# 打印重复次数的消息
private void PrintList(string msg, int repeat)
{
    for(int i=0; i<repeat; ++i)
        Console.Write(msg);
}

# 显示游戏介绍文本
private void DisplayIntroText()
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("Bagels");
# 输出创建位于新泽西州莫里斯敦的计算机公司的信息
print("Creating Computing, Morristown, New Jersey.")
# 输出空行
print()

# 设置控制台前景色为深绿色，并输出原始代码作者信息
print("Original code author unknow but suspected to be from Lawrence Hall of Science, U.C. Berkley")
print("Originally published in 1978 in the book 'Basic Computer Games' by David Ahl.")
print("Modernized and converted to C# in 2021 by James Curran (noveltheory.com).")
print()

# 设置控制台前景色为灰色，并输出猜数字游戏的提示信息
print("I am thinking of a three-digit number.  Try to guess")
print("my number and I will give you clues as follows:")
print("   pico   - One digit correct but in the wrong position")
print("   fermi  - One digit correct and in the right position")
print("   bagels - No digits correct")
print()

# 设置控制台前景色为黄色，并提示玩家按任意键开始游戏
print("Press any key start the game.")
# 等待用户按下任意键后继续执行程序
			Console.ReadKey(true);
		}
	}
}
```

这段代码是一个C#程序的结尾部分，其中的 `Console.ReadKey(true);` 语句的作用是等待用户按下任意键后继续执行程序。在这个例子中，`true` 参数表示是否显示用户按下的按键。
```