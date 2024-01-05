# `29_Craps\csharp\Craps\UserInterface.cs`

```
using System;  // 导入 System 命名空间
using System.Diagnostics;  // 导入 System.Diagnostics 命名空间

namespace Craps  // 命名空间 Craps
{
    public class UserInterface  // 定义公共类 UserInterface
	{
        public void Intro()  // 定义公共方法 Intro
        {
            Console.WriteLine("                                 CRAPS");  // 在控制台输出 CRAPS
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n");  // 在控制台输出 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
            Console.WriteLine("2,3,12 ARE LOSERS; 4,5,6,8,9,10 ARE POINTS; 7,11 ARE NATURAL WINNERS.");  // 在控制台输出游戏规则

            // In the original game a random number would be generated and then thrown away for as many
            // times as the number the user entered. This is presumably something to do with ensuring
            // that different random numbers will be generated each time the program is run.
            //
            // This is not necessary in C#; the random number generator uses the current time as a seed
        }
// 输出提示信息，要求用户选择一个数字并输入以掷骰子
Console.Write("PICK A NUMBER AND INPUT TO ROLL DICE ");
GetInt(); // 调用 GetInt() 函数，获取用户输入的数字

public int PlaceBet()
{
    // 输出提示信息，要求用户输入赌注金额
    Console.Write("INPUT THE AMOUNT OF YOUR WAGER. ");
    int n = GetInt(); // 调用 GetInt() 函数，获取用户输入的赌注金额
    Console.WriteLine("I WILL NOW THROW THE DICE"); // 输出提示信息，表示将要掷骰子

    return n; // 返回用户输入的赌注金额
}

public bool PlayAgain(int winnings)
{
    // 不知道为什么我们必须输入 5 才能玩
            // 提示玩家选择是否再次玩游戏
            Console.Write("IF YOU WANT TO PLAY AGAIN PRINT 5 IF NOT PRINT 2 ");

            // 根据玩家输入判断是否再次玩游戏
            bool playAgain = (GetInt() == 5);

            // 根据赢得的金额情况输出相应的消息
            if (winnings < 0)
            {
                Console.WriteLine($"YOU ARE NOW UNDER ${-winnings}");
            }
            else if (winnings > 0)
            {
                Console.WriteLine($"YOU ARE NOW OVER ${winnings}");
            }
            else
            {
                Console.WriteLine($"YOU ARE NOW EVEN AT ${winnings}");
            }

            // 返回是否再次玩游戏的布尔值
            return playAgain;
        }
# 如果赢得的金额小于0，则输出“太糟糕了，你赔了。再来一次。”
# 如果赢得的金额大于0，则输出“恭喜---你赢了。再来一次！”
# 如果赢得的金额等于0，则输出“恭喜---你赢得了平局，对一个菜鸟来说还不错。”
public void GoodBye(int winnings)
{
    if (winnings < 0)
    {
        Console.WriteLine("TOO BAD, YOU ARE IN THE HOLE. COME AGAIN.");
    }
    else if (winnings > 0)
    {
        Console.WriteLine("CONGRATULATIONS---YOU CAME OUT A WINNER. COME AGAIN!");
    }
    else
    {
        Console.WriteLine("CONGRATULATIONS---YOU CAME OUT EVEN, NOT BAD FOR AN AMATEUR");
    }
}

# 输出骰子的点数和“NO POINT. I WILL ROLL AGAIN”
public void NoPoint(int diceRoll)
{
    Console.WriteLine($"{diceRoll} - NO POINT. I WILL ROLL AGAIN ");
}
        }

        public void Point(int point)
        {
            // 打印点数，并提示将重新掷骰子
            Console.WriteLine($"{point} IS THE POINT. I WILL ROLL AGAIN");
        }

        public void ShowResult(Result result, int diceRoll, int bet)
        {
            // 根据游戏结果进行不同的处理
            switch (result)
            {
                case Result.naturalWin:
                    // 如果是自然胜利，打印相应信息
                    Console.WriteLine($"{diceRoll} - NATURAL....A WINNER!!!!");
                    Console.WriteLine($"{diceRoll} PAYS EVEN MONEY, YOU WIN {bet} DOLLARS");
                    break;

                case Result.naturalLoss:
                    // 如果是自然失败，打印相应信息
                    Console.WriteLine($"{diceRoll} - CRAPS...YOU LOSE.");
                    Console.WriteLine($"YOU LOSE {bet} DOLLARS.");
                    break;
                case Result.snakeEyesLoss: // 如果结果是蛇眼输了
                    Console.WriteLine($"{diceRoll} - SNAKE EYES....YOU LOSE."); // 打印掷骰子结果和输了的消息
                    Console.WriteLine($"YOU LOSE {bet} DOLLARS."); // 打印输掉的赌注金额
                    break; // 结束该case

                case Result.pointLoss: // 如果结果是点数输了
                    Console.WriteLine($"{diceRoll} - CRAPS. YOU LOSE."); // 打印掷骰子结果和输了的消息
                    Console.WriteLine($"YOU LOSE ${bet}"); // 打印输掉的赌注金额
                    break; // 结束该case

                case Result.pointWin: // 如果结果是点数赢了
                    Console.WriteLine($"{diceRoll} - A WINNER.........CONGRATS!!!!!!!!"); // 打印掷骰子结果和赢了的消息
                    Console.WriteLine($"AT 2 TO 1 ODDS PAYS YOU...LET ME SEE... {2 * bet} DOLLARS"); // 打印赢得的赌注金额
                    break; // 结束该case

                // 包括一个默认情况，以便在枚举值发生变化时，如果忘记添加处理新值的代码，我们会收到警告。
                default: // 默认情况
                    Debug.Assert(false); // 我们永远不应该到达这里。
                    break;  # 结束当前循环，跳出循环体
            }
        }

        private int GetInt()  # 定义一个名为GetInt的私有方法，返回一个整数
        {
            while (true)  # 进入一个无限循环
            {
	            string input = Console.ReadLine();  # 从控制台读取用户输入的字符串
                if (int.TryParse(input, out int n))  # 尝试将输入的字符串转换为整数，如果成功则将转换后的整数赋值给n
                {
                    return n;  # 如果转换成功，则返回整数n
                }
                else  # 如果转换失败
                {
                    Console.Write("ENTER AN INTEGER ");  # 提示用户输入一个整数
                }
            }
        }
    }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```