# `11_Bombardment\csharp\Bombardment.cs`

```
# 导入系统模块
import System
# 导入集合模块
import System.Collections.Generic

# 命名空间 Bombardment
namespace Bombardment:
    # 游戏 Bombardment 类
    internal class Bombardment:
        # 最大网格大小
        private static int MAX_GRID_SIZE = 25
        # 最大部队数
        private static int MAX_PLATOONS = 4
        # 创建随机数对象
        private static Random random = new Random()
        # 计算机位置列表
        private List<int> computerPositions = new List<int>()
        # 玩家位置列表
        private List<int> playerPositions = new List<int>()
        # 计算机猜测列表
        private List<int> computerGuesses = new List<int>()
# 打印游戏开始信息
private void PrintStartingMessage()
{
    # 打印游戏标题和地点
    Console.WriteLine("{0}BOMBARDMENT", new string(' ', 33));
    Console.WriteLine("{0}CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY", new string(' ', 15));
    Console.WriteLine();
    Console.WriteLine();
    Console.WriteLine();

    # 打印游戏规则
    Console.WriteLine("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU");
    Console.WriteLine("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.");
    Console.WriteLine("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.");
    Console.WriteLine("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.");
    Console.WriteLine();
    Console.WriteLine("THE OBJECT OF THE GAME IS TO FIRE MISSLES AT THE");
    Console.WriteLine("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.");
    Console.WriteLine("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS");
    Console.WriteLine("FIRST IS THE WINNER.");
    Console.WriteLine();
    Console.WriteLine("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!");
}
            // 打印空行
            Console.WriteLine();
            // 打印提示信息
            Console.WriteLine("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.");

            // 作为替代重复调用 WriteLine()，我们可以打印换行字符五次
            Console.Write(new string('\n', 5));

            // 打印一个示例游戏板（可能最初设计为在纸上打印并进行游戏）
            for (var i = 1; i <= 25; i += 5)
            {
                // 通过使用格式 {tokenPosition, padding} 可以对标记替换进行填充
                // 负值的填充会导致输出左对齐
                Console.WriteLine("{0,-3}{1,-3}{2,-3}{3,-3}{4,-3}", i, i + 1, i + 2, i + 3, i + 4);
            }

            // 打印空行
            Console.WriteLine("\n");
        }

        // 为计算机的部队生成5个随机位置
        // 在棋盘上放置计算机的部队
        private void PlaceComputerPlatoons()
        {
            // 使用随机数生成下一个位置
            var nextPosition = random.Next(1, MAX_GRID_SIZE);
            // 如果计算机的位置列表中不包含该位置，则将其添加进去
            if (!computerPositions.Contains(nextPosition))
            {
                computerPositions.Add(nextPosition);
            }

        } while (computerPositions.Count < MAX_PLATOONS); // 当计算机的位置数量小于最大部队数量时继续执行

        // 存储玩家的位置
        private void StoreHumanPositions()
        {
            Console.WriteLine("WHAT ARE YOUR FOUR POSITIONS"); // 提示玩家输入四个位置

            // 原始游戏假设输入将是一个包含五个逗号分隔值的字符串，都在同一行上
            // 例如：12,22,1,4,17
            var input = Console.ReadLine(); // 读取玩家输入的值
            // 将输入的字符串按逗号分隔成字符串数组
            var playerPositionsAsStrings = input.Split(",");
            // 遍历字符串数组，将每个字符串转换成整数并添加到 playerPositions 列表中
            foreach (var playerPosition in playerPositionsAsStrings) {
                playerPositions.Add(int.Parse(playerPosition));
            }
        }

        // 人类玩家的回合
        private void HumanTurn()
        {
            // 提示玩家输入发射导弹的位置
            Console.WriteLine("WHERE DO YOU WISH TO FIRE YOUR MISSLE");
            // 读取玩家输入的位置并转换成整数
            var input = Console.ReadLine();
            var humanGuess = int.Parse(input);

            // 如果计算机的位置列表包含玩家猜测的位置
            if(computerPositions.Contains(humanGuess))
            {
                // 提示玩家击中了计算机的一个据点
                Console.WriteLine("YOU GOT ONE OF MY OUTPOSTS!");
                // 从计算机的位置列表中移除玩家猜测的位置
                computerPositions.Remove(humanGuess);

                // 根据计算机的位置列表长度进行不同的操作
                switch(computerPositions.Count)
                {
                    // 当计算机的位置列表长度为3时
                    case 3:
                        # 打印消息表示已完成一个任务
                        Console.WriteLine("ONE DOWN, THREE TO GO.");
                        # 跳出 switch 语句
                        break;
                    # 如果 case 的值为 2
                    case 2:
                        # 打印消息表示已完成两个任务
                        Console.WriteLine("TWO DOWN, TWO TO GO.");
                        # 跳出 switch 语句
                        break;
                    # 如果 case 的值为 1
                    case 1:
                        # 打印消息表示已完成三个任务
                        Console.WriteLine("THREE DOWN, ONE TO GO.");
                        # 跳出 switch 语句
                        break;
                    # 如果 case 的值为 0
                    case 0:
                        # 打印消息表示任务失败
                        Console.WriteLine("YOU GOT ME, I'M GOING FAST.");
                        # 打印消息表示敌方将会恢复并反击
                        Console.WriteLine("BUT I'LL GET YOU WHEN MY TRANSISTO&S RECUP%RA*E!");
                        # 跳出 switch 语句
                        break;
                }
            }
            # 如果条件不满足
            else
            {
                # 打印消息表示敌方未命中，轮到玩家行动
                Console.WriteLine("HA, HA YOU MISSED. MY TURN NOW:");
            }
        }
        # 生成计算机猜测的数字
        private int GenerateComputerGuess()
        {
            int computerGuess;
            # 使用随机数生成器生成1到25之间的随机数
            do
            {
                computerGuess = random.Next(1, 25);
            }
            # 如果计算机猜测的数字已经在之前猜测过的列表中，则重新生成
            while(computerGuesses.Contains(computerGuess));
            # 将计算机猜测的数字添加到已猜测列表中
            computerGuesses.Add(computerGuess);

            return computerGuess;
        }

        # 计算机的回合
        private void ComputerTurn()
        {
            # 生成计算机的猜测
            var computerGuess = GenerateComputerGuess();

            # 如果计算机猜测的数字在玩家位置列表中
            if (playerPositions.Contains(computerGuess))
            {
                # 打印命中信息
                Console.WriteLine("I GOT YOU. IT WON'T BE LONG NOW. POST {0} WAS HIT.", computerGuess);
                # 从玩家位置列表中移除计算机猜测的位置
                playerPositions.Remove(computerGuess);

                # 根据玩家位置列表的数量进行不同的操作
                switch(playerPositions.Count)
                {
                    case 3:
                        # 如果玩家位置列表数量为3，输出提示信息
                        Console.WriteLine("YOU HAVE ONLY THREE OUTPOSTS LEFT.");
                        break;
                    case 2:
                        # 如果玩家位置列表数量为2，输出提示信息
                        Console.WriteLine("YOU HAVE ONLY TWO OUTPOSTS LEFT.");
                        break;
                    case 1:
                        # 如果玩家位置列表数量为1，输出提示信息
                        Console.WriteLine("YOU HAVE ONLY ONE OUTPOST LEFT.");
                        break;
                    case 0:
                        # 如果玩家位置列表数量为0，输出提示信息并显示计算机猜测的位置
                        Console.WriteLine("YOU'RE DEAD. YOUR LAST OUTPOST WAS AT {0}. HA, HA, HA.", computerGuess);
                        Console.WriteLine("BETTER LUCK NEXT TIME.");
                        break;
                }
            }
            else
            {
                # 打印消息，显示计算机猜测的结果
                Console.WriteLine("I MISSED YOU, YOU DIRTY RAT. I PICKED {0}. YOUR TURN:", computerGuess);
            }
        }

        public void Play()
        {
            # 打印游戏开始的消息
            PrintStartingMessage();
            # 放置计算机的兵营
            PlaceComputerPlatoons();
            # 存储玩家的位置

            StoreHumanPositions();

            while (playerPositions.Count > 0 && computerPositions.Count > 0)
            {
                # 玩家的回合
                HumanTurn();

                if (computerPositions.Count > 0)
                {
                    # 计算机的回合
                    ComputerTurn();
                }
            }
抱歉，给定的代码片段不完整，无法为其添加注释。
```