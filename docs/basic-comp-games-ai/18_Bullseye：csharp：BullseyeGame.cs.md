# `18_Bullseye\csharp\BullseyeGame.cs`

```
namespace Bullseye
{
    /// <summary>
    /// Class encompassing the game
    /// </summary>
    public class BullseyeGame
    {
        private readonly List<Player> _players; // 声明私有成员变量 _players，用于存储玩家列表

        // define a constant for the winning score so that it is
        // easy to change again in the future
        private const int WinningScore = 200; // 定义一个常量 WinningScore，表示获胜分数，方便将来修改

        public BullseyeGame()
        {
            // create the initial list of players; list is empty, but
            // the setup of the game will add items to this list
            _players = new List<Player>(); // 初始化玩家列表为空
        }
        public void Run()
        {
            // 运行游戏
            PrintIntroduction(); // 打印游戏介绍

            SetupGame(); // 设置游戏

            PlayGame(); // 进行游戏

            PrintResults(); // 打印游戏结果
        }

        private void SetupGame()
        {
            // 首先，允许用户输入要参与游戏的玩家数量。如果用户输入了负数、单词或者太多的玩家数量，就需要进行额外的检查以确保用户没有做出太疯狂的操作。循环直到用户输入有效的内容。
            bool validPlayerCount; // 用于标记玩家数量是否有效
            int playerCount; // 玩家数量
                // 输出空行
                Console.WriteLine();
                // 提示用户输入玩家数量
                Console.Write("HOW MANY PLAYERS? ");
                // 读取用户输入
                string? input = Console.ReadLine();

                // 假设用户输入不正确 - 下一步将验证输入
                validPlayerCount = false;

                // 尝试将用户输入转换为整数，如果成功则执行以下代码
                if (Int32.TryParse(input, out playerCount))
                {
                    // 如果玩家数量大于0且小于等于20，则将validPlayerCount设置为true
                    if (playerCount > 0 && playerCount <= 20)
                    {
                        validPlayerCount = true;
                    }
                    // 否则输出错误信息
                    else
                    {
                        Console.WriteLine("YOU MUST ENTER A NUMBER BETWEEN 1 AND 20!");
                    }
            }
            else
            {
                Console.WriteLine("YOU MUST ENTER A NUMBER");
            }
        }
        while (!validPlayerCount);

        // Next, allow the user to enter names for the players; as each
        // name is entered, create a Player object to track the name
        // and their score, and save the object to the list in this class
        // so the rest of the game has access to the set of players
        for (int i = 0; i < playerCount; i++)
        {
            string? playerName = String.Empty; // 初始化玩家姓名为空字符串
            do
            {
                Console.Write($"NAME OF PLAYER #{i+1}? "); // 提示用户输入玩家姓名
                playerName = Console.ReadLine(); // 读取用户输入的玩家姓名
// 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    // 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    // 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    // 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    // 关闭 ZIP 对象
    zip.close()
    // 返回结果字典
    return fdict
                // 打印当前回合数
                Console.WriteLine($"ROUND {round}");
                // 打印分隔线
                Console.WriteLine("--------------");

                // 遍历玩家列表
                foreach (Player player in _players)
                {
                    // 询问玩家如何投掷
                    Console.Write($"{player.Name.ToUpper()}'S THROW: ");
                    // 读取玩家输入
                    string? input = Console.ReadLine();

                    // 根据玩家输入确定概率
                    int[] probabilities;
                    switch (input)
                    {
                        // 如果输入为"1"
                        case "1":
                        {
                            // 设置概率数组
                            probabilities = new int[] { 65, 55, 50, 50 };
                            break;
                        }
                        // 如果输入为"2"
                        case "2":
                        {
                            # 定义一个整型数组，表示不同情况下的概率
                            probabilities = new int[] { 99, 77, 43, 1 };
                            # 结束当前 case 分支
                            break;
                        }
                        case "3":
                        {
                            # 定义一个整型数组，表示不同情况下的概率
                            probabilities = new int[] { 95, 75, 45, 5 };
                            # 结束当前 case 分支
                            break;
                        }
                        default:
                        {
                            # 如果用户输入了无效的值，假装他们在投掷飞镖时绊倒了
                            # 他们要么打中靶心，要么完全没打中
                            probabilities = new int[] { 95, 95, 95, 95 };
                            # 输出提示信息
                            Console.Write("TRIP! ");
                            # 结束当前 case 分支
                            break;
                        }
                    }
                    // 生成一个介于0到100之间的随机数
                    int chance = random.Next(0, 101);

                    // 如果随机数大于第一个概率值，则给玩家加40分，并打印信息
                    if (chance > probabilities[0])
                    {
                        player.Score += 40;
                        Console.WriteLine("BULLSEYE!!  40 POINTS!");
                    }
                    // 如果随机数大于第二个概率值，则给玩家加30分，并打印信息
                    else if (chance > probabilities[1])
                    {
                        player.Score += 30;
                        Console.WriteLine("30-POINT ZONE!");
                    }
                    // 如果随机数大于第三个概率值，则给玩家加20分，并打印信息
                    else if (chance > probabilities[2])
                    {
                        player.Score += 20;
                        Console.WriteLine("20-POINT ZONE");
                    }
                    else if (chance > probabilities[3])  # 如果 chance 大于 probabilities[3]
                    {
                        player.Score += 10;  # 玩家得分增加 10 分
                        Console.WriteLine("WHEW!  10 POINTS.");  # 输出信息：WHEW!  10 POINTS.
                    }
                    else  # 否则
                    {
                        // missed it  # 注释：错过了目标
                        Console.WriteLine("MISSED THE TARGET!  TOO BAD.");  # 输出信息：MISSED THE TARGET!  TOO BAD.
                    }

                    // check to see if the player has won - if they have, then
                    // break out of the loops
                    if (player.Score > WinningScore)  # 检查玩家是否获胜 - 如果是，则跳出循环
                    {
                        Console.WriteLine();  # 输出空行
                        Console.WriteLine("WE HAVE A WINNER!!");  # 输出信息：WE HAVE A WINNER!!
                        Console.WriteLine($"{player.Name.ToUpper()} SCORED {player.Score} POINTS.");  # 输出信息：玩家姓名大写得分点数
                        Console.WriteLine();  # 输出空行
                        isOver = true; // 设置循环结束的标志为真，跳出 do/while 循环
                        break; // 跳出 foreach (player) 循环
                    }

                    Console.WriteLine(); // 输出空行
                }
            }
            while (!isOver); // 当 isOver 为假时继续循环
        }

        private void PrintResults()
        {
            // 为了炫耀，打印出所有的分数，但按照最高分排序
            var sorted = _players.OrderByDescending(p => p.Score); // 根据玩家得分降序排序

            // padding 用于使结果对齐 - 结果应该看起来像这样：
            //      PLAYER       SCORE
            //      Bravo          210
            // 输出玩家得分排名的表头
            Console.WriteLine("PLAYER       SCORE");
            // 遍历排序后的玩家列表，输出玩家姓名和得分
            foreach (var player in sorted)
            {
                Console.WriteLine($"{player.Name.PadRight(12)} {player.Score.ToString().PadLeft(5)}");
            }

            // 输出空行
            Console.WriteLine();
            // 输出感谢信息
            Console.WriteLine("THANKS FOR THE GAME.");
        }

        // 输出游戏介绍
        private void PrintIntroduction()
        {
            Console.WriteLine(Title);
            Console.WriteLine();
            Console.WriteLine(Introduction);
            Console.WriteLine();
            Console.WriteLine(Operations);
        }
# 定义常量Title，存储游戏标题
private const string Title = @"
                BULLSEYE
CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY";

# 定义常量Introduction，存储游戏介绍
private const string Introduction = @"
IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET
WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS
TO GET 200 POINTS.";

# 定义常量Operations，存储游戏操作说明
private const string Operations = @"
THROW   DESCRIPTION         PROBABLE SCORE
  1     FAST OVERARM        BULLSEYE OR COMPLETE MISS
  2     CONTROLLED OVERARM  10, 20, OR 30 POINTS
  3     UNDERARM            ANYTHING";
```