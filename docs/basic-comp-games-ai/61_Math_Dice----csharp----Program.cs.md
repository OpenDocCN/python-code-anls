# `61_Math_Dice\csharp\Program.cs`

```
// 命名空间 MathDice
namespace MathDice
{
    // 定义 Program 类
    public static class Program
    {
        // 创建一个静态的 Random 对象
        readonly static Random random = new Random();

        // 定义两个静态的整型变量 DieOne 和 DieTwo，并初始化为 0
        static int DieOne = 0;
        static int DieTwo = 0;

        // 定义常量字符串 NoPips、LeftPip、CentrePip、RightPip、TwoPips 和 Edge
        private const string NoPips = "I     I";
        private const string LeftPip = "I *   I";
        private const string CentrePip = "I  *  I";
        private const string RightPip = "I   * I";
        private const string TwoPips = "I * * I";
        private const string Edge = " ----- ";

        // 定义 Main 方法
        static void Main(string[] args)
        {
            int answer; // 声明一个整型变量 answer

            GameState gameState = GameState.FirstAttempt; // 声明一个 GameState 类型的变量 gameState，并赋初值为 GameState.FirstAttempt

            Console.WriteLine("MATH DICE".CentreAlign()); // 在控制台输出居中对齐的字符串 "MATH DICE"
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".CentreAlign()); // 在控制台输出居中对齐的字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
            Console.WriteLine(); // 在控制台输出空行
            Console.WriteLine(); // 在控制台输出空行
            Console.WriteLine(); // 在控制台输出空行
            Console.WriteLine("THIS PROGRAM GENERATES SUCCESSIVE PICTURES OF TWO DICE."); // 在控制台输出字符串 "THIS PROGRAM GENERATES SUCCESSIVE PICTURES OF TWO DICE."
            Console.WriteLine("WHEN TWO DICE AND AN EQUAL SIGN FOLLOWED BY A QUESTION"); // 在控制台输出字符串 "WHEN TWO DICE AND AN EQUAL SIGN FOLLOWED BY A QUESTION"
            Console.WriteLine("MARK HAVE BEEN PRINTED, TYPE YOUR ANSWER AND THE RETURN KEY."); // 在控制台输出字符串 "MARK HAVE BEEN PRINTED, TYPE YOUR ANSWER AND THE RETURN KEY."
            Console.WriteLine("TO CONCLUDE THE LESSON, TYPE CONTROL-C AS YOUR ANSWER."); // 在控制台输出字符串 "TO CONCLUDE THE LESSON, TYPE CONTROL-C AS YOUR ANSWER."
            Console.WriteLine(); // 在控制台输出空行
            Console.WriteLine(); // 在控制台输出空行

            while (true) // 进入无限循环
            {
                if (gameState == GameState.FirstAttempt) // 如果 gameState 的值等于 GameState.FirstAttempt
                {
                    Roll(ref DieOne);  // 调用 Roll 函数，将 DieOne 的值设置为一个随机数
                    Roll(ref DieTwo);  // 调用 Roll 函数，将 DieTwo 的值设置为一个随机数

                    DrawDie(DieOne);  // 调用 DrawDie 函数，绘制 DieOne 的图像
                    Console.WriteLine("   +");  // 在控制台输出一个加号
                    DrawDie(DieTwo);  // 调用 DrawDie 函数，绘制 DieTwo 的图像
                }

                answer = GetAnswer();  // 调用 GetAnswer 函数，获取用户输入的答案

                if (answer == DieOne + DieTwo)  // 如果用户输入的答案等于两个骰子的和
                {
                    Console.WriteLine("RIGHT!");  // 在控制台输出 "RIGHT!"
                    Console.WriteLine();  // 在控制台输出一个空行
                    Console.WriteLine("THE DICE ROLL AGAIN...");  // 在控制台输出 "THE DICE ROLL AGAIN..."

                    gameState = GameState.FirstAttempt;  // 将 gameState 设置为 GameState.FirstAttempt
                }
                else
                {
                    if (gameState == GameState.FirstAttempt)
                    {
                        # 如果游戏状态为第一次尝试，打印提示信息并将游戏状态设置为第二次尝试
                        Console.WriteLine("NO, COUNT THE SPOTS AND GIVE ANOTHER ANSWER.");
                        gameState = GameState.SecondAttempt;
                    }
                    else
                    {
                        # 如果游戏状态不是第一次尝试，打印提示信息和答案，并将游戏状态设置为第一次尝试
                        Console.WriteLine($"NO, THE ANSWER IS{DieOne + DieTwo}");
                        Console.WriteLine();
                        Console.WriteLine("THE DICE ROLL AGAIN...");
                        gameState = GameState.FirstAttempt;
                    }
                }
            }
        }

        private static int GetAnswer()
        {
            # 定义一个整数变量 answer
            int answer;
            Console.Write("      =?");  // 输出提示信息，要求用户输入
            var input = Console.ReadLine();  // 从控制台读取用户输入的内容并存储在变量input中

            int.TryParse(input, out answer);  // 尝试将用户输入的内容转换为整数并存储在变量answer中

            return answer;  // 返回转换后的整数值
        }

        private static void DrawDie(int pips)  // 绘制骰子的方法，参数为骰子的点数
        {
            Console.WriteLine(Edge);  // 输出边框
            Console.WriteLine(OuterRow(pips, true));  // 输出骰子的上半部分
            Console.WriteLine(CentreRow(pips));  // 输出骰子的中间部分
            Console.WriteLine(OuterRow(pips, false));  // 输出骰子的下半部分
            Console.WriteLine(Edge);  // 输出边框
            Console.WriteLine();  // 输出空行
        }

        private static void Roll(ref int die) => die = random.Next(1, 7);  // 掷骰子的方法，通过引用修改骰子的点数
        // 定义一个私有方法，根据骰子点数和位置返回对应的字符串
        private static string OuterRow(int pips, bool top)
        {
            // 使用 switch 语句根据骰子点数进行匹配
            return pips switch
            {
                // 当点数为1时，返回无点的字符串
                1 => NoPips,
                // 当点数为2或3时，根据位置返回左侧点或右侧点的字符串
                var x when x == 2 || x == 3 => top ? LeftPip : RightPip,
                // 其他情况返回两个点的字符串
                _ => TwoPips
            };
        }

        // 定义一个私有方法，根据骰子点数返回对应的字符串
        private static string CentreRow(int pips)
        {
            // 使用 switch 语句根据骰子点数进行匹配
            return pips switch
            {
                // 当点数为2或4时，返回无点的字符串
                var x when x == 2 || x == 4 => NoPips,
                // 当点数为6时，返回两个点的字符串
                6 => TwoPips,
                // 其他情况返回中间一个点的字符串
                _ => CentrePip
            };
        }
    }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```