# `d:/src/tocomm/basic-computer-games\73_Reverse\csharp\Reverse\Program.cs`

```
# 定义一个静态变量，表示数组的长度为9
private static int arrayLength = 9;

# 主程序入口
static void Main(string[] args)
{
    # 打印标题
    PrintTitle();
    # 提示用户是否需要规则
    Console.Write("DO YOU WANT THE RULES? ");
    # 读取用户输入
    var needRulesInput = Console.ReadLine();
    Console.WriteLine();
    # 如果用户输入为"YES"（不区分大小写），则显示规则
    if (string.Equals(needRulesInput, "YES", StringComparison.OrdinalIgnoreCase))
    {
        DisplayRules();
    }

    # 初始化一个空字符串，用于后续判断是否需要再次执行循环
    var tryAgain = string.Empty;
    # 当用户输入不是"NO"（不区分大小写）时，执行循环
    while (!string.Equals(tryAgain, "NO", StringComparison.OrdinalIgnoreCase))
                // 创建一个 Reverser 对象，用于反转数组
                var reverser = new Reverser(arrayLength);

                // 打印数组的当前状态
                Console.WriteLine("HERE WE GO ... THE LIST IS:");
                PrintList(reverser.GetArrayString());

                // 初始化变量，用于记录数组是否按升序排列以及反转操作的次数
                var arrayIsInAscendingOrder = false;
                var numberOfMoves = 0;

                // 当数组未按升序排列时，进行反转操作
                while (arrayIsInAscendingOrder == false)
                {
                    // 读取用户输入的索引值
                    int index = ReadNextInput();

                    // 如果索引值为0，跳出循环
                    if (index == 0)
                    {
                        break;
                    }

                    // 对数组进行反转操作
                    reverser.Reverse(index);
                    // 打印反转后的数组状态
                    PrintList(reverser.GetArrayString());
                    // 检查数组是否按升序排列
                    arrayIsInAscendingOrder = reverser.IsArrayInAscendingOrder();
                    // 记录反转操作的次数
                    numberOfMoves++;
                }  # 结束 if 语句块

                if (arrayIsInAscendingOrder):  # 如果数组按升序排列
                    Console.WriteLine($"YOU WON IT IN {numberOfMoves} MOVES!!!")  # 打印玩家赢得游戏所用的步数

                Console.WriteLine()  # 打印空行
                Console.WriteLine()  # 再次打印空行
                Console.Write("TRY AGAIN (YES OR NO) ")  # 提示玩家是否要再次尝试
                tryAgain = Console.ReadLine()  # 读取玩家输入的再次尝试的选择
            }  # 结束 while 循环

            Console.WriteLine()  # 打印空行
            Console.WriteLine("OK HOPE YOU HAD FUN!!")  # 打印消息，希望玩家玩得开心
        }  # 结束方法

        private static int ReadNextInput():  # 定义一个私有方法 ReadNextInput
            # 提示用户输入需要反转的数量
            Console.Write("HOW MANY SHALL I REVERSE? ");
            # 读取用户输入的整数
            var input = ReadIntegerInput();
            # 当输入大于9或小于0时，循环提示用户重新输入
            while (input > 9 || input < 0)
            {
                # 如果输入大于9，提示用户输入的数量太多
                if (input > 9)
                {
                    Console.WriteLine($"OOPS! TOO MANY! I CAN REVERSE AT MOST {arrayLength}");
                }

                # 如果输入小于0，提示用户输入的数量太少
                if (input < 0)
                {
                    Console.WriteLine($"OOPS! TOO FEW! I CAN REVERSE BETWEEN 1 AND {arrayLength}");
                }
                # 再次提示用户输入需要反转的数量
                Console.Write("HOW MANY SHALL I REVERSE? ");
                # 重新读取用户输入的整数
                input = ReadIntegerInput();
            }

            # 返回用户输入的需要反转的数量
            return input;
        }
# 读取用户输入的整数并返回
private static int ReadIntegerInput()
{
    # 从控制台读取用户输入
    var input = Console.ReadLine();
    # 尝试将输入转换为整数并将结果存储在 index 变量中
    int.TryParse(input, out var index);
    # 返回转换后的整数
    return index;
}

# 在控制台打印列表
private static void PrintList(string list)
{
    # 在控制台打印空行
    Console.WriteLine();
    # 在控制台打印列表内容
    Console.WriteLine(list);
    # 在控制台打印空行
    Console.WriteLine();
}

# 在控制台打印标题
private static void PrintTitle()
{
    # 在控制台打印标题
    Console.WriteLine("\t\t   REVERSE");
    # 在控制台打印 CREATIVE COMPUTING  MORRISTON, NEW JERSEY
    Console.WriteLine("  CREATIVE COMPUTING  MORRISTON, NEW JERSEY");
    # 在控制台打印空行
    Console.WriteLine();
    # 在控制台打印空行
    Console.WriteLine();
}
            Console.WriteLine("REVERSE -- A GAME OF SKILL"); // 打印游戏标题
            Console.WriteLine(); // 打印空行
        }

        private static void DisplayRules()
        {
            Console.WriteLine(); // 打印空行
            Console.WriteLine("THIS IS THE GAME OF 'REVERSE'. TO WIN, ALL YOU HAVE"); // 打印游戏规则
            Console.WriteLine("TO DO IS ARRANGE A LIST OF NUMBERS (1 THOUGH 9 )"); // 打印游戏规则
            Console.WriteLine("IN NUMERICAL ORDER FROM LEFT TO RIGHT. TO MOVE, YOU"); // 打印游戏规则
            Console.WriteLine("TELL ME HOW MANY NUMBERS (COUNTING FROM THE LEFT) TO"); // 打印游戏规则
            Console.WriteLine("REVERSE. FOR EXAMPLE, IF THE CURRENT LIST IS:"); // 打印游戏规则
            Console.WriteLine(); // 打印空行
            Console.WriteLine("2 3 4 5 1 6 7 8 9"); // 打印示例列表
            Console.WriteLine(); // 打印空行
            Console.WriteLine("AND YOU REVERSE 4, THE RESULT WILL BE:"); // 打印游戏规则
            Console.WriteLine(); // 打印空行
            Console.WriteLine("5 4 3 2 1 6 7 8 9"); // 打印示例结果
            Console.WriteLine(); // 打印空行
            Console.WriteLine("NOW IF YOU REVERSE 5, YOU WIN!"); // 打印游戏规则
# 输出空行
Console.WriteLine();
# 输出数字序列
Console.WriteLine("1 2 3 4 5 6 7 8 9");
# 输出空行
Console.WriteLine();
# 输出提示信息
Console.WriteLine("NO DOUBT YOU WILL LIKE THIS GAME, BUT ");
Console.WriteLine("IF YOU WANT TO QUIT, REVERSE 0 (ZERO)");
# 输出空行
Console.WriteLine();
Console.WriteLine();
```