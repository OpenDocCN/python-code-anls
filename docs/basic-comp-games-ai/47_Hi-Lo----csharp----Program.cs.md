# `47_Hi-Lo\csharp\Program.cs`

```
// 输出一行空格和"HI LO"字符串
Console.WriteLine(Tab(34) + "HI LO");
// 输出一行空格和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"字符串
Console.WriteLine(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
// 输出空行
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();
// 输出"THIS IS THE GAME OF HI LO."
Console.WriteLine("THIS IS THE GAME OF HI LO.");
// 输出空行
Console.WriteLine();
// 输出"YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE"字符串
Console.WriteLine("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE");
// 输出"HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU"字符串
Console.WriteLine("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU");
// 输出"GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!"字符串
Console.WriteLine("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!");
// 输出"THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,"字符串
Console.WriteLine("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,");
// 输出"IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS."字符串
Console.WriteLine("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.");
// 输出空行
Console.WriteLine();

// rnd 是我们的随机数生成器
// 创建一个新的随机数生成器对象
Random rnd = new Random();

// playAgain 是一个布尔类型的变量，用于表示是否再玩一次的标志
bool playAgain = false;
int totalWinnings = 0; // 初始化总奖金为0

do // 游戏循环
{
    int jackpot = rnd.Next(100) + 1; // 生成1到100之间的随机数作为奖池
    int guess = 1; // 初始化猜测次数为1

    while (true) // 猜测循环
    {
        Console.WriteLine(); // 输出空行
        int amount = ReadInt("YOUR GUESS "); // 读取用户输入的猜测数

        if (amount == jackpot) // 如果猜中了
        {
            Console.WriteLine($"GOT IT!!!!!!!!!!   YOU WIN {jackpot} DOLLARS."); // 输出猜中奖池金额的消息
            totalWinnings += jackpot; // 将奖金累加到总奖金中
            Console.WriteLine($"YOUR TOTAL WINNINGS ARE NOW {totalWinnings} DOLLARS."); // 输出当前总奖金
            break; // 退出猜测循环
        }
        else if (amount > jackpot) // 如果猜测数大于奖池金额
        {
            Console.WriteLine("YOUR GUESS IS TOO HIGH.");  # 如果猜测的数字太高，则输出提示信息
        }
        else
        {
            Console.WriteLine("YOUR GUESS IS TOO LOW.");   # 如果猜测的数字太低，则输出提示信息
        }

        guess++;  # 猜测次数加一
        if (guess > 6)  # 如果猜测次数超过6次
        {
            Console.WriteLine($"YOU BLEW IT...TOO BAD...THE NUMBER WAS {jackpot}");  # 输出猜测次数超过6次的提示信息，并显示正确数字
            break;  # 结束循环
        }
    }

    Console.WriteLine();  # 输出空行
    Console.Write("PLAY AGAIN (YES OR NO) ");  # 提示用户是否再次玩游戏
    playAgain = Console.ReadLine().ToUpper().StartsWith("Y");  # 读取用户输入并转换为大写，判断是否以"Y"开头，将结果赋值给playAgain变量
} while (playAgain);
// 使用 do-while 循环来实现游戏再次进行的逻辑

Console.WriteLine();
Console.WriteLine("SO LONG.  HOPE YOU ENJOYED YOURSELF!!!");
// 输出结束语句

// Tab(n) returns n spaces
// 定义一个函数，返回 n 个空格的字符串

// ReadInt asks the user to enter a number
// 定义一个函数，提示用户输入一个数字

static int ReadInt(string question)
{
    while (true)
    {
        Console.Write(question);
        var input = Console.ReadLine().Trim();
        if (int.TryParse(input, out int value))
        {
            return value;
        }
        Console.WriteLine("!Invalid Number Entered.");
        // 如果用户输入的不是数字，则输出错误信息
    }
```

这部分代码是一个缩进错误，应该删除。
```