# `basic-computer-games\47_Hi-Lo\csharp\Program.cs`

```py
// 输出带有空格的字符串 "HI LO"
Console.WriteLine(Tab(34) + "HI LO");
// 输出带有空格的字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
Console.WriteLine(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
// 输出空行
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();
// 输出字符串 "THIS IS THE GAME OF HI LO."
Console.WriteLine("THIS IS THE GAME OF HI LO.");
// 输出空行
Console.WriteLine();
// 输出字符串 "YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE"
Console.WriteLine("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE");
// 输出字符串 "HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU"
Console.WriteLine("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU");
// 输出字符串 "GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!"
Console.WriteLine("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!");
// 输出字符串 "THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,"
Console.WriteLine("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,");
// 输出字符串 "IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS."
Console.WriteLine("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.");
// 输出空行
Console.WriteLine();

// 创建随机数生成器对象
Random rnd = new();

// 初始化变量 playAgain 和 totalWinnings
bool playAgain = false;
int totalWinnings = 0;

// 游戏循环
do
{
    // 生成 1 到 100 之间的随机数作为 jackpot
    int jackpot = rnd.Next(100) + 1; // [0..99] + 1 -> [1..100]
    int guess = 1;

    // 猜数循环
    while (true)
    {
        Console.WriteLine();
        // 读取用户输入的整数作为猜测的金额
        int amount = ReadInt("YOUR GUESS ");

        if (amount == jackpot)
        {
            // 如果猜中了，输出中奖金额并更新总奖金
            Console.WriteLine($"GOT IT!!!!!!!!!!   YOU WIN {jackpot} DOLLARS.");
            totalWinnings += jackpot;
            Console.WriteLine($"YOUR TOTAL WINNINGS ARE NOW {totalWinnings} DOLLARS.");
            break;
        }
        else if (amount > jackpot)
        {
            // 如果猜测金额过高，输出提示信息
            Console.WriteLine("YOUR GUESS IS TOO HIGH.");
        }
        else
        {
            // 如果猜测金额过低，输出提示信息
            Console.WriteLine("YOUR GUESS IS TOO LOW.");
        }

        guess++;
        if (guess > 6)
        {
            // 如果猜测次数超过 6 次，输出正确答案并结束游戏
            Console.WriteLine($"YOU BLEW IT...TOO BAD...THE NUMBER WAS {jackpot}");
            break;
        }
    }

    Console.WriteLine();
    // 询问用户是否再玩一次
    Console.Write("PLAY AGAIN (YES OR NO) ");
    playAgain = Console.ReadLine().ToUpper().StartsWith("Y");

} while (playAgain);

// 输出结束语
Console.WriteLine();
Console.WriteLine("SO LONG.  HOPE YOU ENJOYED YOURSELF!!!");

// Tab(n) 返回 n 个空格组成的字符串
static string Tab(int n) => new String(' ', n);
// 读取整数，提示用户输入一个数字
static int ReadInt(string question)
{
    // 循环直到用户输入有效的整数
    while (true)
    {
        // 提示用户输入问题
        Console.Write(question);
        // 读取用户输入并去除两端空格
        var input = Console.ReadLine().Trim();
        // 尝试将输入转换为整数，如果成功则返回该整数
        if (int.TryParse(input, out int value))
        {
            return value;
        }
        // 如果输入无法转换为整数，则提示用户输入无效
        Console.WriteLine("!Invalid Number Entered.");
    }
}
```