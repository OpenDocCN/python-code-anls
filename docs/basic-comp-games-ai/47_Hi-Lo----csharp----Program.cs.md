# `basic-computer-games\47_Hi-Lo\csharp\Program.cs`

```

// 输出带有34个空格的字符串，然后输出"HI LO"
Console.WriteLine(Tab(34) + "HI LO");
// 输出带有15个空格的字符串，然后输出"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
Console.WriteLine(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
// 输出空行
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();
// 输出以下内容
Console.WriteLine("THIS IS THE GAME OF HI LO.");
Console.WriteLine();
Console.WriteLine("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE");
Console.WriteLine("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU");
Console.WriteLine("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!");
Console.WriteLine("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,");
Console.WriteLine("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.");
Console.WriteLine();

// 创建一个随机数生成器对象
Random rnd = new();

// 初始化变量
bool playAgain = false;
int totalWinnings = 0;

// 游戏循环
do
{
    // 生成1到100之间的随机数作为奖池金额
    int jackpot = rnd.Next(100) + 1;
    int guess = 1;

    // 猜数循环
    while (true)
    {
        Console.WriteLine();
        // 从用户输入中读取一个整数
        int amount = ReadInt("YOUR GUESS ");

        if (amount == jackpot)
        {
            Console.WriteLine($"GOT IT!!!!!!!!!!   YOU WIN {jackpot} DOLLARS.");
            totalWinnings += jackpot;
            Console.WriteLine($"YOUR TOTAL WINNINGS ARE NOW {totalWinnings} DOLLARS.");
            break;
        }
        else if (amount > jackpot)
        {
            Console.WriteLine("YOUR GUESS IS TOO HIGH.");
        }
        else
        {
            Console.WriteLine("YOUR GUESS IS TOO LOW.");
        }

        guess++;
        if (guess > 6)
        {
            Console.WriteLine($"YOU BLEW IT...TOO BAD...THE NUMBER WAS {jackpot}");
            break;
        }
    }

    Console.WriteLine();
    Console.Write("PLAY AGAIN (YES OR NO) ");
    // 读取用户输入并判断是否继续游戏
    playAgain = Console.ReadLine().ToUpper().StartsWith("Y");

} while (playAgain);

Console.WriteLine();
Console.WriteLine("SO LONG.  HOPE YOU ENJOYED YOURSELF!!!");

// Tab(n) 返回n个空格组成的字符串
static string Tab(int n) => new String(' ', n);

// ReadInt 要求用户输入一个数字
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
    }
}

```