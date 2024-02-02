# `basic-computer-games\24_Chemist\csharp\Program.cs`

```py
// 定义最大生命值
const int maxLives = 9;

// 居中打印标题和作者信息
WriteCentred("Chemist");
WriteCentred("Creative Computing, Morristown, New Jersey");
Console.WriteLine(@"
The fictitious chemical kryptocyanic acid can only be
diluted by the ratio of 7 parts water to 3 parts acid.
If any other ratio is attempted, the acid becomes unstable
and soon explodes.  Given the amount of acid, you must
decide who much water to add for dilution.  If you miss
you face the consequences.
");

// 创建随机数生成器和记录使用生命的次数
var random = new Random();
int livesUsed = 0;

// 循环进行游戏，直到生命用尽
while (livesUsed < maxLives)
{
    // 生成随机的 krypto 酸量，并计算所需的水量
    int krypto = random.Next(1, 50);
    double water = krypto * 7.0 / 3.0;

    // 提示用户输入水量，并计算用户输入与正确水量的差值
    Console.WriteLine($"{krypto} Liters of kryptocyanic acid.  How much water?");
    double answer = double.Parse(Console.ReadLine());
    double diff = Math.Abs(answer - water);

    // 根据差值判断用户是否成功
    if (diff <= water / 20)
    {
        Console.WriteLine("Good job! You may breathe now, but don't inhale the fumes!");
        Console.WriteLine();
    }
    else
    {
        Console.WriteLine("Sizzle!  You have just been desalinated into a blob\nof quivering protoplasm!");
        Console.WriteLine();
        livesUsed++;

        // 如果生命还有剩余，提示用户可以再试一次
        if (livesUsed < maxLives)
            Console.WriteLine("However, you may try again with another life.");
    }
}

// 打印游戏结束信息
Console.WriteLine($"Your {maxLives} lives are used, but you will be long remembered for\nyour contributions to the field of comic book chemistry.");

// 定义一个函数，用于居中打印文本
static void WriteCentred(string text)
{
    int indent = (Console.WindowWidth + text.Length) / 2;
    Console.WriteLine($"{{0,{indent}}}", text);
}
```