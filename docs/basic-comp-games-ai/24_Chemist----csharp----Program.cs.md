# `basic-computer-games\24_Chemist\csharp\Program.cs`

```

// 设置最大生命值为9
const int maxLives = 9;

// 居中打印文本
WriteCentred("Chemist");
WriteCentred("Creative Computing, Morristown, New Jersey");
Console.WriteLine(@"
...

The fictitious chemical kryptocyanic acid can only be
diluted by the ratio of 7 parts water to 3 parts acid.
If any other ratio is attempted, the acid becomes unstable
and soon explodes.  Given the amount of acid, you must
decide who much water to add for dilution.  If you miss
you face the consequences.
");

// 创建随机数生成器
var random = new Random();
int livesUsed = 0;
while (livesUsed < maxLives)
{
    // 生成1到50之间的随机数
    int krypto = random.Next(1, 50);
    // 根据比例计算所需的水量
    double water = krypto * 7.0 / 3.0;

    Console.WriteLine($"{krypto} Liters of kryptocyanic acid.  How much water?");
    // 读取用户输入的水量
    double answer = double.Parse(Console.ReadLine());

    // 计算用户输入水量与实际所需水量的差值
    double diff = Math.Abs(answer - water);
    // 判断用户输入的水量是否接近实际所需水量
    if (diff <= water / 20)
    {
        Console.WriteLine("Good job! You may breathe now, but don't inhale the fumes"!);
        Console.WriteLine();
    }
    else
    {
        Console.WriteLine("Sizzle!  You have just been desalinated into a blob\nof quivering protoplasm!");
        Console.WriteLine();
        livesUsed++;

        // 如果还有生命值，提示用户可以再次尝试
        if (livesUsed < maxLives)
            Console.WriteLine("However, you may try again with another life.");
    }
}
// 打印使用生命值的消息
Console.WriteLine($"Your {maxLives} lives are used, but you will be long remembered for\nyour contributions to the field of comic book chemistry.");

// 居中打印文本的方法
static void WriteCentred(string text)
{
    int indent = (Console.WindowWidth + text.Length) / 2;
    Console.WriteLine($"{{0,{indent}}}", text);
}

```