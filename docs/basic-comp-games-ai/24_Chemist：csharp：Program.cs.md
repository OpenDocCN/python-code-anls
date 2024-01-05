# `24_Chemist\csharp\Program.cs`

```
// 常量定义，表示最大生命值
const int maxLives = 9;

// 在屏幕上居中打印文本
WriteCentred("Chemist");
WriteCentred("Creative Computing, Morristown, New Jersey");

// 在控制台打印多行文本
Console.WriteLine(@"
The fictitious chemical kryptocyanic acid can only be
diluted by the ratio of 7 parts water to 3 parts acid.
If any other ratio is attempted, the acid becomes unstable
and soon explodes.  Given the amount of acid, you must
decide who much water to add for dilution.  If you miss
you face the consequences.
");

// 创建一个随机数生成器对象
var random = new Random();
// 初始化已使用生命值的变量
int livesUsed = 0;
// 循环，直到使用的生命值达到最大生命值
while (livesUsed < maxLives)
{
    int krypto = random.Next(1, 50);  // 生成一个1到50之间的随机整数赋值给变量krypto
    double water = krypto * 7.0 / 3.0;  // 使用krypto计算水的量，赋值给变量water

    Console.WriteLine($"{krypto} Liters of kryptocyanic acid.  How much water?");  // 输出krypto的值和提示信息，要求输入水的量
    double answer = double.Parse(Console.ReadLine());  // 读取用户输入的水的量，转换为double类型赋值给变量answer

    double diff = Math.Abs(answer - water);  // 计算用户输入的水量和实际水量之间的差值，赋值给变量diff
    if (diff <= water / 20)  // 如果差值小于等于实际水量的1/20
    {
        Console.WriteLine("Good job! You may breathe now, but don't inhale the fumes!");  // 输出恭喜信息
        Console.WriteLine();  // 输出空行
    }
    else  // 如果差值大于实际水量的1/20
    {
        Console.WriteLine("Sizzle!  You have just been desalinated into a blob\nof quivering protoplasm!");  // 输出失败信息
        Console.WriteLine();  // 输出空行
        livesUsed++;  // 生命值减一

        if (livesUsed < maxLives)  // 如果生命值小于最大生命值
            Console.WriteLine("However, you may try again with another life.");  // 输出提示信息
    }
}
Console.WriteLine($"Your {maxLives} lives are used, but you will be long remembered for\nyour contributions to the field of comic book chemistry.");
// 打印消息，使用字符串插值将变量 maxLives 的值插入到字符串中

static void WriteCentred(string text)
{
    int indent = (Console.WindowWidth + text.Length) / 2;
    // 计算文本居中时的缩进量
    Console.WriteLine($"{{0,{indent}}}", text);
    // 打印居中对齐的文本
}
```