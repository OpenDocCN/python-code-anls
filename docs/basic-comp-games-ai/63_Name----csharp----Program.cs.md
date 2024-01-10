# `basic-computer-games\63_Name\csharp\Program.cs`

```
// 命名空间定义
namespace Name
{
    // 定义 Program 类
    public class Program
    {
        // Main 方法
        static void Main(string[] args)
        {
            // 输出字符串 "NAME" 居中对齐
            Console.WriteLine("NAME".CentreAlign());
            // 输出字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY" 居中对齐
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".CentreAlign());
            // 输出空行
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            // 输出字符串 "HELLO."
            Console.WriteLine("HELLO.");
            // 输出字符串 "MY NAME IS CREATIVE COMPUTER."
            Console.WriteLine("MY NAME IS CREATIVE COMPUTER.");
            // 输出提示字符串 "WHAT'S YOUR NAME (FIRST AND LAST? "
            Console.Write("WHAT'S YOUR NAME (FIRST AND LAST? ");
            // 读取用户输入的名字
            var name = Console.ReadLine();
            // 输出空行
            Console.WriteLine();
            // 输出格式化字符串，将用户输入的名字反转
            Console.WriteLine($"THANK YOU, {name.Reverse()}.");
            // 输出字符串 "OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART"
            Console.WriteLine("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART");
            // 输出字符串 "COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!"
            Console.WriteLine("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!");
            // 输出空行
            Console.WriteLine();
            // 输出字符串 "BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER."
            Console.WriteLine("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.");
            // 输出格式化字符串，将用户输入的名字排序
            Console.WriteLine($"LET'S PUT THEM IN ORDER LIKE THIS: {name.Sort()}");
            // 输出空行
            Console.WriteLine();
            // 输出提示字符串 "DON'T YOU LIKE THAT BETTER? "
            Console.Write("DON'T YOU LIKE THAT BETTER? ");
            // 读取用户输入的喜好
            var like = Console.ReadLine();
            // 输出空行
            Console.WriteLine();

            // 判断用户输入的喜好是否为 "YES"，不区分大小写
            if (like.ToUpperInvariant() == "YES")
            {
                // 输出字符串 "I KNEW YOU'D AGREE!!"
                Console.WriteLine("I KNEW YOU'D AGREE!!");
            }
            else
            {
                // 输出字符串 "I'M SORRY YOU DON'T LIKE IT THAT WAY."
                Console.WriteLine("I'M SORRY YOU DON'T LIKE IT THAT WAY.");
            }

            // 输出空行
            Console.WriteLine();
            // 输出格式化字符串，包含用户输入的名字
            Console.WriteLine($"I REALLY ENJOYED MEETING YOU {name}.");
            // 输出字符串 "HAVE A NICE DAY!"
            Console.WriteLine("HAVE A NICE DAY!");
        }
    }
}
```