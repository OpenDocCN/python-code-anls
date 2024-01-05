# `d:/src/tocomm/basic-computer-games\19_Bunny\csharp\Bunny.cs`

```
namespace Bunny
{
    internal class Bunny
    {
        private const int asciiBase = 64;  // 定义一个常量，表示ASCII码的基数

        private readonly int[] bunnyData = {  // 定义一个只读的整型数组，存储一系列整数数据
            2,21,14,14,25,  // 数据项
            1,2,-1,0,2,45,50,-1,0,5,43,52,-1,0,7,41,52,-1,  // 数据项
            // ... （省略部分数据项）
        };
    }
}
22,23,26,29,-1,27,29,-1,28,29,-1,4096
```
这是一个数组的初始化，包含一些整数值。

```
public void Run()
```
这是一个公共方法的声明，名称为Run。

```
PrintString(33, "BUNNY");
```
调用PrintString方法，传入两个参数：33和"BUNNY"。

```
PrintString(15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
```
再次调用PrintString方法，传入两个参数：15和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"。

```
PrintLines(3);
```
调用PrintLines方法，传入一个参数：3。

```
BasicData data = new (bunnyData);
```
创建一个BasicData对象，使用bunnyData作为参数。

```
var a = new char[5];
```
创建一个包含5个元素的字符数组a。

```
for (var i = 0; i < 5; ++i)
```
开始一个for循环，循环条件是i小于5。

```
a[i] = (char)(asciiBase + data.Read());
```
将data.Read()的返回值与asciiBase相加，然后将结果转换为字符并赋值给数组a的第i个元素。
            }
            PrintLines(6);  # 调用自定义函数PrintLines，打印6行空白

            PrintLines(1);  # 调用自定义函数PrintLines，打印1行空白
            var col = 0;  # 初始化变量col为0
            while (true):  # 进入无限循环
                var x = data.Read();  # 从数据中读取一个值赋给变量x
                if (x < 0):  # 如果x小于0
                    PrintLines(1);  # 调用自定义函数PrintLines，打印1行空白
                    col = 0;  # 将col重置为0
                    continue;  # 继续下一次循环
                if (x > 128) break;  # 如果x大于128，跳出循环，结束处理
                col += PrintSpaces(x - col);  # 调用自定义函数PrintSpaces，打印x-col个空格，并将结果加到col上
                var y = data.Read();  # 从数据中读取一个值赋给变量y
                for (var i = x; i <= y; ++i):  # 循环i从x到y
                    # var j = i - 5 * (i / 5);  # 计算j的值，BASIC语言没有模运算符
                    // 在控制台输出数组a中第i%5个元素的值
                    Console.Write(a[i % 5]);
                    // 控制台输出数组a中第col%5个元素的值，这种写法也是可以的
                    // Console.Write(a[col % 5]); // This works, too
                    // col自增1
                    ++col;
                }
            }
            // 调用PrintLines方法，在控制台输出6个空行
            PrintLines(6);
        }
        // 定义一个方法，用于在控制台输出指定数量的空行
        private static void PrintLines(int count)
        {
            for (var i = 0; i < count; ++i)
                // 在控制台输出空行
                Console.WriteLine();
        }
        // 定义一个方法，用于在控制台输出指定数量的空格，并返回输出的空格数量
        private static int PrintSpaces(int count)
        {
            for (var i = 0; i < count; ++i)
                // 在控制台输出空格
                Console.Write(' ');
            // 返回输出的空格数量
            return count;
        }
        // 定义一个方法，用于在控制台输出指定缩进量和字符串值，并可选择是否换行
        public static void PrintString(int tab, string value, bool newLine = true)
# 打印指定数量的空格
PrintSpaces(tab);
# 在控制台上输出指定的值
Console.Write(value);
# 如果需要换行，则在控制台上输出一个换行符
if (newLine) Console.WriteLine();
```