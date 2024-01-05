# `78_Sine_Wave\csharp\Program.cs`

```
// 输出"Sine Wave"标题
Console.WriteLine(Tab(30) + "Sine Wave");
// 输出"Creative Computing Morristown, New Jersey"信息
Console.WriteLine(Tab(15) + "Creative Computing Morristown, New Jersey\n\n\n\n\n");

// 初始化变量isCreative为true
bool isCreative = true;
// 循环计算正弦值，并输出对应的单词
for (double t = 0.0; t <= 40.0; t += 0.25)
{
    // 计算a的值，用于确定单词输出的位置
    int a = (int)(26 + 25 * Math.Sin(t));
    // 根据isCreative的值选择输出的单词
    string word = isCreative ? "Creative" : "Computing";
    // 输出单词，并根据isCreative的值切换下一个单词
    Console.WriteLine($"{Tab(a)}{word}");
    isCreative = !isCreative;
}

// 定义一个函数Tab，用于生成指定数量的空格
static string Tab(int n) => new string(' ', n);
```