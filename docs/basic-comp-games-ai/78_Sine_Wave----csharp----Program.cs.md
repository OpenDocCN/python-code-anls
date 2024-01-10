# `basic-computer-games\78_Sine_Wave\csharp\Program.cs`

```
// 输出标题
Console.WriteLine(Tab(30) + "Sine Wave");
// 输出作者信息
Console.WriteLine(Tab(15) + "Creative Computing Morristown, New Jersey\n\n\n\n\n");

// 初始化变量 isCreative
bool isCreative = true;
// 循环计算正弦值并输出对应的单词
for (double t = 0.0; t <= 40.0; t += 0.25)
{
    // 计算单词输出位置
    int a = (int)(26 + 25 * Math.Sin(t));
    // 根据 isCreative 变量选择输出的单词
    string word = isCreative ? "Creative" : "Computing";
    // 输出单词
    Console.WriteLine($"{Tab(a)}{word}");
    // 切换 isCreative 变量的值
    isCreative = !isCreative;
}

// 定义 Tab 函数，用于生成指定数量的空格
static string Tab(int n) => new string(' ', n);
```