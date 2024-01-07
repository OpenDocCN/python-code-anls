# `basic-computer-games\03_Animal\csharp\Program.cs`

```

// 引入必要的命名空间
using System;
using System.Collections.Generic;
using System.Linq;

// 引入 Animal 命名空间中的类

// 输出 ANIMAL 的标题
Console.WriteLine(new string(' ', 32) + "ANIMAL");
// 输出创意计算公司的信息
Console.WriteLine(new string(' ', 15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
// 输出空行
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();
// 输出游戏的标题和介绍
Console.WriteLine("PLAY 'GUESS THE ANIMAL'");
Console.WriteLine();
Console.WriteLine("THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT.");
Console.WriteLine();

// 创建根节点的问题和答案树
Branch rootBranch = new Branch
{
    Text = "DOES IT SWIM",
    Yes = new Branch { Text = "FISH" },
    No = new Branch { Text = "BIRD" }
};

// 定义输入为真的可能值
string[] TRUE_INPUTS = { "Y", "YES", "T", "TRUE" };
// 定义输入为假的可能值
string[] FALSE_INPUTS = { "N", "NO", "F", "FALSE" };

// 游戏主循环
while (true)
{
    MainGameLoop();
}

// 游戏主循环函数
void MainGameLoop()
}

// 获取用户输入
string GetInput(string prompt)
{
    Console.Write($"{prompt}? ");
    string result = Console.ReadLine();
    // 如果输入为空，则重新获取输入
    if (string.IsNullOrWhiteSpace(result))
    {
        return GetInput(prompt);
    }

    return result.Trim().ToUpper();
}

// 判断输入是否为真
bool IsInputYes(string input) => TRUE_INPUTS.Contains(input.ToUpperInvariant().Trim());

// 判断输入是否为假
bool IsInputNo(string input) => FALSE_INPUTS.Contains(input.ToUpperInvariant().Trim());

// 判断输入是否为列表命令
bool IsInputListCommand(string input) => input.ToUpperInvariant().Trim() == "LIST";

// 获取已知动物的列表
string[] GetKnownAnimals(Branch branch)
{
    List<string> result = new List<string>();
    if (branch.IsEnd)
    {
        return new[] { branch.Text };
    }
    else
    {
        result.AddRange(GetKnownAnimals(branch.Yes));
        result.AddRange(GetKnownAnimals(branch.No));
        return result.ToArray();
    }
}

// 列出已知动物
void ListKnownAnimals(Branch branch)
{
    string[] animals = GetKnownAnimals(branch);
    for (int x = 0; x < animals.Length; x++)
    {
        int column = (x % 4);
        if (column == 0)
        {
            Console.WriteLine();
        }

        Console.Write(new string(' ', column == 0 ? 0 : 15) + animals[x]);
    }
    Console.WriteLine();
}

```