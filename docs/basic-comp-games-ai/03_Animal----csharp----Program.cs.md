# `basic-computer-games\03_Animal\csharp\Program.cs`

```py
// 引入必要的命名空间
using System;
using System.Collections.Generic;
using System.Linq;

// 引入 Animal 命名空间中的类

// 输出标题和介绍信息
Console.WriteLine(new string(' ', 32) + "ANIMAL");
Console.WriteLine(new string(' ', 15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();
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

// 定义用户输入的肯定和否定的可能值
string[] TRUE_INPUTS = { "Y", "YES", "T", "TRUE" };
string[] FALSE_INPUTS = { "N", "NO", "F", "FALSE" };

// 游戏主循环
while (true)
{
    MainGameLoop();
}

// 游戏主循环函数
void MainGameLoop()
{
    // 等待用户输入肯定或列出已知动物的命令
    string input = null;
    while (true)
    {
        input = GetInput("ARE YOU THINKING OF AN ANIMAL");
        if (IsInputListCommand(input))
        {
            ListKnownAnimals(rootBranch);
        }
        else if (IsInputYes(input))
        {
            break;
        }
    }

    // 根据用户输入遍历问题和答案树
    Branch currentBranch = rootBranch;
    while (!currentBranch.IsEnd)
    {
        while (true)
        {
            input = GetInput(currentBranch.Text);
            if (IsInputYes(input))
            {
                currentBranch = currentBranch.Yes;
                break;
            }
            else if (IsInputNo(input))
            {
                currentBranch = currentBranch.No;
                break;
            }
        }
    }

    // 判断计算机猜测的动物是否正确
    input = GetInput($"IS IT A {currentBranch.Text}");
    if (IsInputYes(input))
    {
        Console.WriteLine("WHY NOT TRY ANOTHER ANIMAL?");
        return;
    }

    // 询问用户以添加新的问题和答案分支到树中
}
    # 获取用户输入的新动物名称
    string newAnimal = GetInput("THE ANIMAL YOU WERE THINKING OF WAS A");
    # 获取用户输入的新问题，用于区分新动物和当前分支的动物
    string newQuestion = GetInput($"PLEASE TYPE IN A QUESTION THAT WOULD DISTINGUISH A {newAnimal} FROM A {currentBranch.Text}");
    # 初始化新答案为null
    string newAnswer = null;
    # 循环直到用户输入有效的答案
    while (true)
    {
        # 获取用户输入的新答案
        newAnswer = GetInput($"FOR A {newAnimal} THE ANSWER WOULD BE");
        # 如果新答案是否定的
        if (IsInputNo(newAnswer))
        {
            # 设置当前分支的否定分支为新动物，肯定分支为当前分支的动物，文本为新问题
            currentBranch.No = new Branch { Text = newAnimal };
            currentBranch.Yes = new Branch { Text = currentBranch.Text };
            currentBranch.Text = newQuestion;
            # 退出循环
            break;
        }
        # 如果新答案是肯定的
        else if (IsInputYes(newAnswer))
        {
            # 设置当前分支的肯定分支为新动物，否定分支为当前分支的动物，文本为新问题
            currentBranch.Yes = new Branch { Text = newAnimal };
            currentBranch.No = new Branch { Text = currentBranch.Text };
            currentBranch.Text = newQuestion;
            # 退出循环
            break;
        }
    }
# 获取用户输入的字符串，并显示提示信息
string GetInput(string prompt)
{
    Console.Write($"{prompt}? ");
    string result = Console.ReadLine();
    # 如果输入为空或者只包含空格，则递归调用 GetInput 函数重新获取输入
    if (string.IsNullOrWhiteSpace(result))
    {
        return GetInput(prompt);
    }
    # 返回去除首尾空格并转换为大写的结果
    return result.Trim().ToUpper();
}

# 判断输入是否为肯定回答
bool IsInputYes(string input) => TRUE_INPUTS.Contains(input.ToUpperInvariant().Trim());

# 判断输入是否为否定回答
bool IsInputNo(string input) => FALSE_INPUTS.Contains(input.ToUpperInvariant().Trim());

# 判断输入是否为列表命令
bool IsInputListCommand(string input) => input.ToUpperInvariant().Trim() == "LIST";

# 获取给定分支下的所有已知动物
string[] GetKnownAnimals(Branch branch)
{
    List<string> result = new List<string>();
    # 如果分支为结束节点，则返回包含该节点文本的数组
    if (branch.IsEnd)
    {
        return new[] { branch.Text };
    }
    else
    {
        # 递归获取 Yes 和 No 分支下的已知动物，并合并到结果列表中
        result.AddRange(GetKnownAnimals(branch.Yes));
        result.AddRange(GetKnownAnimals(branch.No));
        return result.ToArray();
    }
}

# 列出给定分支下的所有已知动物
void ListKnownAnimals(Branch branch)
{
    # 获取给定分支下的所有已知动物
    string[] animals = GetKnownAnimals(branch);
    # 遍历已知动物数组
    for (int x = 0; x < animals.Length; x++)
    {
        int column = (x % 4);
        # 如果是新的一行，则输出换行符
        if (column == 0)
        {
            Console.WriteLine();
        }
        # 输出已知动物，根据列数添加相应的空格
        Console.Write(new string(' ', column == 0 ? 0 : 15) + animals[x]);
    }
    # 输出换行符
    Console.WriteLine();
}
```