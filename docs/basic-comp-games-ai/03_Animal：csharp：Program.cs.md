# `d:/src/tocomm/basic-computer-games\03_Animal\csharp\Program.cs`

```
# 导入必要的模块
import System
import Collections.Generic
import Linq

# 导入 Animal 模块
import Animal

# 打印 ANIMAL 字样
Console.WriteLine(new string(' ', 32) + "ANIMAL")
# 打印 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY 字样
Console.WriteLine(new string(' ', 15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
# 打印空行
Console.WriteLine()
Console.WriteLine()
Console.WriteLine()
# 打印 "PLAY 'GUESS THE ANIMAL'" 字样
Console.WriteLine("PLAY 'GUESS THE ANIMAL'")
# 打印空行
Console.WriteLine()
# 打印 "THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT." 字样
Console.WriteLine("THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT.")
# 打印空行
Console.WriteLine()

# 创建树的根节点
rootBranch = new Branch
{
    # 设置节点文本为 "DOES IT SWIM"
    Text = "DOES IT SWIM",
    Yes = new Branch { Text = "FISH" },  // 创建一个新的分支对象，文本为“FISH”
    No = new Branch { Text = "BIRD" }    // 创建一个新的分支对象，文本为“BIRD”
};

string[] TRUE_INPUTS = { "Y", "YES", "T", "TRUE" };  // 创建一个包含真值输入的字符串数组
string[] FALSE_INPUTS = { "N", "NO", "F", "FALSE" };  // 创建一个包含假值输入的字符串数组


while (true)  // 无限循环
{
    MainGameLoop();  // 调用主游戏循环函数
}

void MainGameLoop()
{
    // Wait fora YES or LIST command  // 等待输入“YES”或“LIST”命令
    string input = null;  // 初始化输入字符串为null
    while (true)  // 无限循环
    {
        input = GetInput("ARE YOU THINKING OF AN ANIMAL");  // 获取用户输入，提示为“ARE YOU THINKING OF AN ANIMAL”
        # 如果输入是列表命令，则列出已知动物
        if (IsInputListCommand(input)):
            ListKnownAnimals(rootBranch)
        # 如果输入是肯定回答，则跳出循环
        else if (IsInputYes(input)):
            break
    }

    // Walk through the tree following the YES and NO
    // branches based on user input.
    # 从根节点开始遍历树，根据用户输入跟随YES和NO分支
    Branch currentBranch = rootBranch
    while (!currentBranch.IsEnd):
        while (true):
            # 获取用户输入
            input = GetInput(currentBranch.Text)
            # 如果输入是肯定回答
            if (IsInputYes(input)):
                currentBranch = currentBranch.Yes;  # 设置当前分支为“是”分支
                break;  # 跳出循环
            }
            else if (IsInputNo(input))  # 如果输入为“否”
            {
                currentBranch = currentBranch.No;  # 设置当前分支为“否”分支
                break;  # 跳出循环
            }
        }
    }

    // 判断用户的回答是否正确
    input = GetInput($"IS IT A {currentBranch.Text}");  # 获取用户输入，询问是否是当前分支对应的动物
    if (IsInputYes(input))  # 如果用户回答是
    {
        Console.WriteLine("WHY NOT TRY ANOTHER ANIMAL?");  # 输出提示信息
        return;  # 结束程序
    }

    // 与用户交谈，添加新的问题和答案
    // 询问用户猜测的动物，并存储在newAnimal变量中
    string newAnimal = GetInput("THE ANIMAL YOU WERE THINKING OF WAS A");
    // 询问用户提出一个可以区分新动物和当前动物的问题，并存储在newQuestion变量中
    string newQuestion = GetInput($"PLEASE TYPE IN A QUESTION THAT WOULD DISTINGUISH A {newAnimal} FROM A {currentBranch.Text}");
    // 初始化newAnswer变量
    string newAnswer = null;
    // 循环直到用户输入有效的答案
    while (true)
    {
        // 询问用户新动物的答案，并存储在newAnswer变量中
        newAnswer = GetInput($"FOR A {newAnimal} THE ANSWER WOULD BE");
        // 如果答案是否定的，更新当前分支的No属性，并设置新的Yes和Text属性，然后跳出循环
        if (IsInputNo(newAnswer))
        {
            currentBranch.No = new Branch { Text = newAnimal };
            currentBranch.Yes = new Branch { Text = currentBranch.Text };
            currentBranch.Text = newQuestion;
            break;
        }
        // 如果答案是肯定的，更新当前分支的Yes属性，并设置新的No和Text属性，然后跳出循环
        else if (IsInputYes(newAnswer))
        {
            currentBranch.Yes = new Branch { Text = newAnimal };
            currentBranch.No = new Branch { Text = currentBranch.Text };
            currentBranch.Text = newQuestion;
            break;
string GetInput(string prompt)
{
    // 输出提示信息并获取用户输入
    Console.Write($"{prompt}? ");
    string result = Console.ReadLine();
    // 如果用户输入为空或者只包含空格，则递归调用 GetInput 函数重新获取输入
    if (string.IsNullOrWhiteSpace(result))
    {
        return GetInput(prompt);
    }
    // 返回去除空格并转换为大写的用户输入
    return result.Trim().ToUpper();
}

// 判断用户输入是否为肯定回答
bool IsInputYes(string input) => TRUE_INPUTS.Contains(input.ToUpperInvariant().Trim());

// 判断用户输入是否为否定回答
bool IsInputNo(string input) => FALSE_INPUTS.Contains(input.ToUpperInvariant().Trim());
# 检查输入是否为"LIST"命令
bool IsInputListCommand(string input) => input.ToUpperInvariant().Trim() == "LIST";

# 获取已知动物的数组
string[] GetKnownAnimals(Branch branch)
{
    List<string> result = new List<string>();
    # 如果分支已结束，返回包含分支文本的数组
    if (branch.IsEnd)
    {
        return new[] { branch.Text };
    }
    else
    {
        # 递归获取Yes分支和No分支的已知动物，并添加到结果列表中
        result.AddRange(GetKnownAnimals(branch.Yes));
        result.AddRange(GetKnownAnimals(branch.No));
        return result.ToArray();
    }
}

# 列出已知动物
void ListKnownAnimals(Branch branch)
{
    # 获取已知动物的数组
    string[] animals = GetKnownAnimals(branch);
    # 遍历动物列表
    for (int x = 0; x < animals.Length; x++)
    {
        # 计算当前动物在第几列
        int column = (x % 4);
        # 如果是新的一列，换行输出
        if (column == 0)
        {
            Console.WriteLine();
        }
        # 输出动物名，根据当前列数添加对应的空格
        Console.Write(new string(' ', column == 0 ? 0 : 15) + animals[x]);
    }
    # 输出换行，结束循环
    Console.WriteLine();
}
```