# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\Program.cs`

```

// 导入所需的命名空间
using System.Xml.Linq;
using DotnetUtils;
using static System.Console;
using static System.IO.Path;
using static DotnetUtils.Methods;
using static DotnetUtils.Functions;

// 获取端口信息
var infos = PortInfos.Get;

// 定义操作和描述的元组数组
var actions = new (Action action, string description)[] {
    // ... (后续操作和描述)
};

// 遍历操作数组并输出描述
foreach (var (_, description, index) in actions.WithIndex()) {
    WriteLine($"{index}: {description}");
}

// 选择并执行操作
actions[getChoice(actions.Length - 1)].action();

// 定义打印解决方案的方法
void printSlns(PortInfo pi) {
    // ... (后续操作)
}

// 定义打印项目的方法
void printProjs(PortInfo pi) {
    // ... (后续操作)
}

// 打印信息
void printInfos() {
    // ... (后续操作)
}

// 检查项目
void checkProjects() {
    // ... (后续操作)
}

// 检查可执行项目
void checkExecutableProject() {
    // ... (后续操作)
}

// 无代码文件
void noCodeFiles() {
    // ... (后续操作)
}

// 尝试构建
void tryBuild() {
    // ... (后续操作)
}

// 打印端口信息
void printPortInfo() {
    // ... (后续操作)
}

```