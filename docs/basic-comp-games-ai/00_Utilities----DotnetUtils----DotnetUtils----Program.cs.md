# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\Program.cs`

```py
// 引入所需的命名空间
using System.Xml.Linq;
using DotnetUtils;
using static System.Console;
using static System.IO.Path;
using static DotnetUtils.Methods;
using static DotnetUtils.Functions;

// 获取端口信息
var infos = PortInfos.Get;

// 定义包含动作和描述的元组数组
var actions = new (Action action, string description)[] {
    (printInfos, "Output information -- solution, project, and code files"),
    (missingSln, "Output missing sln"),
    (unexpectedSlnName, "Output misnamed sln"),
    (multipleSlns, "Output multiple sln files"),
    (missingProj, "Output missing project file"),
    (unexpectedProjName, "Output misnamed project files"),
    (multipleProjs, "Output multiple project files"),
    (checkProjects, "Check .csproj/.vbproj files for target framework, nullability etc."),
    (checkExecutableProject, "Check that there is at least one executable project per port"),
    (noCodeFiles, "Output ports without any code files"),
    (printPortInfo, "Print info about a single port"),
    (generateMissingSlns, "Generate solution files when missing"),
    (generateMissingProjs, "Generate project files when missing")
};

// 遍历动作数组并输出每个动作的描述和索引
foreach (var (_, description, index) in actions.WithIndex()) {
    WriteLine($"{index}: {description}");
}

WriteLine();

// 执行用户选择的动作
actions[getChoice(actions.Length - 1)].action();

// 输出端口信息中的解决方案文件
void printSlns(PortInfo pi) {
    switch (pi.Slns.Length) {
        case 0:
            WriteLine("No sln");
            break;
        case 1:
            WriteLine($"Solution: {pi.Slns[0].RelativePath(pi.LangPath)}");
            break;
        case > 1:
            WriteLine("Solutions:");
            foreach (var sln in pi.Slns) {
                Write(sln.RelativePath(pi.LangPath));
                WriteLine();
            }
            break;
    }
}

// 输出端口信息中的项目文件
void printProjs(PortInfo pi) {
    # 根据项目数量进行不同的处理
    switch (pi.Projs.Length) {
        # 如果项目数量为0，输出提示信息
        case 0:
            WriteLine("No project");
            break;
        # 如果项目数量为1，输出项目相对路径
        case 1:
            WriteLine($"Project: {pi.Projs[0].RelativePath(pi.LangPath)}");
            break;
        # 如果项目数量大于1，输出所有项目的相对路径
        case > 1:
            WriteLine("Projects:");
            # 遍历所有项目，输出每个项目的相对路径
            foreach (var proj in pi.Projs) {
                Write(proj.RelativePath(pi.LangPath));
                WriteLine();
            }
            break;
    }
// 打印所有信息
void printInfos() {
    // 遍历infos列表中的每个元素
    foreach (var item in infos) {
        // 打印item的LangPath
        WriteLine(item.LangPath);
        // 打印空行
        WriteLine();

        // 调用printSlns函数打印item的解决方案
        printSlns(item);
        // 打印空行
        WriteLine();

        // 调用printProjs函数打印item的项目
        printProjs(item);
        // 打印空行
        WriteLine();

        // 获取代码文件
        foreach (var file in item.CodeFiles) {
            // 打印文件相对于LangPath的相对路径
            WriteLine(file.RelativePath(item.LangPath));
        }
        // 打印一行由50个'-'组成的分隔线
        WriteLine(new string('-', 50));
    }
}

// 打印缺失的解决方案
void missingSln() {
    // 选择infos中没有解决方案的元素，转换为数组
    var data = infos.Where(x => x.Slns.None()).ToArray();
    // 遍历data数组中的每个元素
    foreach (var item in data) {
        // 打印item的LangPath
        WriteLine(item.LangPath);
    }
    // 打印空行
    WriteLine();
    // 打印data数组的长度
    WriteLine($"Count: {data.Length}");
}

// 打印意外的解决方案名称
void unexpectedSlnName() {
    // 计数器初始化为0
    var counter = 0;
    // 遍历infos列表中的每个元素
    foreach (var item in infos) {
        // 如果item没有解决方案，则继续下一次循环
        if (item.Slns.None()) { continue; }

        // 期望的解决方案名称
        var expectedSlnName = $"{item.GameName}.sln";
        // 如果item的解决方案不包含期望的解决方案名称，则继续下一次循环
        if (item.Slns.Contains(Combine(item.LangPath, expectedSlnName), StringComparer.InvariantCultureIgnoreCase)) { continue; }

        // 计数器加1
        counter += 1;
        // 打印item的LangPath
        WriteLine(item.LangPath);
        // 打印"Expected: 期望的解决方案名称"
        WriteLine($"Expected: {expectedSlnName}");

        // 调用printSlns函数打印item的解决方案
        printSlns(item);

        // 打印空行
        WriteLine();
    }
    // 打印"Count: 计数器的值"
    WriteLine($"Count: {counter}");
}

// 打印多个解决方案
void multipleSlns() {
    // 选择infos中解决方案数量大于1的元素，转换为数组
    var data = infos.Where(x => x.Slns.Length > 1).ToArray();
    // 遍历data数组中的每个元素
    foreach (var item in data) {
        // 打印item的LangPath
        WriteLine(item.LangPath);
        // 调用printSlns函数打印item的解决方案
        printSlns(item);
    }
    // 打印空行
    WriteLine();
    // 打印"Count: data数组的长度"
    WriteLine($"Count: {data.Length}");
}

// 打印缺失的项目
void missingProj() {
    // 选择infos中没有项目的元素，转换为数组
    var data = infos.Where(x => x.Projs.None()).ToArray();
    // 遍历data数组中的每个元素
    foreach (var item in data) {
        // 打印item的LangPath
        WriteLine(item.LangPath);
    }
    // 打印空行
    WriteLine();
    // 打印"Count: data数组的长度"
    WriteLine($"Count: {data.Length}");
}

// 打印意外的项目名称
void unexpectedProjName() {
    // 计数器初始化为0
    var counter = 0;
    # 遍历infos列表中的每个元素
    foreach (var item in infos) {
        # 如果item中的Projs为空，则跳过当前循环，继续下一个元素
        if (item.Projs.None()) { continue; }

        # 构建预期的项目名称，格式为"{item.GameName}.{item.ProjExt}"
        var expectedProjName = $"{item.GameName}.{item.ProjExt}";
        # 如果item中的Projs包含预期的项目名称，则跳过当前循环，继续下一个元素
        if (item.Projs.Contains(Combine(item.LangPath, expectedProjName))) { continue; }

        # 计数器加1
        counter += 1;
        # 输出item的LangPath
        WriteLine(item.LangPath);
        # 输出预期的项目名称
        WriteLine($"Expected: {expectedProjName}");

        # 调用printProjs函数，输出item的项目信息
        printProjs(item);

        # 输出空行
        WriteLine();
    }
    # 输出计数器的值
    WriteLine($"Count: {counter}");
}

// 处理多个项目的情况
void multipleProjs() {
    // 从 infos 中筛选出包含多个项目的数据，并转换为数组
    var data = infos.Where(x => x.Projs.Length > 1).ToArray();
    // 遍历数据数组
    foreach (var item in data) {
        // 输出语言路径
        WriteLine(item.LangPath);
        // 输出空行
        WriteLine();
        // 调用 printProjs 方法打印项目信息
        printProjs(item);
    }
    // 输出空行
    WriteLine();
    // 输出数据数组的长度
    WriteLine($"Count: {data.Length}");
}

// 生成缺失的解决方案文件
void generateMissingSlns() {
    // 遍历 infos 中不包含解决方案文件的数据
    foreach (var item in infos.Where(x => x.Slns.None())) {
        // 运行 dotnet 命令创建解决方案文件
        var result = RunProcess("dotnet", $"new sln -n {item.GameName} -o {item.LangPath}");
        // 输出结果
        WriteLine(result);

        // 获取解决方案文件的完整路径
        var slnFullPath = Combine(item.LangPath, $"{item.GameName}.sln");
        // 遍历项目列表，将项目添加到解决方案文件中
        foreach (var proj in item.Projs) {
            result = RunProcess("dotnet", $"sln {slnFullPath} add {proj}");
            // 输出结果
            WriteLine(result);
        }
    }
}

// 生成缺失的项目文件
void generateMissingProjs() {
    // 遍历 infos 中不包含项目文件的数据
    foreach (var item in infos.Where(x => x.Projs.None())) {
        // 根据语言类型生成项目文件的文本内容
        var projText = item.Lang switch {
            "csharp" => @"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
    <LangVersion>10</LangVersion>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
",
            "vbnet" => @$"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <RootNamespace>{item.GameName}</RootNamespace>
    <TargetFramework>net6.0</TargetFramework>
    <LangVersion>16.9</LangVersion>
  </PropertyGroup>
</Project>
        // 使用 switch 语句根据项目类型执行不同的操作
        switch (info.Lang) {
            // 如果项目类型为 CSharp，则生成 .csproj 文件
            case "CSharp":
                var projText = GenerateCsProj(info);
                break;
            // 如果项目类型为 FSharp，则生成 .fsproj 文件
            case "FSharp":
                var projText = GenerateFsProj(info);
                break;
            // 如果项目类型为其他，则抛出异常
            _ => throw new InvalidOperationException()
        };
        // 将项目路径和项目文件扩展名组合成完整的项目文件路径
        var projFullPath = Combine(item.LangPath, $"{item.GameName}.{item.ProjExt}");
        // 将生成的项目文件内容写入到项目文件路径
        File.WriteAllText(projFullPath, projText);

        // 如果只有一个解决方案文件
        if (item.Slns.Length == 1) {
            // 使用 dotnet 命令将项目文件添加到解决方案中
            var result = RunProcess("dotnet", $"sln {item.Slns[0]} add {projFullPath}");
            // 输出结果
            WriteLine(result);
        }
    }
}

// 检查项目
void checkProjects() {
    // 遍历项目信息列表
    foreach (var info in infos) {
        // 输出语言路径
        WriteLine(info.LangPath);
        // 打印项目警告
        printProjectWarnings(info);
    }
}

// 打印项目警告
void printProjectWarnings(PortInfo info) {
    // 遍历 info.Projs 中的每个项目
    foreach (var proj in info.Projs) {
        // 创建一个字符串列表用于存储警告信息
        var warnings = new List<string>();
        // 加载项目文件并获取根元素 "Project" 下的 "PropertyGroup" 元素
        var parent = XDocument.Load(proj).Element("Project")?.Element("PropertyGroup");

        // 使用解构赋值获取属性值
        var (
            framework,
            nullable,
            implicitUsing,
            rootNamespace,
            langVersion,
            optionStrict
        ) = (
            getValue(parent, "TargetFramework", "TargetFrameworks"),
            getValue(parent, "Nullable"),
            getValue(parent, "ImplicitUsings"),
            getValue(parent, "RootNamespace"),
            getValue(parent, "LangVersion"),
            getValue(parent, "OptionStrict")
        );

        // 如果 framework 不是 "net6.0"，则添加警告信息
        if (framework != "net6.0") {
            warnings.Add($"Target: {framework}");
        }

        // 如果 info.Lang 是 "csharp"，则检查 C# 相关属性并添加警告信息
        if (info.Lang == "csharp") {
            if (nullable != "enable") {
                warnings.Add($"Nullable: {nullable}");
            }
            if (implicitUsing != "enable") {
                warnings.Add($"ImplicitUsings: {implicitUsing}");
            }
            if (rootNamespace != null && rootNamespace != info.GameName) {
                warnings.Add($"RootNamespace: {rootNamespace}");
            }
            if (langVersion != "10") {
                warnings.Add($"LangVersion: {langVersion}");
            }
        }

        // 如果 info.Lang 是 "vbnet"，则检查 VB.NET 相关属性并添加警告信息
        if (info.Lang == "vbnet") {
            if (rootNamespace != info.GameName) {
                warnings.Add($"RootNamespace: {rootNamespace}");
            }
            if (langVersion != "16.9") {
                warnings.Add($"LangVersion: {langVersion}");
            }
            if (optionStrict != "On") {
                warnings.Add($"OptionStrict: {optionStrict}");
            }
        }

        // 如果存在警告信息，则输出项目相对路径和警告信息
        if (warnings.Any()) {
            WriteLine(proj.RelativePath(info.LangPath));
            WriteLine(string.Join("\n", warnings));
            WriteLine();
        }
    }
// 检查可执行项目
void checkExecutableProject() {
    // 遍历项目信息列表
    foreach (var item in infos) {
        // 如果所有项目的输出类型不是 "Exe"
        if (item.Projs.All(proj => getValue(proj, "OutputType") != "Exe")) {
            // 输出语言路径
            WriteLine($"{item.LangPath}");
        }
    }
}

// 没有代码文件
void noCodeFiles() {
    // 查询没有代码文件的项目信息，按语言排序
    var qry = infos
        .Where(x => x.CodeFiles.None())
        .OrderBy(x => x.Lang);
    // 遍历查询结果
    foreach (var item in qry) {
        // 输出语言路径
        WriteLine(item.LangPath);
    }
}

// 尝试构建
void tryBuild() {
    // 如果有代码文件，尝试构建
}

// 打印端口信息
void printPortInfo() {
    // 提示输入端口号
    Write("Enter number from 1 to 96 ");
    // 获取用户输入的选择
    var index = getChoice(1, 96);

    Write("Enter 0 for C#, 1 for VB ");
    // 获取用户输入的语言选择
    var lang = getChoice(1) switch {
        0 => "csharp",
        1 => "vbnet",
        _ => throw new InvalidOperationException()
    };

    WriteLine();

    // 获取指定索引和语言的项目信息
    var info = infos.Single(x => x.Index == index && x.Lang == lang);

    WriteLine(info.LangPath);
    WriteLine(new string('-', 50));

    // 打印解决方案
    printSlns(info);

    // 解决方案名称/位置不匹配？（预期 x）
    var expectedSlnName = Combine(info.LangPath, $"{info.GameName}.sln");
    if (!info.Slns.Contains(expectedSlnName)) {
        WriteLine($"Expected name/path: {expectedSlnName.RelativePath(info.LangPath)}");
    }

    // 是否有可执行项目？
    if (info.Projs.All(proj => getValue(proj, "OutputType") != "Exe")) {
        WriteLine("No executable project");
    }

    WriteLine();

    // 打印项目
    printProjs(info);

    // 项目名称/位置不匹配？（预期 x）
    var expectedProjName = Combine(info.LangPath, $"{info.GameName}.{info.ProjExt}");
    if (info.Projs.Length < 2 && !info.Projs.Contains(expectedProjName)) {
        WriteLine($"Expected name/path: {expectedProjName.RelativePath(info.LangPath)}");
    }

    WriteLine();

    // 验证项目属性
    printProjectWarnings(info);

    WriteLine("Code files:");

    // 列出代码文件
}
    # 遍历 info 对象中的 CodeFiles 列表
    foreach (var codeFile in info.CodeFiles) {
        # 输出每个 codeFile 对象的相对路径
        WriteLine(codeFile.RelativePath(info.LangPath));
    }

    # 尝试构建代码
    // try build
# 闭合前面的函数定义
```