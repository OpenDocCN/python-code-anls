# BasicComputerGames源码解析 9

# `00_Utilities/DotnetUtils/DotnetUtils/PortInfo.cs`

This code appears to be a .NET class that contains methods for creating ports for various games. The class is derived from `PortInfo`, which is part of the `SimG育碧` library.

The class has a `Create` method, which takes two arguments: `gamePath` and `langKeyword`. The `gamePath` argument is the path to the game's root directory, and the `langKeyword` argument is a keywords that identifies the language of the game.

The method returns a `PortInfo` object, which contains information about the port, such as its index, the game name, and the language code.

The class also has several methods for getting file names and paths related to the games, such as `GetFileName`, `GetGameName`, `GetLangDataPath`, and `GetFiles`. These methods are not shown in the code provided, but they are called by the `Create` method in the `PortInfo` class.


```
﻿using static System.IO.Directory;
using static System.IO.Path;
using static DotnetUtils.Globals;

namespace DotnetUtils;

public record PortInfo(
    string GamePath, string FolderName, int Index, string GameName,
    string LangPath, string Lang, string Ext, string ProjExt,
    string[] CodeFiles, string[] Slns, string[] Projs
) {

    private static readonly EnumerationOptions enumerationOptions = new() {
        RecurseSubdirectories = true,
        MatchType = MatchType.Simple,
        MatchCasing = MatchCasing.CaseInsensitive
    };

    // .NET namespaces cannot have a digit as the first character
    // For games whose name starts with a digit, we map the name to a specific string
    private static readonly Dictionary<string, string> specialGameNames = new() {
        { "3-D_Plot", "Plot" },
        { "3-D_Tic-Tac-Toe", "ThreeDTicTacToe" },
        { "23_Matches", "TwentyThreeMatches"}
    };

    public static PortInfo? Create(string gamePath, string langKeyword) {
        var folderName = GetFileName(gamePath);
        var parts = folderName.Split('_', 2);

        if (parts.Length <= 1) { return null; }

        var (index, gameName) = (
            int.TryParse(parts[0], out var n) && n > 0 ? // ignore utilities folder
                n :
                (int?)null,
            specialGameNames.TryGetValue(parts[1], out var specialName) ?
                specialName :
                parts[1].Replace("_", "").Replace("-", "")
        );

        if (index is null || gameName is null) { return null; }

        var (ext, projExt) = LangData[langKeyword];
        var langPath = Combine(gamePath, langKeyword);
        var codeFiles =
            GetFiles(langPath, $"*.{ext}", enumerationOptions)
                .Where(x => !x.Contains("\\bin\\") && !x.Contains("\\obj\\"))
                .ToArray();

        return new PortInfo(
            gamePath, folderName, index.Value, gameName,
            langPath, langKeyword, ext, projExt,
            codeFiles,
            GetFiles(langPath, "*.sln", enumerationOptions),
            GetFiles(langPath, $"*.{projExt}", enumerationOptions)
        );
    }
}

```

# `00_Utilities/DotnetUtils/DotnetUtils/PortInfos.cs`

该代码是一个名为 `PortInfos` 的类，其目的是提供对游戏中可寻物(如游戏中的任务、目标、奖励等)的文本信息(即 "Portinfos")的访问。

以下是代码的功能解释：

1. 导入 `System.Reflection`、`System.IO.Directory` 和 `DotnetUtils.Globals` 命名空间。

2. 定义一个名为 `Root` 的字符串变量，用于存储游戏根目录的路径。然后，通过调用 `GetParent` 方法获取游戏根目录的父目录，并将其存储在 `Root` 变量中。接着，通过 `IndexOf` 方法获取到根目录中与 `_Utilities` 字符串下标为 0 的字符串，并将其存储在 `Root` 变量中。

3. 定义一个名为 `GetDirectories` 的方法，用于获取指定目录下的所有子目录。

4. 定义一个名为 `LangData` 的类，其中包含各种语言的单词表(即用于识别游戏中的数据和信息的关键字)。

5. 定义一个名为 `PortInfo` 的类，其中包含用于表示游戏中的数据和信息的标准类。

6. 通过 `static` 关键字声明一个名为 `Get` 静态方法，用于获取指定目录下的所有子目录，并将其转换为相应的 `PortInfo` 对象。

7. 通过调用 `GetDirectories` 方法获取游戏根目录，并将其存储在 `Root` 变量中。

8. 通过 `.SelectMany` 方法将获取到的子目录遍历并转换为相应的 `LangData` 类的实例，即单词表。

9. 通过 `.SelectT` 方法将遍历的每个单词表转换为一个 `PortInfo` 类的实例，并将其存储在 `Get` 方法的返回值中。

10. 通过 `.Where` 方法筛选出所有非空实例，并将它们存储在 `Get` 方法的返回值中。

11. 通过 `!` 取反操作获取一个包含所有 `PortInfo` 类的实例的数组，即 `Get` 方法的返回值。


```
﻿using System.Reflection;
using static System.IO.Directory;
using static DotnetUtils.Globals;

namespace DotnetUtils;

public static class PortInfos {
    public static readonly string Root;

    static PortInfos() {
        Root = GetParent(Assembly.GetEntryAssembly()!.Location)!.FullName;
        Root = Root[..Root.IndexOf(@"\00_Utilities")];

        Get = GetDirectories(Root)
            .SelectMany(gamePath => LangData.Keys.Select(keyword => (gamePath, keyword)))
            .SelectT((gamePath, keyword) => PortInfo.Create(gamePath, keyword))
            .Where(x => x is not null)
            .ToArray()!;
    }

    public static readonly PortInfo[] Get;
}

```

# `00_Utilities/DotnetUtils/DotnetUtils/Program.cs`

这段代码的作用是执行一系列的操作来检查软件开发中的各个流程，包括项目文件、代码文件、目标框架等。以下是具体的实现步骤：

1. 从指定的位置（可能是用户提供的文件或可能是硬编码）获取项目信息。
2. 遍历 infos 数组，为每个 infos 执行对应的操作。
3. 如果 infos 数组中包含操作数组 actions 的元素，则执行 actions 中的操作，并将结果输出。
4. 如果 infos 数组中包含操作数组 missingSln 的元素，则执行该操作，并将结果输出。
5. 如果 infos 数组中包含操作数组 unexpectedSlnName 的元素，则执行该操作，并将结果输出。
6. 如果 infos 数组中包含操作数组 multipleSlns 的元素，则执行该操作，并将结果输出。
7. 如果 infos 数组中包含操作数组 missingProj 的元素，则执行该操作，并将结果输出。
8. 如果 infos 数组中包含操作数组 unexpectedProjName 的元素，则执行该操作，并将结果输出。
9. 如果 infos 数组中包含操作数组 multipleProjs 的元素，则执行该操作，并将结果输出。
10. 如果 infos 数组中包含操作函数 checkProjects 的元素，则执行该函数，并将结果输出。
11. 如果 infos 数组中包含操作函数 checkExecutableProject 的元素，则执行该函数，并将结果输出。
12. 如果 infos 数组中包含操作函数 noCodeFiles 的元素，则执行该函数，并将结果输出。
13. 如果 infos 数组中包含操作函数 printInfos 的元素，则执行该函数，并将结果输出。
14. 如果 infos 数组中包含操作函数 generateMissingSlns 的元素，则执行该函数，并将结果输出。
15. 如果 infos 数组中包含操作函数 generateMissingProjs 的元素，则执行该函数，并将结果输出。

这些操作可以用于检查软件开发过程中的各个流程，以帮助开发人员更好地管理他们的项目。


```
﻿using System.Xml.Linq;
using DotnetUtils;
using static System.Console;
using static System.IO.Path;
using static DotnetUtils.Methods;
using static DotnetUtils.Functions;

var infos = PortInfos.Get;

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

```

这段代码使用了LINQ（Language-Integrated Query）库，用于在控制台输出一系列的行动（Action）对象的属性。这里的作用是循环遍历actions数组中的每个Action对象，并在每个Action对象上执行一个匿名函数。

匿名函数的作用是输出该Action对象的索引（Index）和描述（Description）。循环的变量包括：Action对象的索引、描述和当前的索引。

接下来，会输出一个包含GetChoice函数的结果。GetChoice函数的作用是获取数组actions的最后一个新的行动对象，并调用该行动对象的action方法。

然后定义了一个名为printSlns的函数，该函数接收一个PortInfo对象pi，输出与该pi.Slns.Length相关的信息。当pi.Slns.Length为0时，输出"No sln"；当pi.Slns.Length为1时，输出"Solution: {0}"；当pi.Slns.Length大于1时，输出"Solutions: {0} {1} ... {2} ... {pi.Slns.Length}"；然后遍历pi.Slns数组中的每个sln，输出其相对路径。

总的来说，这段代码的主要目的是输出与actions数组相关的信息，以及通过调用GetChoice函数输出数组的最后一个行动对象的 action 方法的结果。


```
foreach (var (_, description, index) in actions.WithIndex()) {
    WriteLine($"{index}: {description}");
}

WriteLine();

actions[getChoice(actions.Length - 1)].action();

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

```

这段代码是一个函数，名为 printProjs，它的参数是一个名为 PortInfo 的类，这个类可能包含与文件或项目相关的信息。

函数的作用是打印出与给定项目相关的信息。具体来说，它会在控制台输出以下信息：

- 如果只有一个项目，输出 "No project"。
- 如果只有一个项目，输出该项目的相对路径，以及该项目的名称。
- 如果多个项目，输出 "Projects:";
- 遍历所有项目，输出项目的相对路径，并在其后面输出一个空行。

pi.Projs 是 PortInfo 类的实例，它可能包含一个包含项目信息的数组。通过调用 printProjs 函数，我们可以方便地在这个数组中查找并输出对应的项目信息。


```
void printProjs(PortInfo pi) {
    switch (pi.Projs.Length) {
        case 0:
            WriteLine("No project");
            break;
        case 1:
            WriteLine($"Project: {pi.Projs[0].RelativePath(pi.LangPath)}");
            break;
        case > 1:
            WriteLine("Projects:");
            foreach (var proj in pi.Projs) {
                Write(proj.RelativePath(pi.LangPath));
                WriteLine();
            }
            break;
    }
}

```

这段代码定义了一个名为 `printInfos()` 的函数，其作用是打印出 `infos` 数组中的每个元素，并将相关信息输出到控制台。

具体来说，该函数包含以下步骤：

1. 通过 `foreach` 循环遍历 `infos` 数组中的每个元素。
2. 在每次遍历过程中，先输出该元素的 `LangPath` 字段，然后输出两个空行，接着调用 `printSlns()` 和 `printProjs()` 函数打印相关信息，最后获取该元素的 `CodeFiles` 数组并输出其中的每个元素。
3. 在循环结束后，输出一个新的一行，并在其左侧显示一个 50 字符的横线，以便在输出的过程中对齐。

由于 `printInfos()` 函数中调用了 `printSlns()` 和 `printProjs()` 函数，因此这些函数的具体实现可能因程序而异。


```
void printInfos() {
    foreach (var item in infos) {
        WriteLine(item.LangPath);
        WriteLine();

        printSlns(item);
        WriteLine();

        printProjs(item);
        WriteLine();

        // get code files
        foreach (var file in item.CodeFiles) {
            WriteLine(file.RelativePath(item.LangPath));
        }
        WriteLine(new string('-', 50));
    }
}

```

这两位代码定义了一个名为 missingSln 的函数和一个名为 unexpectedSlnName 的函数。

missingSln 的作用是查找信息集中所有缺少 Sln 的项，并将它们存储在一个名为 data 的数组中。然后使用 foreach 循环遍历 data 数组中的每个元素，并输出该元素的langPath属性。最后输出一个计数，表明找到的元素个数。

unexpectedSlnName 的作用是遍历 infos 集合中的每个元素，检查该元素是否缺少 Sln。如果是，该函数将记录下来，并输出该元素的 langPath 属性和一个预期的 Sln 名称。如果不是，该函数将记录计数器，并输出一个带有计数信息的字符串。最后输出一个计数，表明找到的元素个数。


```
void missingSln() {
    var data = infos.Where(x => x.Slns.None()).ToArray();
    foreach (var item in data) {
        WriteLine(item.LangPath);
    }
    WriteLine();
    WriteLine($"Count: {data.Length}");
}

void unexpectedSlnName() {
    var counter = 0;
    foreach (var item in infos) {
        if (item.Slns.None()) { continue; }

        var expectedSlnName = $"{item.GameName}.sln";
        if (item.Slns.Contains(Combine(item.LangPath, expectedSlnName), StringComparer.InvariantCultureIgnoreCase)) { continue; }

        counter += 1;
        WriteLine(item.LangPath);
        WriteLine($"Expected: {expectedSlnName}");

        printSlns(item);

        WriteLine();
    }
    WriteLine($"Count: {counter}");
}

```



这两段代码都是使用 Angular 中的口令(Slns)和项目(Projs)来存储用户信息。第一个代码的作用是获取所有 Slns 长度大于 1 的选项(即只存储 Slns 数量大于 1 的行)，并将它们存储在一个名为 data 的数组中。第二个代码的作用是获取所有 Projs 缺少的选项，并将它们存储在一个名为 data 的数组中。

然后，这两段代码都使用一个循环来遍历存储在 data 和 missingProj 数组中的所有选项，并输出每个选项的langPath。最后，第一段代码还计算了 data 数组中所有选项的数量，并输出了一个消息。第二段代码也计算了 missingProj 数组中所有选项的数量，并输出了一个消息。


```
void multipleSlns() {
    var data = infos.Where(x => x.Slns.Length > 1).ToArray();
    foreach (var item in data) {
        WriteLine(item.LangPath);
        printSlns(item);
    }
    WriteLine();
    WriteLine($"Count: {data.Length}");
}

void missingProj() {
    var data = infos.Where(x => x.Projs.None()).ToArray();
    foreach (var item in data) {
        WriteLine(item.LangPath);
    }
    WriteLine();
    WriteLine($"Count: {data.Length}");
}

```

这段代码是一个名为"unexpectedProjName"的函数，其目的是输出一个 unexpected project name，并包含一个包含该 unexpected project name 的信息列表。

该函数使用了几个循环，第一个循环遍历 infos 数组中的每个项目。对于每个项目，它首先检查该项目的 projs 数组是否为空。如果是，则直接跳过该项目。否则，它会遍历该项目的语言路径，并检查是否与给定的 expectedProjName 相同。如果是，则继续跳过该项目，并增加计数器。

接下来，该函数将遍历每个项目，并输出该项目的语言路径和意外的项目名称。然后，它将打印出该项目的所有 projs，并再次增加计数器。最后，它将打印结果，并输出计数器。

该函数的输出结果将是一个包含意外项目名称和对应项目信息的列表。


```
void unexpectedProjName() {
    var counter = 0;
    foreach (var item in infos) {
        if (item.Projs.None()) { continue; }

        var expectedProjName = $"{item.GameName}.{item.ProjExt}";
        if (item.Projs.Contains(Combine(item.LangPath, expectedProjName))) { continue; }

        counter += 1;
        WriteLine(item.LangPath);
        WriteLine($"Expected: {expectedProjName}");

        printProjs(item);

        WriteLine();
    }
    WriteLine($"Count: {counter}");
}

```



以上代码有两个主要的作用：

1. `multipleProjs()` 函数的主要目的是遍历 infos 集合中 Projs 数量大于 1 的项目，并将它们的信息存储在 data 数组中。

2. `generateMissingSlns()` 函数的主要目的是生成缺少SLN文件的缺失项目。对于每个缺少SLN文件的项目，该函数将运行 "dotnet" 命令来生成新的SLN文件，并将结果打印出来。然后，它将生成的SLN文件与项目文件名组合，并运行 "dotnet" 命令将它们添加到项目中。

"multipleProjs()" 和 "generateMissingSlns()" 函数的具体实现没有在代码中显示，因此我们无法看到它们如何工作。


```
void multipleProjs() {
    var data = infos.Where(x => x.Projs.Length > 1).ToArray();
    foreach (var item in data) {
        WriteLine(item.LangPath);
        WriteLine();
        printProjs(item);
    }
    WriteLine();
    WriteLine($"Count: {data.Length}");
}

void generateMissingSlns() {
    foreach (var item in infos.Where(x => x.Slns.None())) {
        var result = RunProcess("dotnet", $"new sln -n {item.GameName} -o {item.LangPath}");
        WriteLine(result);

        var slnFullPath = Combine(item.LangPath, $"{item.GameName}.sln");
        foreach (var proj in item.Projs) {
            result = RunProcess("dotnet", $"sln {slnFullPath} add {proj}");
            WriteLine(result);
        }
    }
}

```

这段代码是一个名为 `generateMissingProjs` 的函数，其目的是生成缺失的 projects。

具体来说，该函数遍历 `infos` 集合中的每个元素，其中每个元素都是一个缺少 `Projs` 属性的物品。对于每个元素，函数首先检查它是否已经包含一个与当前语言相同的项目模板。如果是，则不需要再创建一个新的空项目；如果不是，则需要创建一个新的空项目。

对于每个空项目，函数会生成一个 C# 项目模板，其中包括项目的名称、目标框架、目标语言版本等信息。这个模板中包含了 `<OutputType>Exe</OutputType>` 属性，表示生成的项目是一个可执行文件。


```
void generateMissingProjs() {
    foreach (var item in infos.Where(x => x.Projs.None())) {
        // We can't use the dotnet command to create a new project using the built-in console template, because part of that template
        // is a Program.cs / Program.vb file. If there already are code files, there's no need to add a new empty one; and
        // if there's already such a file, it might try to overwrite it.

        var projText = item.Lang switch {
            "csharp" => @"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
    <LangVersion>10</LangVersion>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
```

这段代码的作用是创建一个新的 VBnet 项目。它首先定义了一个 `Project` 对象，其中包含项目的元数据、目标框架和目标平台。然后，它定义了一个 `Properties` 对象，其中包含用于构建项目的所有属性。

接下来，它读取一个或多个 .NET 依赖项，并将它们添加到 `Properties` 对象中。最后，它根据一个名为 `Slns` 的数组来生成一个新的 .NET 依赖项，并运行 `dotnet` 命令来构建项目。

需要注意的是，这段代码在运行之前需要确保 `.NET` 和 `sln` 命令可用。此外，它还假设项目是 VBnet 格式，并使用 `Microsoft.NET.Sdk` 命名空间。


```
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
",
            _ => throw new InvalidOperationException()
        };
        var projFullPath = Combine(item.LangPath, $"{item.GameName}.{item.ProjExt}");
        File.WriteAllText(projFullPath, projText);

        if (item.Slns.Length == 1) {
            var result = RunProcess("dotnet", $"sln {item.Slns[0]} add {projFullPath}");
            WriteLine(result);
        }
    }
}

```

This is a code snippet written in C# that outputs warnings based on certain configurations if it is not satisfied by the given parents' values.

It first gets the TargetFramework property from the parent element, and then parses through the other properties. If the TargetFramework property is not "net6.0", a warning is output.

It then checks the Lang version and optionStrict properties. If the Lang version is not "10", a warning is output. And if optionStrict is "On", a warning is also output.

Finally, it checks for certain specific warnings if the Lang is "csharp" and optionStrict is "enable" or "On".

It is recommended to have the proper context to have a clear understanding of what the code is checking and why.


```
void checkProjects() {
    foreach (var info in infos) {
        WriteLine(info.LangPath);
        printProjectWarnings(info);
    }
}

void printProjectWarnings(PortInfo info) {
    foreach (var proj in info.Projs) {
        var warnings = new List<string>();
        var parent = XDocument.Load(proj).Element("Project")?.Element("PropertyGroup");

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

        if (framework != "net6.0") {
            warnings.Add($"Target: {framework}");
        }

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

        if (warnings.Any()) {
            WriteLine(proj.RelativePath(info.LangPath));
            WriteLine(string.Join("\n", warnings));
            WriteLine();
        }
    }
}

```



这两个函数的主要目的是分别检查项目和输出类型，并输出每个项目的输出类型，如果输出类型不是 "Exe"，则输出该项目的语言路径。

第一个函数 `checkExecutableProject()` 可以分为以下几个步骤：

1. 遍历 `infos` 数组中的每个项目。
2. 对于每个项目，使用 `getValue()` 函数获取该项目的输出类型。
3. 如果输出类型不是 "Exe"，则输出该项目的语言路径。

第二个函数 `noCodeFiles()` 也可以分为以下几个步骤：

1. 遍历 `infos` 数组中的每个项目。
2. 对于每个项目，使用 `Where()` 函数筛选出 `CodeFiles` 为非空的行。
3. 对于每个符合条件的项目，使用 `OrderBy()` 函数按照lang的顺序输出该项目的语言路径。
4. 在每个输出语句后，输出一条新行。


```
void checkExecutableProject() {
    foreach (var item in infos) {
        if (item.Projs.All(proj => getValue(proj, "OutputType") != "Exe")) {
            WriteLine($"{item.LangPath}");
        }
    }
}

void noCodeFiles() {
    var qry = infos
        .Where(x => x.CodeFiles.None())
        .OrderBy(x => x.Lang);
    foreach (var item in qry) {
        WriteLine(item.LangPath);
    }
}

```

It looks like you're trying to accomplish something with a language client and a series of questions and prompts. Without more context, it's difficult to know what you're trying to do. Could you please provide some more information about what you're trying to accomplish?


```
void tryBuild() {
    // if has code files, try to build
}

void printPortInfo() {
    // prompt for port number
    Write("Enter number from 1 to 96 ");
    var index = getChoice(1, 96);

    Write("Enter 0 for C#, 1 for VB ");
    var lang = getChoice(1) switch {
        0 => "csharp",
        1 => "vbnet",
        _ => throw new InvalidOperationException()
    };

    WriteLine();

    var info = infos.Single(x => x.Index == index && x.Lang == lang);

    WriteLine(info.LangPath);
    WriteLine(new string('-', 50));

    // print solutions
    printSlns(info);

    // mismatched solution name/location? (expected x)
    var expectedSlnName = Combine(info.LangPath, $"{info.GameName}.sln");
    if (!info.Slns.Contains(expectedSlnName)) {
        WriteLine($"Expected name/path: {expectedSlnName.RelativePath(info.LangPath)}");
    }

    // has executable project?
    if (info.Projs.All(proj => getValue(proj, "OutputType") != "Exe")) {
        WriteLine("No executable project");
    }

    WriteLine();

    // print projects
    printProjs(info);

    // mimsatched project name/location? (expected x)
    var expectedProjName = Combine(info.LangPath, $"{info.GameName}.{info.ProjExt}");
    if (info.Projs.Length < 2 && !info.Projs.Contains(expectedProjName)) {
        WriteLine($"Expected name/path: {expectedProjName.RelativePath(info.LangPath)}");
    }

    WriteLine();

    // verify project properties
    printProjectWarnings(info);

    WriteLine("Code files:");

    // list code files
    foreach (var codeFile in info.CodeFiles) {
        WriteLine(codeFile.RelativePath(info.LangPath));
    }

    // try build
}

```

### Acey Ducey

This is a simulation of the Acey Ducey card game. In the game, the dealer (the computer) deals two cards face up. You have an option to bet or not to bet depending on whether or not you feel the next card dealt will have a value between the first two.

Your initial money is set to $100; you may want to alter this value if you want to start with more or less than $100. The game keeps going on until you lose all your money or interrupt the program.

The original program author was Bill Palmby of Prairie View, Illinois.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=2)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=17)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- Entering a negative bet allows you to gain arbitrarily large amounts of money upon losing the round.

#### Porting Notes

- The assignment `N = 100` in line 100 has no effect; variable `N` is not used anywhere else in the program.

#### External Links
 - Common Lisp: https://github.com/koalahedron/lisp-computer-games/blob/master/01%20Acey%20Ducey/common-lisp/acey-deucy.lisp
 - PowerShell: https://github.com/eweilnau/basic-computer-games-powershell/blob/main/AceyDucey.ps1


# `01_Acey_Ducey/csharp/Game.cs`

This appears to be a program written in C# for the .NET framework. It is a simple game where the player must decide whether to bet or pass the next card. The program uses the `Console` class to interact with the user, displays the prompt on the screen, and uses a `while` loop to keep prompting until the player has made a decision.

The program starts by displaying a message asking the player if they want to try again or if they are done playing. If the player chooses to try again, the program will prompt them to enter a bet amount of $0. If the player chooses to pass, the program will tell them to input a bet amount.

The program then uses a `do-while` loop to keep prompting the user until they have made a decision. within this loop it will change the background color to yellow and display the prompt on the screen.

It is important to note that this program does not provide any way to implement the game logic, such as checking the rules of the game. It is only a simple interface to display the prompt to the user.


```
﻿using System;
using System.Collections.Generic;
using System.Text;

namespace AceyDucey
{
    /// <summary>
    /// The main class that implements all the game logic
    /// </summary>
    internal class Game
    {
        /// <summary>
        /// Our Random number generator object
        /// </summary>
        private Random Rnd { get; } = new Random();

        /// <summary>
        /// A line of underscores that we'll print between turns to separate them from one another on screen
        /// </summary>
        private string SeparatorLine { get; } = new string('_', 70);


        /// <summary>
        /// Main game loop function. This will play the game endlessly until the player chooses to quit.
        /// </summary>
        internal void GameLoop()
        {
            // First display instructions to the player
            DisplayIntroText();

            // We'll loop for each game until the player decides not to continue
            do
            {
                // Play a game!
                PlayGame();

                // Play again?
            } while (TryAgain());
        }

        /// <summary>
        /// Play the game
        /// </summary>
        private void PlayGame()
        {
            GameState state = new GameState();

            // Clear the display
            Console.Clear();
            // Keep looping until the player has no money left
            do
            {
                // Play the next turn. Pass in our state object so the turn can see the money available,
                // can update it after the player makes a bet, and can update the turn count.
                PlayTurn(state);

                // Keep looping until the player runs out of money
            } while (state.Money > 0);

            // Looks like the player is bankrupt, let them know how they did.
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("");
            Console.WriteLine($"Sorry, friend, but you blew your wad. Your game is over after {state.TurnCount} {(state.TurnCount == 1 ? "turn" : "turns")}. Your highest balance was ${state.MaxMoney}.");
        }


        /// <summary>
        /// Play a turn
        /// </summary>
        /// <param name="state">The current game state</param>
        private void PlayTurn(GameState state)
        {
            // Let the player know what's happening
            Console.WriteLine("");
            Console.WriteLine(SeparatorLine);
            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("");
            Console.WriteLine("Here are your next two cards:");

            // Generate two random cards
            int firstCard = GetCard();
            int secondCard = GetCard();

            // If the second card is lower than the first card, swap them over
            if (secondCard < firstCard)
            {
                (firstCard, secondCard) = (secondCard, firstCard);
            }

            // Display the cards
            DisplayCard(firstCard);
            DisplayCard(secondCard);

            // Ask the player what they want to do
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("");
            Console.Write("You currently have ");
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write($"${state.Money}");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine(". How much would you like to bet?");

            // Read the bet amount
            int betAmount = PlayTurn_GetBetAmount(state.Money);

            // Display a summary of their inpout
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("");
            Console.WriteLine($"You choose to {(betAmount == 0 ? "pass" : $"bet {betAmount}")}.");

            // Generate and display the final card
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("");
            Console.WriteLine("The next card is:");

            int thirdCard = GetCard();
            DisplayCard(thirdCard);
            Console.WriteLine("");

            // Was the third card between the first two cards?
            if (thirdCard > firstCard && thirdCard < secondCard)
            {
                // It was! Inform the player and add to their money
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("You win!");
                if (betAmount == 0)
                {
                    Console.WriteLine("(It's just a shame you chose not to bet!)");
                }
                else
                {
                    state.Money += betAmount;
                    // If their money exceeds the MaxMoney, update that too
                    state.MaxMoney = Math.Max(state.Money, state.MaxMoney);
                }
            }
            else
            {
                // Oh dear, the player lost. Let them know the bad news and take their bet from their money
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("You lose!");
                if (betAmount == 0)
                {
                    Console.WriteLine("(It's lucky you chose not to bet!)");
                }
                else
                {
                    state.Money -= betAmount;
                }
            }

            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("You now have ");
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write($"${state.Money}");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine(".");

            // Update the turn count now that another turn has been played
            state.TurnCount += 1;

            // Ready for the next turn...
            Console.ForegroundColor = ConsoleColor.DarkGreen;
            Console.WriteLine("");
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey(true);
        }

        /// <summary>
        /// Prompt the user for their bet amount and validate their input
        /// </summary>
        /// <param name="currentMoney">The player's current money</param>
        /// <returns>Returns the amount the player chooses to bet</returns>
        private int PlayTurn_GetBetAmount(int currentMoney)
        {
            int betAmount;
            // Loop until the user enters a valid value
            do
            {
                // Move this to a separate function...
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("> $");
                string input = Console.ReadLine();

                // Is this a valid number?
                if (!int.TryParse(input, out betAmount))
                {
                    // No
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Sorry, I didn't understand. Please enter how much you would like to bet.");
                    // Continue looping
                    continue;
                }

                // If the amount between 0 and their available money?
                if (betAmount < 0 || betAmount > currentMoney)
                {
                    // No
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"Please enter a bet amount between $0 and ${currentMoney}.");
                    // Continue looping
                    continue;
                }

                // We have a valid bet, stop looping
                break;
            } while (true);

            // Return whatever the player entered
            return betAmount;
        }

        /// <summary>
        /// Generate a new random card.
        /// </summary>
        /// <returns>Will return a value between 2 and 14, inclusive.</returns>
        /// <remarks>Values 2 to 10 are their face values. 11 represents a Jack, 12 is a Queen, 13 a King and 14 an Ace.
        /// Even though this is a slightly offset sequence, it allows us to perform a simple greater-than/less-than
        /// comparison with the card values, treating an Ace as a high card.</remarks>
        private int GetCard()
        {
            return Rnd.Next(2, 15);
        }

        /// <summary>
        /// Display the card number on screen, translating values 11 through to 14 into their named equivalents.
        /// </summary>
        /// <param name="card"></param>
        private void DisplayCard(int card)
        {
            string cardText;
            switch (card)
            {
                case 11:
                    cardText = "Jack";
                    break;
                case 12:
                    cardText = "Queen";
                    break;
                case 13:
                    cardText = "King";
                    break;
                case 14:
                    cardText = "Ace";
                    break;
                default:
                    cardText = card.ToString();
                    break;
            }

            // Format as black text on a white background
            Console.Write("   ");
            Console.BackgroundColor = ConsoleColor.White;
            Console.ForegroundColor = ConsoleColor.Black;
            Console.Write($"  {cardText}  ");
            Console.BackgroundColor = ConsoleColor.Black;
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("");
        }

        /// <summary>
        /// Display instructions on how to play the game and wait for the player to press a key.
        /// </summary>
        private void DisplayIntroText()
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Acey Ducey Gard Game.");
            Console.WriteLine("Creating Computing, Morristown, New Jersey.");
            Console.WriteLine("");

            Console.ForegroundColor = ConsoleColor.DarkGreen;
            Console.WriteLine("Originally published in 1978 in the book 'Basic Computer Games' by David Ahl.");
            Console.WriteLine("Modernised and converted to C# in 2021 by Adam Dawes (@AdamDawes575).");
            Console.WriteLine("");

            Console.ForegroundColor = ConsoleColor.Gray;
            Console.WriteLine("Acey Ducey is played in the following manner:");
            Console.WriteLine("");
            Console.WriteLine("The dealer (computer) deals two cards, face up.");
            Console.WriteLine("");
            Console.WriteLine("You have an option to bet or pass, depending on whether or not you feel the next card will have a value between the");
            Console.WriteLine("first two.");
            Console.WriteLine("");
            Console.WriteLine("If the card is between, you will win your stake, otherwise you will lose it. Ace is 'high' (higher than a King).");
            Console.WriteLine("");
            Console.WriteLine("If you want to pass, enter a bet amount of $0.");
            Console.WriteLine("");

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Press any key start the game.");
            Console.ReadKey(true);

        }

        /// <summary>
        /// Prompt the player to try again, and wait for them to press Y or N.
        /// </summary>
        /// <returns>Returns true if the player wants to try again, false if they have finished playing.</returns>
        private bool TryAgain()
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("Would you like to try again? (Press 'Y' for yes or 'N' for no)");

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("> ");

            char pressedKey;
            // Keep looping until we get a recognised input
            do
            {
                // Read a key, don't display it on screen
                ConsoleKeyInfo key = Console.ReadKey(true);
                // Convert to upper-case so we don't need to care about capitalisation
                pressedKey = Char.ToUpper(key.KeyChar);
                // Is this a key we recognise? If not, keep looping
            } while (pressedKey != 'Y' && pressedKey != 'N');
            // Display the result on the screen
            Console.WriteLine(pressedKey);

            // Return true if the player pressed 'Y', false for anything else.
            return (pressedKey == 'Y');
        }

    }
}

```

# `01_Acey_Ducey/csharp/GameState.cs`

This code defines a class called `GameState` that is used to keep track of game variables while the game is being played.

The `GameState` class has several properties, including an integer `Money` that represents the player's current balance, a variable of type `int` `MaxMoney` that keeps track of the highest amount of money the player has had at any point in the game, and an integer `TurnCount` that keeps track of the number of turns the player has played.

The class has a constructor that initializes all of the values of the properties to their default values. In this case, `Money` is set to 100, `MaxMoney` is also set to 100, and `TurnCount` is set to 0.

Overall, this code defines a `GameState` class that is used to store information about the game state, including the player's balance and the number of turns they have played.


```
﻿using System;
using System.Collections.Generic;
using System.Text;

namespace AceyDucey
{
    /// <summary>
    /// The GameState class keeps track of all the game variables while the game is being played
    /// </summary>
    internal class GameState
    {

        /// <summary>
        /// How much money does the player have at the moment?
        /// </summary>
        internal int Money { get; set; }

        /// <summary>
        /// What's the highest amount of money they had at any point in the game?
        /// </summary>
        internal int MaxMoney { get; set; }

        /// <summary>
        /// How many turns have they played?
        /// </summary>
        internal int TurnCount { get; set; }

        /// <summary>
        /// Class constructor -- initialise all values to their defaults.
        /// </summary>
        internal GameState()
        {
            // Setting Money to 100 gives the player their starting balance. Changing this will alter how much they have to begin with.
            Money = 100;
            MaxMoney = Money;
            TurnCount = 0;
        }
    }
}

```

# `01_Acey_Ducey/csharp/Program.cs`

这段代码是一个 C# 应用程序类，其中包含一个名为 `Main` 的静态方法。该方法的参数为字符串数组 `args`，这在代码中没有具体的作用。

在方法体内，首先创建一个名为 `Game` 的类实例。然后，调用该实例的 `GameLoop` 方法，该方法会无限循环地调用一个包含游戏逻辑的函数。在这里，调用 `game.GameLoop()` 将启动游戏的主循环，使游戏在后台持续运行，直到玩家选择退出。


```
﻿using System;
using System.Threading;

namespace AceyDucey
{
    /// <summary>
    /// The application's entry point
    /// </summary>
    class Program
    {

        /// <summary>
        /// This function will be called automatically when the application begins
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // Create an instance of our main Game class
            Game game = new Game();

            // Call its GameLoop function. This will play the game endlessly in a loop until the player chooses to quit.
            game.GameLoop();
        }


    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/) by Adam Dawes (@AdamDawes575, https://adamdawes.com).


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)

Two versions of Acey Ducey have been contributed.

The original upload supported JDK 8/JDK 11 and uses multiple files and the second uses features in JDK 17 and is implemented in a single file AceyDucey17.java.

Both are in the src folder.


# `01_Acey_Ducey/java/src/AceyDucey.java`

This looks like a card game where the player must decide whether to bet or not based on the two cards the dealer has shown. The player will have to decide whether to bet or not, and if they bet, they will have to choose a number between the first two cards the dealer has shown. The number will determine the value of the bet. The game will continue until the player either wins or the player chooses not to bet.


```
import java.util.Scanner;

/**
 * Game of AceyDucey
 * <p>
 * Based on the Basic game of AceyDucey here
 * https://github.com/coding-horror/basic-computer-games/blob/main/01%20Acey%20Ducey/aceyducey.bas
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class AceyDucey {

    // Current amount of players cash
    private int playerAmount;

    // First drawn dealer card
    private Card firstCard;

    // Second drawn dealer card
    private Card secondCard;

    // Players drawn card
    private Card playersCard;

    // User to display game intro/instructions
    private boolean firstTimePlaying = true;

    // game state to determine if game over
    private boolean gameOver = false;

    // Used for keyboard input
    private final Scanner kbScanner;

    // Constant value for cards from a deck - 2 lowest, 14 (Ace) highest
    public static final int LOW_CARD_RANGE = 2;
    public static final int HIGH_CARD_RANGE = 14;

    public AceyDucey() {
        // Initialise players cash
        playerAmount = 100;

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
    }

    // Play again method - public method called from class invoking game
    // If player enters YES then the game can be played again (true returned)
    // otherwise not (false)
    public boolean playAgain() {
        System.out.println();
        System.out.println("SORRY, FRIEND, BUT YOU BLEW YOUR WAD.");
        System.out.println();
        System.out.println();
        System.out.print("TRY AGAIN (YES OR NO) ");
        String playAgain = kbScanner.next().toUpperCase();
        System.out.println();
        System.out.println();
        if (playAgain.equals("YES")) {
            return true;
        } else {
            System.out.println("O.K., HOPE YOU HAD FUN!");
            return false;
        }
    }

    // game loop method

    public void play() {

        // Keep playing hands until player runs out of cash
        do {
            if (firstTimePlaying) {
                intro();
                firstTimePlaying = false;
            }
            displayBalance();
            drawCards();
            displayCards();
            int betAmount = getBet();
            playersCard = randomCard();
            displayPlayerCard();
            if (playerWon()) {
                System.out.println("YOU WIN!!");
                playerAmount += betAmount;
            } else {
                System.out.println("SORRY, YOU LOSE");
                playerAmount -= betAmount;
                // Player run out of money?
                if (playerAmount <= 0) {
                    gameOver = true;
                }
            }

        } while (!gameOver); // Keep playing until player runs out of cash
    }

    // Method to determine if player won (true returned) or lost (false returned)
    // to win a players card has to be in the range of the first and second dealer
    // drawn cards inclusive of first and second cards.
    private boolean playerWon() {
        // winner
        return (playersCard.getValue() >= firstCard.getValue())
                && playersCard.getValue() <= secondCard.getValue();

    }

    private void displayPlayerCard() {
        System.out.println(playersCard.getName());
    }

    // Get the players bet, and return the amount
    // 0 is considered a valid bet, but better more than the player has available is not
    // method will loop until a valid bet is entered.
    private int getBet() {
        boolean validBet = false;
        int amount;
        do {
            System.out.print("WHAT IS YOUR BET ");
            amount = kbScanner.nextInt();
            if (amount == 0) {
                System.out.println("CHICKEN!!");
                validBet = true;
            } else if (amount > playerAmount) {
                System.out.println("SORRY, MY FRIEND, BUT YOU BET TOO MUCH.");
                System.out.println("YOU HAVE ONLY " + playerAmount + " DOLLARS TO BET.");
            } else {
                validBet = true;
            }
        } while (!validBet);

        return amount;
    }

    private void displayBalance() {
        System.out.println("YOU NOW HAVE " + playerAmount + " DOLLARS.");
    }

    private void displayCards() {
        System.out.println("HERE ARE YOUR NEXT TWO CARDS: ");
        System.out.println(firstCard.getName());
        System.out.println(secondCard.getName());
    }

    // Draw two dealer cards, and save them for later use.
    // ensure that the first card is a smaller value card than the second one
    private void drawCards() {

        do {
            firstCard = randomCard();
            secondCard = randomCard();
        } while (firstCard.getValue() >= secondCard.getValue());
    }

    // Creates a random card
    private Card randomCard() {
        return new Card((int) (Math.random()
                * (HIGH_CARD_RANGE - LOW_CARD_RANGE + 1) + LOW_CARD_RANGE));
    }

    public void intro() {
        System.out.println("ACEY DUCEY CARD GAME");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println();
        System.out.println("ACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER");
        System.out.println("THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP");
        System.out.println("YOU HAVE AN OPTION TO BET OR NOT BET DEPENDING");
        System.out.println("ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE");
        System.out.println("A VALUE BETWEEN THE FIRST TWO.");
        System.out.println("IF YOU DO NOT WANT TO BET, INPUT: 0");
    }
}

```

# `01_Acey_Ducey/java/src/AceyDucey17.java`

This is likely a section of code for a video game, specifically a financial game where players must make decisions to管理 their virtual cash. The code appears to be explaining some aspects of the game and how it works, such as the different ways the game can be progressed, the different types of cards that can be used, and the different ways the game can output information to the player. It seems to be based on the original Ultimate Cash game, which was released in 2001.



```
import java.util.Random;
import java.util.Scanner;

/**
 * A modern version (JDK17) of ACEY DUCEY using post Java 8 features. Notes
 * regarding new java features or differences in the original basic
 * implementation are numbered and at the bottom of this code.
 * The goal is to recreate the exact look and feel of the original program
 * minus a large glaring bug in the original code that lets you cheat.
 */
public class AceyDucey17 {

  public static void main(String[] args) {
    // notes [1]
    System.out.println("""
                                        ACEY DUCEY CARD GAME
                             CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY


              ACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER
              THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP
              YOU HAVE AN OPTION TO BET OR NOT BET DEPENDING
              ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE
              A VALUE BETWEEN THE FIRST TWO.
              IF YOU DO NOT WANT TO BET, INPUT A 0""");

    do {
      playGame();
    } while (stillInterested());
    System.out.println("O.K., HOPE YOU HAD FUN!");
  }

  public static void playGame() {
    int cashOnHand = 100; // our only mutable variable  note [11]
    System.out.println("YOU NOW HAVE  "+ cashOnHand +"  DOLLARS.");// note [6]
    while (cashOnHand > 0) {
      System.out.println();
      System.out.println("HERE ARE YOUR NEXT TWO CARDS:");

      final Card lowCard = Card.getRandomCard(2, Card.KING); //note [3]
      System.out.println(lowCard);
      final Card highCard = Card.getRandomCard(lowCard.rank() + 1, Card.ACE);
      System.out.println(highCard);

      final int bet = getBet(cashOnHand);
      final int winnings = determineWinnings(lowCard,highCard,bet);
      cashOnHand += winnings;
      if(winnings != 0 || cashOnHand != 0){  //note [2]
        System.out.println("YOU NOW HAVE  "+ cashOnHand +"  DOLLARS.");//note [6]
      }
    }
  }

  public static int determineWinnings(Card lowCard, Card highCard, int bet){
    if (bet <= 0) {    // note [5]
      System.out.println("CHICKEN!!");
      return 0;
    }
    Card nextCard = Card.getRandomCard(2, Card.ACE);
    System.out.println(nextCard);
    if(nextCard.between(lowCard,highCard)){
      System.out.println("YOU WIN!!!");
      return bet;
    }
    System.out.println("SORRY, YOU LOSE");
    return -bet;
  }

  public static boolean stillInterested(){
    System.out.println();
    System.out.println();
    System.out.println("SORRY, FRIEND, BUT YOU BLEW YOUR WAD.");
    System.out.println();
    System.out.println();
    System.out.print("TRY AGAIN (YES OR NO)? ");
    Scanner input = new Scanner(System.in);
    boolean playAgain = input.nextLine()
                             .toUpperCase()
                             .startsWith("Y"); // note [9]
    System.out.println();
    System.out.println();
    return playAgain;
  }

  public static int getBet(int cashOnHand){
    int bet;
    do{
      System.out.println();
      System.out.print("WHAT IS YOUR BET? ");
      bet = inputNumber();
      if (bet > cashOnHand) {
        System.out.println("SORRY, MY FRIEND, BUT YOU BET TOO MUCH.");
        System.out.println("YOU HAVE ONLY  "+cashOnHand+"  DOLLARS TO BET.");
      }
    }while(bet > cashOnHand);
    return bet;
  }

  public static int inputNumber() {
    final Scanner input = new Scanner(System.in);
    // set to negative to mark as not entered yet in case of input error.
    int number = -1;
    while (number < 0) {
      try {
        number = input.nextInt();
      } catch(Exception ex) {   // note [7]
        System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");
        System.out.print("? ");
        try{
          input.nextLine();
        }
        catch(Exception ns_ex){ // received EOF (ctrl-d or ctrl-z if windows)
          System.out.println("END OF INPUT, STOPPING PROGRAM.");
          System.exit(1);
        }
      }
    }
    return number;
  }

  record Card(int rank){
    // Some constants to describe face cards.
    public static final int JACK = 11, QUEEN = 12, KING = 13, ACE = 14;
    private static final Random random = new Random();

    public static Card getRandomCard(int from, int to){
      return new Card(random.nextInt(from, to+1));  // note [4]
    }

    public boolean between(Card lower, Card higher){
      return lower.rank() < this.rank() && this.rank() < higher.rank();
    }

    @Override
    public String toString() { // note [13]
      return switch (rank) {
        case JACK -> "JACK";
        case QUEEN -> "QUEEN";
        case KING -> "KING";
        case ACE -> "ACE\n"; // note [10]
        default -> " "+rank+" "; // note [6]
      };
    }
  }

  /*
    Notes:
    1. Multiline strings, a.k.a. text blocks, were added in JDK15.
    2. The original game only displays the players balance if it changed,
       which it does not when the player chickens out and bets zero.
       It also doesn't display the balance when it becomes zero because it has
       a more appropriate message: Sorry, You Lose.
    3. To pick two cards to show, the original BASIC implementation has a
       bug that could cause a race condition if the RND function never chose
       a lower number first and higher number second. It loops infinitely
       re-choosing random numbers until the condition is met of the first
       one being lower. The logic is changed a bit here so that the first
       card picked is anything but an ACE, the highest possible card,
       and then the second card is between the just picked first card upto
       and including the ACE.
    4. Random.nextInt(origin, bound) was added in JDK17, and allows to
       directly pick a range for a random integer to be generated. The second
       parameter is exclusive of the range and thus why they are stated with
       +1's to the face card.
    5. The original BASIC implementation has a bug that allows negative value
       bets. Since you can't bet MORE cash than you have you can always bet
       less including a very, very large negative value. You would do this when
       the chances of winning are slim or zero since losing a hand SUBTRACTS
       your bet from your cash; subtracting a negative number actually ADDS
       to your cash, potentially making you an instant billionaire.
       This loophole is now closed.
    6. The subtle behavior of the BASIC PRINT command causes a space to be
       printed before all positive numbers as well as a trailing space. Any
       place a non-face card or the players balance is printed has extra space
       to mimic this behavior.
    7. Errors on input were probably specific to the interpreter. This program
       tries to match the Vintage Basic interpreter's error messages. The final
       input.nextLine() command exists to clear the blockage of whatever
       non-number input was entered.  But even that could fail if the user
       types Ctrl-D (windows Ctrl-Z), signifying an EOF (end of file) and thus
       the closing of STDIN channel. The original program on an EOF signal prints
       "END OF INPUT IN LINE 660" and thus we cover it roughly the same way.
       All of this is necessary to avoid a messy stack trace from being
       printed as the program crashes.
    9. The original game only accepted a full upper case "YES" to continue
       playing if bankrupted. This program is more lenient and will accept
       any input that starts with the letter 'y', uppercase or not.
   10. The original game prints an extra blank line if the card is an ACE. There
       is seemingly no rationale for this.
   11. Modern java best practices are edging toward a more functional paradigm
       and as such, mutating state is discouraged. All other variables besides
       the cashOnHand are final and initialized only once.
   12. Refactoring of the concept of a card is done with a record. Records were
       introduced in JDK14. Card functionality is encapsulated in this example
       of a record.  An enum could be a better alternative since there are
       technically only 13 cards possible.
   13. Switch expressions were introduced as far back as JDK12 but continue to
       be refined for clarity, exhaustiveness. As of JDK17 pattern matching
       for switch expressions can be accessed by enabling preview features.
   */
}

```