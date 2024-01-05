# `00_Utilities\find-missing-implementations.js`

```
# Program to find games that are missing solutions in a given language
# Scan each game folder, check for a folder for each language, and also make
# sure there's at least one file of the expected extension and not just a
# readme or something

# 引入文件系统模块
const fs = require("fs");
# 引入文件匹配模块
const glob = require("glob");

# 相对于仓库根目录的路径
const ROOT_PATH = "../.";

# 定义语言列表
const languages = [
  { name: "csharp", extension: "cs" },  # C#语言，文件扩展名为.cs
  { name: "java", extension: "java" },  # Java语言，文件扩展名为.java
  { name: "javascript", extension: "html" },  # JavaScript语言，文件扩展名为.html
  { name: "pascal", extension: "pas" },  # Pascal语言，文件扩展名为.pas
  { name: "perl", extension: "pl" },  # Perl语言，文件扩展名为.pl
```
```python
  { name: "python", extension: "py" },  # 创建一个包含语言名称和文件扩展名的字典
  { name: "ruby", extension: "rb" },  # 创建一个包含语言名称和文件扩展名的字典
  { name: "vbnet", extension: "vb" },  # 创建一个包含语言名称和文件扩展名的字典
];

const getFilesRecursive = async (path, extension) => {  # 创建一个异步函数，用于递归获取指定路径下指定扩展名的文件
  return new Promise((resolve, reject) => {  # 返回一个 Promise 对象
    glob(`${path}/**/*.${extension}`, (err, matches) => {  # 使用 glob 模块匹配指定路径下指定扩展名的文件
      if (err) {
        reject(err);  # 如果出现错误，reject Promise
      }
      resolve(matches);  # 如果成功，resolve Promise 并返回匹配的文件列表
    });
  });
};

const getPuzzleFolders = () => {  # 创建一个函数，用于获取谜题文件夹
  return fs
    .readdirSync(ROOT_PATH, { withFileTypes: true })  # 同步读取指定路径下的文件和子目录
    .filter((dirEntry) => dirEntry.isDirectory())  # 过滤出子目录
    .filter(
      (dirEntry) =>
        ![".git", "node_modules", "00_Utilities"].includes(dirEntry.name)
    )
    .map((dirEntry) => dirEntry.name);
```
这段代码是使用JavaScript的数组方法对dirEntry进行过滤和映射操作，根据条件排除名字为[".git", "node_modules", "00_Utilities"]的文件夹，然后返回剩余文件夹的名字数组。

```python
};

(async () => {
  let missingGames = {};
  let missingLanguageCounts = {};
  languages.forEach((l) => (missingLanguageCounts[l.name] = 0));
  const puzzles = getPuzzleFolders();
  for (const puzzle of puzzles) {
    for (const { name: language, extension } of languages) {
      const files = await getFilesRecursive(
        `${ROOT_PATH}/${puzzle}/${language}`,
        extension
      );
      if (files.length === 0) {
        if (!missingGames[puzzle]) missingGames[puzzle] = [];
```
这段代码是一个异步函数，使用了async/await语法。首先初始化了两个空对象missingGames和missingLanguageCounts，然后对languages数组进行遍历，将每个语言的名字作为missingLanguageCounts对象的属性，初始值为0。接着调用getPuzzleFolders函数获取谜题文件夹列表，然后对每个谜题文件夹进行遍历，再对每个语言进行遍历，调用getFilesRecursive函数获取文件列表，如果文件列表为空，则将该谜题加入missingGames对象中。
        missingGames[puzzle].push(language);  # 将缺失的游戏与语言的组合添加到missingGames对象中
        missingLanguageCounts[language]++;    # 增加缺失语言的计数
      }
    }
  }
  const missingCount = Object.values(missingGames).flat().length;  # 计算缺失的游戏与语言的组合数量
  if (missingCount === 0) {  # 如果没有缺失的游戏与语言的组合
    console.log("All games have solutions for all languages");  # 输出所有游戏都有所有语言的解决方案
  } else {
    console.log(`Missing ${missingCount} implementations:`);  # 输出缺失的游戏与语言的组合数量

    Object.entries(missingGames).forEach(  # 遍历missingGames对象的条目
      ([p, ls]) => (missingGames[p] = ls.join(", "))  # 将游戏与缺失语言的组合转换为字符串
    );

    console.log(`\nMissing languages by game:`);  # 输出按游戏分类的缺失语言
    console.table(missingGames);  # 以表格形式输出missingGames对象

    console.log(`\nBy language:`);  # 输出按语言分类的缺失数量
    console.table(missingLanguageCounts);  # 以表格形式输出missingLanguageCounts对象
  }
})();  # 这是一个立即执行函数表达式（Immediately Invoked Function Expression，IIFE），用于创建一个局部作用域并执行其中的代码

return;  # 返回语句，用于结束函数的执行并返回到调用函数的位置
```