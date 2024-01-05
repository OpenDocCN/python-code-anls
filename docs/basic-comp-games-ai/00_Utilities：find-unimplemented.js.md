# `d:/src/tocomm/basic-computer-games\00_Utilities\find-unimplemented.js`

```
# Program to show unimplemented games by language, optionally filtered by language
# Usage: node find-unimplemented.js [[[lang1] lang2] ...]
# Adapted from find-missing-implementtion.js

# 引入文件系统模块
const fs = require("fs");
# 引入文件匹配模块
const glob = require("glob");

# 相对路径到仓库根目录
const ROOT_PATH = "../.";

# 定义语言列表
let languages = [
  { name: "csharp", extension: "cs" },
  { name: "java", extension: "java" },
  { name: "javascript", extension: "html" },
  { name: "pascal", extension: "pas" },
  { name: "perl", extension: "pl" },  // 创建一个对象，表示文件类型为 perl，扩展名为 pl
  { name: "python", extension: "py" },  // 创建一个对象，表示文件类型为 python，扩展名为 py
  { name: "ruby", extension: "rb" },  // 创建一个对象，表示文件类型为 ruby，扩展名为 rb
  { name: "vbnet", extension: "vb" },  // 创建一个对象，表示文件类型为 vbnet，扩展名为 vb
];

const getFilesRecursive = async (path, extension) => {  // 创建一个异步函数，用于递归获取指定路径下指定扩展名的文件
  return new Promise((resolve, reject) => {  // 返回一个 Promise 对象
    glob(`${path}/**/*.${extension}`, (err, matches) => {  // 使用 glob 模块匹配指定路径下指定扩展名的文件
      if (err) {
        reject(err);  // 如果出现错误，reject Promise
      }
      resolve(matches);  // 如果匹配成功，resolve Promise 并返回匹配的文件列表
    });
  });
};

const getPuzzleFolders = () => {  // 创建一个函数，用于获取谜题文件夹
  return fs
    .readdirSync(ROOT_PATH, { withFileTypes: true })  // 同步读取指定路径下的文件夹，并返回文件夹信息
    .filter((dirEntry) => dirEntry.isDirectory())  // 过滤出目录类型的文件
    .filter(
      (dirEntry) =>
        ![".git", "node_modules", "00_Utilities", "buildJvm"].includes(dirEntry.name)  // 过滤掉特定的目录名
    )
    .map((dirEntry) => dirEntry.name);  // 将目录名映射成一个新的数组
};

(async () => {
  const result = {};  // 创建一个空对象
  if (process.argv.length > 2) {  // 如果命令行参数的长度大于2
    languages = languages.filter((language) => process.argv.slice(2).includes(language.name));  // 过滤出包含在命令行参数中的语言
  }
  for (const { name: language } of languages) {  // 遍历语言数组
    result[language] = [];  // 在结果对象中创建以语言名为键的空数组
  }

  const puzzleFolders = getPuzzleFolders();  // 获取谜题文件夹
  for (const puzzleFolder of puzzleFolders) {  // 遍历谜题文件夹
    for (const { name: language, extension } of languages) {  // 遍历语言数组
      # 递归获取指定文件夹下指定语言的所有文件
      files = await getFilesRecursive(
        f'{ROOT_PATH}/{puzzleFolder}/{language}', extension
      )
      # 如果获取的文件列表长度为0，将当前语言下的文件夹名称添加到结果字典中
      if (files.length === 0) {
        result[language].push(puzzleFolder)
      }
    }
  }
  # 打印未实现的语言和文件夹
  console.log('Unimplementation by language:')
  console.dir(result)
})()
# 返回，结束函数执行
return
```