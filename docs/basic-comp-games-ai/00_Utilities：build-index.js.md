# `d:/src/tocomm/basic-computer-games\00_Utilities\build-index.js`

```
#!/usr/bin/env node
/**
 * This script creates an "index.html" file in the root of the directory.
 * 
 * Call this from the root of the project with 
 *  `node ./00_Utilities/build-index.js`
 * 
 * @author Alexander Wunschik <https://github.com/mojoaxel>
 */

const fs = require('fs'); // 引入文件系统模块
const path = require('path'); // 引入路径模块

const TITLE = 'BASIC Computer Games'; // 定义标题常量
const JAVASCRIPT_FOLDER = 'javascript'; // 定义 JavaScript 文件夹名称常量
const IGNORE_FOLDERS_START_WITH = ['.', '00_', 'buildJvm', 'Sudoku']; // 定义需要忽略的文件夹名称列表
const IGNORE_FILES = [
	// "84 Super Star Trek"  has it's own node/js implementation (using xterm)
	'cli.mjs', 'superstartrek.mjs' // 定义需要忽略的文件列表
];
# 定义一个名为createGameLinks的函数，接受一个名为game的参数
def createGameLinks(game):
    # 定义一个名为creatFileLink的函数，接受一个名为file和一个默认值为path.basename(file)的name参数
    def creatFileLink(file, name = path.basename(file)):
        # 如果文件以.html结尾
        if (file.endswith('.html')):
            # 返回一个包含文件链接的HTML列表项
            return `
                <li><a href="${file}">${name.replace('.html', '')}</a></li>
            `;
        # 如果文件以.mjs结尾
        elif (file.endswith('.mjs')):
            # 返回一个包含文件链接的HTML列表项
            return `
                <li><a href="./00_Common/javascript/WebTerminal/terminal.html#${file}">${name.replace('.mjs', '')} (node.js)</a></li>
            `;
        # 如果文件类型未知
        else:
            # 抛出一个错误
            throw new Error(`Unknown file-type found: ${file}`);
    
    # 如果游戏文件数量大于1
    if (len(game.files) > 1):
        # 对游戏文件列表中的每个文件调用creatFileLink函数，并将结果存储在entries中
        entries = [creatFileLink(file) for file in game.files]
		return `
			<li>
				<span>${game.name}</span>
				<ul>${entries.map(e => `\t\t\t${e}`).join('\n')}</ul>
			</li>
		`;
	} else {
		return creatFileLink(game.files[0], game.name);
	}
}
```
这部分代码是一个函数，根据条件返回不同的结果。如果条件为真，返回一个包含游戏名称和文件列表的 HTML 列表项，否则返回一个文件链接。

```
function createIndexHtml(title, games) {
	const listEntries = games.map(game => 
		createGameLinks(game)
	).map(entry => `\t\t\t${entry}`).join('\n');

	const head = `
		<head>
			<meta charset="UTF-8">
			<title>${title}</title>
```
这部分代码是一个函数，用于创建包含游戏列表的 HTML 页面。它首先使用 map 函数将游戏列表转换为包含游戏链接的数组，然后使用 join 函数将数组中的元素连接成一个字符串。接着创建了 head 变量，包含了 HTML 页面的头部信息。
			<link rel="stylesheet" href="./00_Utilities/javascript/style_terminal.css" />
		</head>
	`;
```
这行代码是将一个样式表链接添加到 HTML 文档的头部。

```python
	const body = `
		<body>
			<article id="output">
				<header>
					<h1>${title}</h1>
				</header>
				<main>
					<ul>
						${listEntries}
					</ul>
				</main>
			</article>
		</body>
	`;
```
这段代码是创建一个包含文章标题和条目列表的 HTML 文档的主体部分。

```python
	return `
```
这行代码是返回一个包含样式表链接和主体内容的完整 HTML 文档。
		<!DOCTYPE html>  # 声明文档类型为 HTML
		<html lang="en">  # 设置页面语言为英文
		${head}  # 插入 head 部分的内容
		${body}  # 插入 body 部分的内容
		</html>  # HTML 结束标签
	`.trim().replace(/\s\s+/g, '');  # 去除字符串两端的空白字符，并替换多个连续空白字符为单个空白字符
}

function findJSFilesInFolder(folder) {
	// filter folders that do not include a subfolder called "javascript"  # 过滤不包含名为 "javascript" 的子文件夹的文件夹
	const hasJavascript = fs.existsSync(`${folder}/${JAVASCRIPT_FOLDER}`);  # 检查文件夹是否包含名为 "javascript" 的子文件夹
	if (!hasJavascript) {  # 如果没有名为 "javascript" 的子文件夹
		throw new Error(`Game "${folder}" is missing a javascript folder`);  # 抛出错误，提示缺少名为 "javascript" 的子文件夹
	}

	// get all files in the javascript folder  # 获取 "javascript" 文件夹中的所有文件
	const files = fs.readdirSync(`${folder}/${JAVASCRIPT_FOLDER}`);  # 读取 "javascript" 文件夹中的所有文件

	// filter files only allow .html files  # 过滤只允许 .html 文件
	const htmlFiles = files.filter(file => file.endsWith('.html'));  # 过滤出只有 .html 扩展名的文件
	const mjsFiles = files.filter(file => file.endsWith('.mjs')); // 从文件列表中筛选出以'.mjs'结尾的文件，存入mjsFiles数组中
	const entries = [ // 创建一个包含htmlFiles和mjsFiles的数组
		...htmlFiles, // 将htmlFiles数组中的元素展开
		...mjsFiles // 将mjsFiles数组中的元素展开
	].filter(file => !IGNORE_FILES.includes(file)); // 从数组中筛选出不在IGNORE_FILES数组中的文件

	if (entries.length == 0) { // 如果entries数组长度为0
		throw new Error(`Game "${folder}" is missing a HTML or node.js file in the folder "${folder}/${JAVASCRIPT_FOLDER}"`); // 抛出一个错误，指示缺少HTML或node.js文件
	}

	return entries.map(file => path.join(folder, JAVASCRIPT_FOLDER, file)); // 返回一个新数组，其中包含folder、JAVASCRIPT_FOLDER和entries数组中的每个元素拼接而成的路径
}

function main() {
	// Get the list of all folders in the current director
	let folders = fs.readdirSync(process.cwd()); // 获取当前目录下的所有文件和文件夹

	// filter files only allow folders
	folders = folders.filter(folder => fs.statSync(folder).isDirectory()); // 从folders数组中筛选出只包含文件夹的元素
	// 过滤掉以点或00_开头的文件夹
	folders = folders.filter(folder => {
		return !IGNORE_FOLDERS_START_WITH.some(ignore => folder.startsWith(ignore));
	});

	// 按字母顺序（按数字）对文件夹进行排序
	folders = folders.sort();

	// 从文件夹中获取名称和 JavaScript 文件
	const games = folders.map(folder => {
		const name = folder.replace('_', ' ');  // 用空格替换下划线
		let files;
		
		try {
			files = findJSFilesInFolder(folder);  // 在文件夹中查找 JavaScript 文件
		} catch (error) {
			console.warn(`Game "${name}" is missing a javascript implementation: ${error.message}`);  // 输出警告信息
			return null;  // 返回空值
		}
		return {
			name,  # 返回一个包含 name 和 files 的对象
			files  # 返回一个包含 name 和 files 的对象
		}
	}).filter(game => game !== null);  # 使用 filter 方法过滤掉值为 null 的元素

	// create a index.html file with a list of all games  # 创建一个包含所有游戏列表的 index.html 文件
	const htmlContent = createIndexHtml(TITLE, games);  # 调用 createIndexHtml 函数创建 HTML 内容
	fs.writeFileSync('index.html', htmlContent);  # 将 HTML 内容写入 index.html 文件
	console.log(`index.html successfully created!`);  # 打印成功创建 index.html 文件的消息
}

main();  # 调用 main 函数
```