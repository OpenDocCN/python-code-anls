# `basic-computer-games\00_Utilities\build-index.js`

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

const TITLE = 'BASIC Computer Games'; // 定义标题
const JAVASCRIPT_FOLDER = 'javascript'; // 定义 JavaScript 文件夹名称
const IGNORE_FOLDERS_START_WITH = ['.', '00_', 'buildJvm', 'Sudoku']; // 定义需要忽略的文件夹名称前缀
const IGNORE_FILES = [
	// "84 Super Star Trek"  has it's own node/js implementation (using xterm)
	'cli.mjs', 'superstartrek.mjs'
]; // 定义需要忽略的文件名

function createGameLinks(game) {
	// 创建游戏链接
	const creatFileLink = (file, name = path.basename(file)) => {
		if (file.endsWith('.html')) {
			return `
				<li><a href="${file}">${name.replace('.html', '')}</a></li>
			`;
		} else if (file.endsWith('.mjs')) {
			return `
				<li><a href="./00_Common/javascript/WebTerminal/terminal.html#${file}">${name.replace('.mjs', '')} (node.js)</a></li>
			`;
		} else {
			throw new Error(`Unknown file-type found: ${file}`);
		}
	}

	if (game.files.length > 1) {
		const entries = game.files.map(file => {
			return creatFileLink(file);
		});
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

function createIndexHtml(title, games) {
	// 创建 index.html 文件内容
	const listEntries = games.map(game => 
		createGameLinks(game)
	).map(entry => `\t\t\t${entry}`).join('\n');

	const head = `
		<head>
			<meta charset="UTF-8">
			<title>${title}</title>
			<link rel="stylesheet" href="./00_Utilities/javascript/style_terminal.css" />
		</head>
	`; // 定义 HTML 头部内容

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
	`; // 定义 HTML 主体内容

	return `
		<!DOCTYPE html>
		<html lang="en">
		${head}
		${body}
		</html>
	`.trim().replace(/\s\s+/g, ''); // 返回整个 HTML 内容
}

function findJSFilesInFolder(folder) {
	// 查找文件夹中的 JavaScript 文件
	const hasJavascript = fs.existsSync(`${folder}/${JAVASCRIPT_FOLDER}`);
	if (!hasJavascript) {
		throw new Error(`Game "${folder}" is missing a javascript folder`);
	}

	const files = fs.readdirSync(`${folder}/${JAVASCRIPT_FOLDER}`); // 获取文件夹中的所有文件

	const htmlFiles = files.filter(file => file.endsWith('.html')); // 过滤出 HTML 文件
	const mjsFiles = files.filter(file => file.endsWith('.mjs')); // 过滤出 MJS 文件
	const entries = [
		...htmlFiles,
		...mjsFiles
	].filter(file => !IGNORE_FILES.includes(file)); // 合并并过滤出需要忽略的文件

	if (entries.length == 0) {
		throw new Error(`Game "${folder}" is missing a HTML or node.js file in the folder "${folder}/${JAVASCRIPT_FOLDER}"`);
	}

	return entries.map(file => path.join(folder, JAVASCRIPT_FOLDER, file)); // 返回文件路径数组
}

function main() {
	// 获取当前目录中的所有文件夹
	let folders = fs.readdirSync(process.cwd());

	folders = folders.filter(folder => fs.statSync(folder).isDirectory()); // 过滤出文件夹

	folders = folders.filter(folder => {
		return !IGNORE_FOLDERS_START_WITH.some(ignore => folder.startsWith(ignore));
	}); // 过滤出需要忽略的文件夹

	folders = folders.sort(); // 对文件夹进行排序

	const games = folders.map(folder => {
		const name = folder.replace('_', ' '); // 替换文件夹名称中的下划线
		let files;
		
		try {
			files = findJSFilesInFolder(folder); // 查找文件夹中的 JavaScript 文件
		} catch (error) {
			console.warn(`Game "${name}" is missing a javascript implementation: ${error.message}`);
			return null;
		}

		return {
			name,
			files
		}
	}).filter(game => game !== null); // 创建游戏对象数组并过滤掉空对象

	const htmlContent = createIndexHtml(TITLE, games); // 创建 index.html 文件内容
	fs.writeFileSync('index.html', htmlContent); // 将内容写入 index.html 文件
	console.log(`index.html successfully created!`); // 输出成功信息
}

main(); // 执行主函数

```