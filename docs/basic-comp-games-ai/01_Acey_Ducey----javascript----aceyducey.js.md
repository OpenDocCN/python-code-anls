# `basic-computer-games\01_Acey_Ducey\javascript\aceyducey.js`

```

// UTILITY VARIABLES

// 默认情况下：
// — 浏览器有一个 window 对象
// — Node.js 没有
// 检查未定义的 window 对象是一种宽松的检查，
// 以支持浏览器和 Node.js
const isRunningInBrowser = typeof window !== 'undefined';

// 用于轻松验证输入字符串的实用函数
const validLowerCaseYesStrings = ['yes', 'y'];
const validLowerCaseNoStrings = ['no', 'n'];
const validLowerCaseYesAndNoStrings = [
    ...validLowerCaseYesStrings,
    ...validLowerCaseNoStrings,
];
// UTILITY VARIABLES

// 获取一个随机数（牌）2-14（ACE 是 14）
function getRandomCard() {
    // 在我们的游戏中，ACE 的值大于花色牌；
    // 我们不希望 ACE 的值为 1，而是希望它为 14。
    // 因此，我们希望将随机数的范围从 1-13 转换为 2-14
    let min = 2;
    let max = 14;
    // 返回两个值之间的随机整数，包括这两个值
    return Math.floor(Math.random() * (max - min + 1) + min);
}

function newGameCards() {
    let cardOne = getRandomCard();
    let cardTwo = getRandomCard();
    let cardThree = getRandomCard();
    // 我们希望：
    // 1. cardOne 和 cardTwo 是不同的牌
    // 2. cardOne 小于 cardTwo
    // 因此，当 cardOne 大于或等于 cardTwo 时，
    // 我们将继续生成随机牌。
    while (cardOne >= cardTwo) {
        cardOne = getRandomCard();
        cardTwo = getRandomCard();
    }
    return [cardOne, cardTwo, cardThree];
}

// 获取牌的值的函数
function getCardValue(card) {
    let faceOrAce = {
        11: 'JACK',
        12: 'QUEEN',
        13: 'KING',
        14: 'ACE',
    };
    // 如果牌的值与 faceOrAce 中的键匹配，则使用 faceOrAce 的值；
    // 否则，返回 undefined 并使用空值合并运算符（??）处理，并默认为牌的值。
    let cardValue = faceOrAce[card] ?? card;
    return cardValue;
}

print(spaces(26) + 'ACEY DUCEY CARD GAME');
print(spaces(15) + 'CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n');
print('ACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER');
print('THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP');
print('YOU HAVE AN OPTION TO BET OR NOT BET DEPENDING');
print('ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE');
print('A VALUE BETWEEN THE FIRST TWO.');
print("IF YOU DO NOT WANT TO BET, INPUT '0'");

main();

}

// UTILITY FUNCTIONS
function isValidYesNoString(string) {
    return validLowerCaseYesAndNoStrings.includes(string.toLowerCase());
}

function isValidYesString(string) {
    return validLowerCaseYesStrings.includes(string.toLowerCase());
}

function isValidNoString(string) {
    return validLowerCaseNoStrings.includes(string.toLowerCase());
}

function print(string) {
    if (isRunningInBrowser) {
        // 添加尾随换行符以匹配 console.log 的行为
        document
            .getElementById('output')
            .appendChild(document.createTextNode(string + '\n'));
    } else {
        console.log(string);
    }
}

function input() {
    if (isRunningInBrowser) {
        // 从浏览器 DOM 输入接受输入
        return new Promise((resolve) => {
            const outputElement = document.querySelector('#output');
            const inputElement = document.createElement('input');
            outputElement.append(inputElement);
            inputElement.focus();

            inputElement.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') {
                    const result = inputElement.value;
                    inputElement.remove();
                    print(result);
                    print('');
                    resolve(result);
                }
            });
        });
    } else {
        // 从 Node.js 命令行接受输入
        // 参见：https://nodejs.dev/learn/accept-input-from-the-command-line-in-nodejs
        return new Promise(function (resolve) {
            const readline = require('readline').createInterface({
                input: process.stdin,
                output: process.stdout,
            });
            readline.question('', function (input) {
                resolve(input);
                readline.close();
            });
        });
    }
}

function printInline(string) {
    if (isRunningInBrowser) {
        document
            .getElementById('output')
            .appendChild(document.createTextNode(string));
    } else {
        process.stdout.write(string);
    }
}

function spaces(numberOfSpaces) {
    return ' '.repeat(numberOfSpaces);
}

// UTILITY FUNCTIONS

```