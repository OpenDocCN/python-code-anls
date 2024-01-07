# `basic-computer-games\94_War\javascript\war.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space) {
    let str = "";
    while (space-- > 0) {
        str += " ";
    }
    return str;
}

// 定义一个输入函数，返回一个 Promise 对象，等待用户输入并解析结果
function input() {
    return new Promise(function (resolve) {
        // 创建一个文本输入框
        const input_element = document.createElement("INPUT");

        // 在输出区域打印提示符
        print("? ");
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");
        document.getElementById("output").appendChild(input_element);
        input_element.focus();
        // 监听用户输入，当按下回车键时解析输入结果并返回
        input_element.addEventListener("keydown", function (event) {
            if (event.keyCode == 13) {
                const input_str = input_element.value;
                document.getElementById("output").removeChild(input_element);
                print(input_str);
                print("\n");
                resolve(input_str);
            }
        });
    });
}

// 定义一个异步函数，询问用户是否要继续，返回一个布尔值
async function askYesOrNo(question) {
    while (1) {
        print(question);
        const str = (await input()).toUpperCase();
        if (str === "YES") {
            return true;
        }
        else if (str === "NO") {
            return false;
        }
        else {
            print("YES OR NO, PLEASE.  ");
        }
    }
}

// 定义一个异步函数，询问用户是否需要游戏说明
async function askAboutInstructions() {
    const playerWantsInstructions = await askYesOrNo("DO YOU WANT DIRECTIONS");
    if (playerWantsInstructions) {
        print("THE COMPUTER GIVES YOU AND IT A 'CARD'.  THE HIGHER CARD\n");
        print("(NUMERICALLY) WINS.  THE GAME ENDS WHEN YOU CHOOSE NOT TO\n");
        print("CONTINUE OR WHEN YOU HAVE FINISHED THE PACK.\n");
    }
    print("\n");
    print("\n");
}

// 创建游戏牌堆，返回一个包含指定数量卡牌索引的数组
function createGameDeck(cards, gameSize) {
    const deck = [];
    const deckSize = cards.length;
    for (let j = 0; j < gameSize; j++) {
        let card;

        // 计算一个新的卡牌索引，直到找到一个不在新牌堆中的索引
        do {
            card = Math.floor(deckSize * Math.random());
        } while (deck.includes(card));
        deck.push(card);
    }
    return deck;
}

// 计算卡牌的值，返回卡牌索引除以4的整数部分
function computeCardValue(cardIndex) {
    return Math.floor(cardIndex / 4);
}

// 打印游戏结束信息，包括玩家和电脑的得分
function printGameOver(playerScore, computerScore) {
    print("\n");
    print("\n");
    print(`WE HAVE RUN OUT OF CARDS.  FINAL SCORE:  YOU: ${playerScore}  THE COMPUTER: ${computerScore}\n`);
    print("\n");
}

// 打印游戏标题
function printTitle() {
    print(tab(33) + "WAR\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS THE CARD GAME OF WAR.  EACH CARD IS GIVEN BY SUIT-#\n");
    print("AS S-7 FOR SPADE 7.  ");
}

// 打印玩家和电脑的卡牌
function printCards(playerCard, computerCard) {
    print("\n");
    print(`YOU: ${playerCard}\tCOMPUTER: ${computerCard}\n`);
}

// 定义一副扑克牌
const cards = [
    "S-2", "H-2", "C-2", "D-2",
    // ... (省略了一些卡牌)
    "S-A", "H-A", "C-A", "D-A"
];

// 主控制部分，异步函数
async function main() {
    printTitle();
    await askAboutInstructions();

    let computerScore = 0;
    let playerScore = 0;

    // 生成一个随机的牌堆
    const gameSize = cards.length;    // 要洗入游戏牌堆的卡牌数量，可以小于等于总卡牌数量
    const deck = createGameDeck(cards, gameSize);
    let shouldContinuePlaying = true;

    while (deck.length > 0 && shouldContinuePlaying) {
        const playerCard = deck.shift();    // 拿一张卡牌
        const computerCard = deck.shift();    // 拿一张卡牌
        printCards(cards[playerCard], cards[computerCard]);

        const playerCardValue = computeCardValue(playerCard);
        const computerCardValue = computeCardValue(computerCard);
        if (playerCardValue < computerCardValue) {
            computerScore++;
            print("THE COMPUTER WINS!!! YOU HAVE " + playerScore + " AND THE COMPUTER HAS " + computerScore + "\n");
        } else if (playerCardValue > computerCardValue) {
            playerScore++;
            print("YOU WIN. YOU HAVE " + playerScore + " AND THE COMPUTER HAS " + computerScore + "\n");
        } else {
            print("TIE.  NO SCORE CHANGE.\n");
        }

        if (deck.length === 0) {
            printGameOver(playerScore, computerScore);
        }
        else {
            shouldContinuePlaying = await askYesOrNo("DO YOU WANT TO CONTINUE");
        }
    }
    print("THANKS FOR PLAYING.  IT WAS FUN.\n");
    print("\n");
}

// 调用主控制函数
main();

```