# `basic-computer-games\16_Bug\javascript\bug.js`

```
// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    return new Promise(function (resolve) {
                       // 创建一个输入元素
                       const input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入元素的类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入元素添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入元素获得焦点
                       input_element.focus();
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown",
                           function (event) {
                                      // 如果按下的是回车键
                                      if (event.keyCode === 13) {
                                          // 获取输入的字符串
                                          const input_str = input_element.value;
                                          // 移除输入元素
                                          document.getElementById("output").removeChild(input_element);
                                          // 打印输入的字符串
                                          print(input_str);
                                          // 打印换行符
                                          print("\n");
                                          // 解析 Promise 对象
                                          resolve(input_str);
                                      }
                                  });
                       });
}

// 定义一个生成指定数量空格的函数
function tab(space)
{
    let str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个等待指定秒数的函数
function waitNSeconds(n) {
    return new Promise(resolve => setTimeout(resolve, n*1000));
}

// 定义一个滚动到页面底部的函数
function scrollToBottom() {
    window.scrollTo(0, document.body.scrollHeight);
}

// 定义一个绘制头部的函数
function draw_head()
{
    print("        HHHHHHH\n");
    print("        H     H\n");
    print("        H O O H\n");
    print("        H     H\n");
    print("        H  V  H\n");
    print("        HHHHHHH\n");
}

// 定义一个绘制触角的函数
function drawFeelers(feelerCount, character) {
    for (let z = 1; z <= 4; z++) {
        print(tab(10));
        for (let x = 1; x <= feelerCount; x++) {
            print(character + " ");
        }
        print("\n");
    # 代码块结束
// 画出虫子的脖子
function drawNeck() {
    // 循环两次，打印出虫子的脖子
    for (let z = 1; z <= 2; z++)
        print("          N N\n");
}

// 画出虫子的身体
function drawBody(computerTailCount) {
    // 打印出虫子的身体
    print("     BBBBBBBBBBBB\n");
    // 循环两次，打印出虫子的身体
    for (let z = 1; z <= 2; z++)
        print("     B          B\n");
    // 如果计算机的尾巴数量为1，则打印出相应的尾巴
    if (computerTailCount === 1)
        print("TTTTTB          B\n");
    // 打印出虫子的身体
    print("     BBBBBBBBBBBB\n");
}

// 画出虫子的脚
function drawFeet(computerFeetCount) {
    // 循环两次，打印出虫子的脚
    for (let z = 1; z <= 2; z++) {
        // 打印出制表符
        print(tab(5));
        // 循环计算机脚的数量，打印出相应数量的脚
        for (let x = 1; x <= computerFeetCount; x++)
            print(" L");
        // 打印换行符
        print("\n");
    }
}

// 画出完整的虫子
function drawBug(playerFeelerCount, playerHeadCount, playerNeckCount, playerBodyCount, playerTailCount, playerFeetCount, feelerCharacter) {
    // 如果玩家的触角数量不为0，则画出触角
    if (playerFeelerCount !== 0) {
        drawFeelers(playerFeelerCount, feelerCharacter);
    }
    // 如果玩家的头部数量不为0，则画出头部
    if (playerHeadCount !== 0)
        draw_head();
    // 如果玩家的脖子数量不为0，则画出脖子
    if (playerNeckCount !== 0) {
        drawNeck();
    }
    // 如果玩家的身体数量不为0，则画出身体
    if (playerBodyCount !== 0) {
        drawBody(playerTailCount)
    }
    // 如果玩家的脚数量不为0，则画出脚
    if (playerFeetCount !== 0) {
        drawFeet(playerFeetCount);
    }
    // 打印四个换行符
    for (let z = 1; z <= 4; z++)
        print("\n");
}

// 主程序
async function main()
{
    // 打印出游戏标题
    print(tab(34) + "BUG\n");
    // 打印出游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印出四个换行符
    print("\n");
    print("\n");
    print("\n");
    // 初始化玩家和计算机的各个身体部位数量
    let playerFeelerCount = 0;
    let playerHeadCount = 0;
    let playerNeckCount = 0;
    let playerBodyCount = 0;
    let playerFeetCount = 0;
    let playerTailCount = 0;

    let computerFeelerCount = 0;
    let computerHeadCount = 0;
    let computerNeckCount = 0;
    let computerBodyCount = 0;
    let computerTailCount = 0;
    let computerFeetCount = 0;

    // 打印出游戏标题
    print("THE GAME BUG\n");
    // 打印出游戏提示信息
    print("I HOPE YOU ENJOY THIS GAME.\n");
    // 打印出四个换行符
    print("\n");
    // 打印出是否需要游戏说明的提示信息，并等待输入
    print("DO YOU WANT INSTRUCTIONS");
    const instructionsRequired = await input();
}
    // 如果需要说明书，则打印游戏说明
    if (instructionsRequired.toUpperCase() !== "NO") {
        print("THE OBJECT OF BUG IS TO FINISH YOUR BUG BEFORE I FINISH\n");
        print("MINE. EACH NUMBER STANDS FOR A PART OF THE BUG BODY.\n");
        print("I WILL ROLL THE DIE FOR YOU, TELL YOU WHAT I ROLLED FOR YOU\n");
        print("WHAT THE NUMBER STANDS FOR, AND IF YOU CAN GET THE PART.\n");
        print("IF YOU CAN GET THE PART I WILL GIVE IT TO YOU.\n");
        print("THE SAME WILL HAPPEN ON MY TURN.\n");
        print("IF THERE IS A CHANGE IN EITHER BUG I WILL GIVE YOU THE\n");
        print("OPTION OF SEEING THE PICTURES OF THE BUGS.\n");
        print("THE NUMBERS STAND FOR PARTS AS FOLLOWS:\n");
        print("NUMBER\tPART\tNUMBER OF PART NEEDED\n");
        print("1\tBODY\t1\n");
        print("2\tNECK\t1\n");
        print("3\tHEAD\t1\n");
        print("4\tFEELERS\t2\n");
        print("5\tTAIL\t1\n");
        print("6\tLEGS\t6\n");
        print("\n");
        print("\n");
    }

    // 设置游戏进行中的标志为 true
    let gameInProgress = true;
    }
    // 打印游戏结束语
    print("I HOPE YOU ENJOYED THE GAME, PLAY IT AGAIN SOON!!\n");
    // 滚动到页面底部
    scrollToBottom();
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```