# `16_Bug\javascript\bug.js`

```
# BUG
# 
# Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
# 

# 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

# 定义一个输入函数，返回一个 Promise 对象
function input()
{
    return new Promise(function (resolve) {
                       # 创建一个 INPUT 元素
                       const input_element = document.createElement("INPUT');

                       # 在输出元素中打印问号
                       print("? ");
                       # 设置 INPUT 元素的类型为文本，长度为50
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       # 将 INPUT 元素添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       # 让 INPUT 元素获得焦点
                       input_element.focus();
# 为输入元素添加一个事件监听器，当按下键盘时触发
input_element.addEventListener("keydown",
    function (event) {
        # 如果按下的是回车键（键码为13）
        if (event.keyCode === 13) {
            # 获取输入框中的值
            const input_str = input_element.value;
            # 从文档中移除输入元素
            document.getElementById("output").removeChild(input_element);
            # 打印输入的字符串
            print(input_str);
            # 打印换行符
            print("\n");
            # 返回输入的字符串
            resolve(input_str);
        }
    });
});

# 定义一个函数，用于生成指定数量空格的字符串
function tab(space)
{
    # 初始化一个空字符串
    let str = "";
    # 当 space 大于 0 时，循环添加空格到字符串中
    while (space-- > 0)
        str += " ";
    # 返回生成的字符串
    return str;
}
# 定义一个名为waitNSeconds的函数，接受一个参数n，返回一个Promise对象，延迟n秒后解析
function waitNSeconds(n) {
    return new Promise(resolve => setTimeout(resolve, n*1000));
}

# 定义一个名为scrollToBottom的函数，将窗口滚动到页面底部
function scrollToBottom() {
    window.scrollTo(0, document.body.scrollHeight);
}

# 定义一个名为draw_head的函数，用于绘制一个头部的ASCII图形
function draw_head()
{
    print("        HHHHHHH\n");
    print("        H     H\n");
    print("        H O O H\n");
    print("        H     H\n");
    print("        H  V  H\n");
    print("        HHHHHHH\n");
}

# 定义一个名为drawFeelers的函数，接受两个参数feelerCount和character，用于绘制触角的ASCII图形
function drawFeelers(feelerCount, character) {
# 循环4次，打印10个空格
    for (let z = 1; z <= 4; z++) {
        print(tab(10));
        # 循环feelerCount次，打印character和空格
        for (let x = 1; x <= feelerCount; x++) {
            print(character + " ");
        }
        # 换行
        print("\n");
    }
}

# 绘制颈部
function drawNeck() {
    # 循环2次，打印"N N"并换行
    for (let z = 1; z <= 2; z++)
        print("          N N\n");
}

# 绘制身体
function drawBody(computerTailCount) {
    # 打印"BBBBBBBBBBBBB"
    print("     BBBBBBBBBBBB\n");
    # 循环2次，打印"B          B"并换行
    for (let z = 1; z <= 2; z++)
        print("     B          B\n");
    # 如果computerTailCount等于1，则打印"TTTTTB          B"
    if (computerTailCount === 1)
        print("TTTTTB          B\n");
}
    print("     BBBBBBBBBBBB\n");  # 打印字符串 "     BBBBBBBBBBBB\n"

def drawFeet(computerFeetCount):
    for (let z = 1; z <= 2; z++) {  # 循环两次
        print(tab(5));  # 打印5个空格
        for (let x = 1; x <= computerFeetCount; x++)  # 循环computerFeetCount次
            print(" L");  # 打印 " L"
        print("\n");  # 换行

function drawBug(playerFeelerCount, playerHeadCount, playerNeckCount, playerBodyCount, playerTailCount, playerFeetCount, feelerCharacter):
    if (playerFeelerCount !== 0):  # 如果playerFeelerCount不等于0
        drawFeelers(playerFeelerCount, feelerCharacter)  # 调用drawFeelers函数
    if (playerHeadCount !== 0):  # 如果playerHeadCount不等于0
        draw_head()  # 调用draw_head函数
    if (playerNeckCount !== 0):  # 如果playerNeckCount不等于0
        drawNeck()  # 调用drawNeck函数
    }  // 结束 if 语句的代码块

    if (playerBodyCount !== 0) {  // 如果 playerBodyCount 不等于 0
        drawBody(playerTailCount)  // 调用 drawBody 函数，传入 playerTailCount 参数
    }

    if (playerFeetCount !== 0) {  // 如果 playerFeetCount 不等于 0
        drawFeet(playerFeetCount);  // 调用 drawFeet 函数，传入 playerFeetCount 参数
    }

    for (let z = 1; z <= 4; z++)  // 循环，z 从 1 到 4
        print("\n");  // 打印换行符

}

// Main program
async function main()  // 异步函数 main
{
    print(tab(34) + "BUG\n");  // 调用 tab 函数，传入参数 34，然后与 "BUG\n" 拼接并打印
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 调用 tab 函数，传入参数 15，然后与 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n" 拼接并打印
    print("\n");  // 打印换行符
    print("\n");  // 打印换行符
    print("\n");  // 打印换行符
    let playerFeelerCount = 0;  // 声明并初始化变量 playerFeelerCount 为 0
    // 初始化玩家和电脑各个部位的计数器
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

    // 打印游戏标题和欢迎语
    print("THE GAME BUG\n");
    print("I HOPE YOU ENJOY THIS GAME.\n");
    print("\n");
    print("DO YOU WANT INSTRUCTIONS");
    // 等待用户输入是否需要游戏说明
    const instructionsRequired = await input();
    // 如果用户需要游戏说明，则打印游戏目标
    if (instructionsRequired.toUpperCase() !== "NO") {
        print("THE OBJECT OF BUG IS TO FINISH YOUR BUG BEFORE I FINISH\n");
        # 打印游戏规则说明
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

    # 设置游戏进行中的标志为真
    let gameInProgress = true;
    while (gameInProgress) {  # 当游戏正在进行时
        let dieRoll = Math.floor(6 * Math.random() + 1);  # 生成1到6之间的随机整数，模拟掷骰子
        let partFound = false;  # 初始化部位未找到标志为假
        print("YOU ROLLED A " + dieRoll + "\n");  # 打印掷骰子的结果

        switch (dieRoll) {  # 根据掷骰子的结果进行不同的操作
            case 1:  # 如果掷骰子结果为1
                print("1=BODY\n");  # 打印身体部位
                if (playerBodyCount === 0) {  # 如果玩家还没有身体部位
                    print("YOU NOW HAVE A BODY.\n");  # 打印获得身体部位的消息
                    playerBodyCount = 1;  # 玩家身体部位数量加1
                    partFound = true;  # 设置部位找到标志为真
                } else {
                    print("YOU DO NOT NEED A BODY.\n");  # 打印不需要身体部位的消息
                }
                break;  # 结束case
            case 2:  # 如果掷骰子结果为2
                print("2=NECK\n");  # 打印颈部部位
                if (playerNeckCount === 0) {  # 如果玩家还没有颈部部位
                    if (playerBodyCount === 0) {  # 如果玩家还没有身体部位
                        print("YOU DO NOT HAVE A BODY.\n");  # 打印没有身体部位的消息
                } else {
                    print("YOU NOW HAVE A NECK.\n");  # 如果条件不满足，打印提示信息
                    playerNeckCount = 1;  # 将playerNeckCount设置为1
                    partFound = true;  # 将partFound设置为true
                } else {
                    print("YOU DO NOT NEED A NECK.\n");  # 如果条件不满足，打印提示信息
                }
                break;  # 结束switch语句
            case 3:  # 如果switch的值为3
                print("3=HEAD\n");  # 打印提示信息
                if (playerNeckCount === 0) {  # 如果playerNeckCount等于0
                    print("YOU DO NOT HAVE A NECK.\n");  # 打印提示信息
                } else if (playerHeadCount === 0) {  # 如果playerHeadCount等于0
                    print("YOU NEEDED A HEAD.\n");  # 打印提示信息
                    playerHeadCount = 1;  # 将playerHeadCount设置为1
                    partFound = true;  # 将partFound设置为true
                } else {
                    print("YOU HAVE A HEAD.\n");  # 如果条件不满足，打印提示信息
                }
                break;  # 结束当前的 case 分支
            case 4:  # 如果 switch 的表达式的值等于 4，则执行以下代码
                print("4=FEELERS\n");  # 打印输出 "4=FEELERS"
                if (playerHeadCount === 0) {  # 如果玩家的头部计数为 0
                    print("YOU DO NOT HAVE A HEAD.\n");  # 打印输出 "YOU DO NOT HAVE A HEAD."
                } else if (playerFeelerCount === 2) {  # 如果玩家的触角计数为 2
                    print("YOU HAVE TWO FEELERS ALREADY.\n");  # 打印输出 "YOU HAVE TWO FEELERS ALREADY."
                } else {
                    print("I NOW GIVE YOU A FEELER.\n");  # 打印输出 "I NOW GIVE YOU A FEELER."
                    playerFeelerCount ++;  # 玩家的触角计数加一
                    partFound = true;  # 将 partFound 标记为 true
                }
                break;  # 结束当前的 case 分支
            case 5:  # 如果 switch 的表达式的值等于 5，则执行以下代码
                print("5=TAIL\n");  # 打印输出 "5=TAIL"
                if (playerBodyCount === 0) {  # 如果玩家的身体计数为 0
                    print("YOU DO NOT HAVE A BODY.\n");  # 打印输出 "YOU DO NOT HAVE A BODY."
                } else if (playerTailCount === 1) {  # 如果玩家的尾巴计数为 1
                    print("YOU ALREADY HAVE A TAIL.\n");  # 打印输出 "YOU ALREADY HAVE A TAIL."
                } else {
                    # 打印消息，表示给玩家一个尾巴
                    print("I NOW GIVE YOU A TAIL.\n");
                    # 玩家尾巴数量加一
                    playerTailCount++;
                    # 设置找到身体部位的标志为真
                    partFound = true;
                }
                # 结束当前情况
                break;
            # 如果情况为6
            case 6:
                # 打印消息，表示给玩家一条腿
                print("6=LEG\n");
                # 如果玩家已经有6条腿
                if (playerFeetCount === 6) {
                    # 打印消息，表示玩家已经有6条腿
                    print("YOU HAVE 6 FEET ALREADY.\n");
                } 
                # 如果玩家没有身体
                else if (playerBodyCount === 0) {
                    # 打印消息，表示玩家没有身体
                    print("YOU DO NOT HAVE A BODY.\n");
                } 
                # 如果玩家还没有6条腿且有身体
                else {
                    # 玩家腿数量加一
                    playerFeetCount++;
                    # 设置找到身体部位的标志为真
                    partFound = true;
                    # 打印消息，表示玩家现在有多少条腿
                    print("YOU NOW HAVE " + playerFeetCount + " LEGS.\n");
                }
                # 结束当前情况
                break;
        }
        # 通过随机数生成1到6的整数
        dieRoll = Math.floor(6 * Math.random() + 1) ;
        # 打印换行符
        print("\n");
        # 滚动到页面底部
        scrollToBottom();
        # 等待1秒
        await waitNSeconds(1);

        # 打印掷骰子的结果
        print("I ROLLED A " + dieRoll + "\n");
        # 根据骰子的结果进行不同的操作
        switch (dieRoll) {
            # 如果骰子结果为1
            case 1:
                # 打印身体部位
                print("1=BODY\n");
                # 如果已经有一个身体部位
                if (computerBodyCount === 1) {
                    # 打印不需要身体部位
                    print("I DO NOT NEED A BODY.\n");
                } else {
                    # 打印现在有了身体部位
                    print("I NOW HAVE A BODY.\n");
                    # 标记找到了身体部位
                    partFound = true;
                    # 计算机身体部位数量加1
                    computerBodyCount = 1;
                }
                break;
            # 如果骰子结果为2
            case 2:
                # 打印颈部部位
                print("2=NECK\n");
                # 如果已经有一个颈部部位
                if (computerNeckCount === 1) {
                    # 打印不需要颈部部位
                    print("I DO NOT NEED A NECK.\n");
                } else if (computerBodyCount === 0) {
# 打印"I DO NOT HAVE A BODY.\n"字符串
                    print("I DO NOT HAVE A BODY.\n");
                # 如果条件不满足，则执行下面的语句
                } else {
                    # 打印"I NOW HAVE A NECK.\n"字符串
                    print("I NOW HAVE A NECK.\n");
                    # 将computerNeckCount赋值为1
                    computerNeckCount = 1;
                    # 将partFound赋值为true
                    partFound = true;
                # 结束switch语句
                }
                # 跳出当前循环
                break;
            # 当switch表达式的值为3时执行下面的语句
            case 3:
                # 打印"3=HEAD\n"字符串
                print("3=HEAD\n");
                # 如果条件不满足，则执行下面的语句
                if (computerNeckCount === 0) {
                    # 打印"I DO NOT HAVE A NECK.\n"字符串
                    print("I DO NOT HAVE A NECK.\n");
                # 如果条件不满足，则执行下面的语句
                } else if (computerHeadCount === 1) {
                    # 打印"I DO NOT NEED A HEAD.\n"字符串
                    print("I DO NOT NEED A HEAD.\n");
                # 如果条件不满足，则执行下面的语句
                } else {
                    # 打印"I NEEDED A HEAD.\n"字符串
                    print("I NEEDED A HEAD.\n");
                    # 将computerHeadCount赋值为1
                    computerHeadCount = 1;
                    # 将partFound赋值为true
                    partFound = true;
                # 结束switch语句
                }
                # 跳出当前循环
                break;
            # 当switch表达式的值为4时执行下面的语句
                print("4=FEELERS\n");  # 打印信息，表示正在处理感觉器官
                if (computerHeadCount === 0) {  # 如果计算机的头部数量为0
                    print("I DO NOT HAVE A HEAD.\n");  # 打印信息，表示计算机没有头部
                } else if (computerFeelerCount === 2) {  # 否则，如果计算机的感觉器官数量为2
                    print("I HAVE 2 FEELERS ALREADY.\n");  # 打印信息，表示计算机已经有2个感觉器官
                } else {  # 否则
                    print("I GET A FEELER.\n");  # 打印信息，表示计算机获得一个感觉器官
                    computerFeelerCount++;  # 感觉器官数量加1
                    partFound = true;  # 找到了部件，设为true
                }
                break;  # 跳出switch语句
            case 5:
                print("5=TAIL\n");  # 打印信息，表示正在处理尾部
                if (computerBodyCount === 0) {  # 如果计算机的身体数量为0
                    print("I DO NOT HAVE A BODY.\n");  # 打印信息，表示计算机没有身体
                } else if (computerTailCount === 1) {  # 否则，如果计算机的尾部数量为1
                    print("I DO NOT NEED A TAIL.\n");  # 打印信息，表示计算机不需要尾部
                } else {  # 否则
                    print("I NOW HAVE A TAIL.\n");  # 打印信息，表示计算机现在有一个尾部
                    computerTailCount = 1;  # 尾部数量设为1
                partFound = true;  # 设置 partFound 变量为 true
                break;  # 跳出 switch 语句
            case 6:  # 如果 switch 变量的值为 6
                print("6=LEGS\n");  # 打印 "6=LEGS\n"
                if (computerFeetCount === 6) {  # 如果 computerFeetCount 等于 6
                    print("I HAVE 6 FEET.\n");  # 打印 "I HAVE 6 FEET.\n"
                } else if (computerBodyCount === 0) {  # 否则如果 computerBodyCount 等于 0
                    print("I DO NOT HAVE A BODY.\n");  # 打印 "I DO NOT HAVE A BODY.\n"
                } else {  # 否则
                    computerFeetCount++;  # computerFeetCount 加一
                    partFound = true;  # 设置 partFound 变量为 true
                    print("I NOW HAVE " + computerFeetCount + " LEGS.\n");  # 打印 "I NOW HAVE " + computerFeetCount + " LEGS.\n"
                }
                break;  # 跳出 switch 语句
        }
        if (playerFeelerCount === 2 && playerTailCount === 1 && playerFeetCount === 6) {  # 如果 playerFeelerCount 等于 2 并且 playerTailCount 等于 1 并且 playerFeetCount 等于 6
            print("YOUR BUG IS FINISHED.\n");  # 打印 "YOUR BUG IS FINISHED.\n"
            gameInProgress = false;  # 设置 gameInProgress 变量为 false
        }
        if (computerFeelerCount === 2 && computerBodyCount === 1 && computerFeetCount === 6) {
            // 如果计算机的触角数量为2，身体数量为1，脚的数量为6，则打印消息并将游戏状态设置为结束
            print("MY BUG IS FINISHED.\n");
            gameInProgress = false;
        }
        // 如果没有找到部件，则继续循环
        if (!partFound)
            continue;
        // 打印消息询问是否要查看图片
        print("DO YOU WANT THE PICTURES");
        // 等待输入，并将结果存储在showPictures变量中
        const showPictures = await input();
        // 如果输入的内容转换为大写后等于"NO"，则继续循环
        if (showPictures.toUpperCase() === "NO")
            continue;
        // 打印玩家的虫子图案
        print("*****YOUR BUG*****\n");
        print("\n");
        print("\n");
        drawBug(playerFeelerCount, playerHeadCount, playerNeckCount, playerBodyCount, playerTailCount, playerFeetCount, "A");
        // 打印计算机的虫子图案
        print("*****MY BUG*****\n");
        print("\n");
        print("\n");
        drawBug(computerFeelerCount, computerHeadCount, computerNeckCount, computerBodyCount, computerTailCount, computerFeetCount, "F");
        // 打印4行空行
        for (let z = 1; z <= 4; z++)
            print("\n");
    }
    # 打印游戏结束的提示信息
    print("I HOPE YOU ENJOYED THE GAME, PLAY IT AGAIN SOON!!\n");
    # 滚动到页面底部
    scrollToBottom();
}

# 调用主函数
main();
```