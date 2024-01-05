# `29_Craps\javascript\craps.js`

```
// 定义一个名为print的函数，用于向页面输出内容
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为input的函数，用于获取用户输入
function input()
{
    var input_element;
    var input_str;

    // 返回一个Promise对象，表示异步操作的最终完成或失败
    return new Promise(function (resolve) {
                       // 创建一个input元素
                       input_element = document.createElement("INPUT");

                       // 在页面输出问号提示
                       print("? ");
                       // 设置input元素的类型为文本
                       input_element.setAttribute("type", "text");
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

function roll()  // 定义名为roll的函数
{
    return Math.floor(6 * Math.random())+1 + Math.floor(6 * Math.random())+1;  // 返回两次骰子点数之和
}

// Main program  // 主程序部分的注释
async function main()  // 定义名为main的异步函数
{
    print(tab(33) + "CRAPS\n");  // 打印CRAPS
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
    print("\n");  // 打印换行
    print("\n");  // 打印换行
    print("\n");  // 打印换行
    r = 0;  // 初始化变量r为0
    print("2,3,12 ARE LOSERS: 4,5,6,8,9,10 ARE POINTS: 7,11 ARE NATURAL WINNERS.\n");  // 打印提示信息
    while (1) {  // 进入无限循环
        # 打印提示信息，要求输入赌注金额
        print("INPUT THE AMOUNT OF YOUR WAGER.");
        # 从输入中获取赌注金额并转换为整数
        f = parseInt(await input());
        # 打印提示信息，表示将要掷骰子
        print("I WILL NOW THROW THE DICE\n");
        # 调用掷骰子的函数，获取点数
        x = roll();
        # 判断点数是否为7或11，如果是则赢得赌注金额
        if (x == 7 || x == 11) {
            print(x + " - NATURAL....A WINNER!!!!\n");
            print(x + " PAYS EVEN MONEY, YOU WIN " + f + " DOLLARS\n");
            r += f;
        } else if (x == 2) {
            # 判断点数是否为2，如果是则输掉赌注金额
            print(x + " - SNAKE EYES....YOU LOSE.\n");
            print("YOU LOSE " + f + " DOLLARS.\n");
            r -= f;
        } else if (x == 3 || x == 12) { # 原始代码在第70行中进行了重复的比较
            # 判断点数是否为3或12，如果是则输掉赌注金额
            print(x + " - CRAPS....YOU LOSE.\n");
            print("YOU LOSE " + f + " DOLLARS.\n");
            r -= f;
        } else {
            # 如果点数不是以上情况，则继续掷骰子
            print(x + " IS THE POINT. I WILL ROLL AGAIN\n");
            while (1) {
                # 继续掷骰子，直到出现特定点数
                o = roll();
                if (o == 7) {  # 如果 o 等于 7
                    print(o + " - CRAPS, YOU LOSE.\n");  # 打印 o 和提示信息
                    print("YOU LOSE $" + f + "\n");  # 打印失去的金额
                    r -= f;  # 从总金额中减去失去的金额
                    break;  # 跳出循环
                }
                if (o == x) {  # 如果 o 等于 x
                    print(x + " - A WINNER.........CONGRATS!!!!!!!!\n");  # 打印 x 和祝贺信息
                    print(x + " AT 2 TO 1 ODDS PAYS YOU...LET ME SEE..." + 2 * f + " DOLLARS\n");  # 打印赢得的金额
                    r += f * 2;  # 总金额增加赢得的金额的两倍
                    break;  # 跳出循环
                }
                print(o + " - NO POINT. I WILL ROLL AGAIN\n");  # 打印 o 和提示信息
            }
        }
        print(" IF YOU WANT TO PLAY AGAIN PRINT 5 IF NOT PRINT 2");  # 打印提示信息
        m = parseInt(await input());  # 将输入的值转换为整数并赋值给 m
        if (r < 0) {  # 如果总金额小于 0
            print("YOU ARE NOW UNDER $" + -r + "\n");  # 打印欠款金额
        } else if (r > 0) {  # 如果总金额大于 0
# 主函数，程序的入口
function main() {
    # 初始化变量
    var r = 0;
    var m = 0;
    
    # 循环进行赌博游戏
    while (true) {
        # 生成一个随机数
        var i = Math.floor(Math.random() * 6) + 1;
        # 更新赌注
        r += i;
        # 更新游戏次数
        m++;
        
        # 根据赌注的变化输出不同的消息
        if (r > 0) {
            print("YOU ARE NOW AHEAD $" + r + "\n");
        } else {
            print("YOU ARE NOW EVEN AT 0\n");
        }
        
        # 如果游戏次数不等于5，则跳出循环
        if (m != 5)
            break;
    }
    
    # 根据赌注的情况输出不同的消息
    if (r < 0) {
        print("TOO BAD, YOU ARE IN THE HOLE. COME AGAIN.\n");
    } else if (r > 0) {
        print("CONGRATULATIONS---YOU CAME OUT A WINNER. COME AGAIN!\n");
    } else {
        print("CONGRATULATIONS---YOU CAME OUT EVEN, NOT BAD FOR AN AMATEUR\n");
    }
}

# 调用主函数
main();
```