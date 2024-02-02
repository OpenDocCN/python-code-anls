# `basic-computer-games\50_Horserace\javascript\horserace.js`

```py
// 定义一个打印函数，将字符串输出到指定的元素上
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素上
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一些数组变量
var sa = [];
var ws = [];
var da = [];
var qa = [];
var pa = [];
var ma = [];
var ya = [];
var vs = [];

// 主程序
async function main()
{
    // 输出标题
    print(tab(31) + "HORSERACE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("WELCOME TO SOUTH PORTLAND HIGH RACETRACK\n");
    print("                      ...OWNED BY LAURIE CHEVALIER\n");
    # 打印询问是否需要指南
    print("DO YOU WANT DIRECTIONS");
    # 等待用户输入
    str = await input();
    # 如果用户输入为"YES"，则打印游戏规则说明
    if (str == "YES") {
        print("UP TO 10 MAY PLAY.  A TABLE OF ODDS WILL BE PRINTED.  YOU\n");
        print("MAY BET ANY + AMOUNT UNDER 100000 ON ONE HORSE.\n");
        print("DURING THE RACE, A HORSE WILL BE SHOWN BY ITS\n");
        print("NUMBER.  THE HORSES RACE DOWN THE PAPER!\n");
        print("\n");
    }
    # 打印询问有多少人想下注
    print("HOW MANY WANT TO BET");
    # 将用户输入的下注人数转换为整数
    c = parseInt(await input());
    # 打印提示用户在什么时候输入名字
    print("WHEN ? APPEARS,TYPE NAME\n");
    # 循环，让每个下注的人输入名字
    for (a = 1; a <= c; a++) {
        ws[a] = await input();
    }
    # 循环结束后，如果用户仍然需要指南，则继续循环
    } while (str == "YES") ;
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```