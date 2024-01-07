# `basic-computer-games\66_Number\javascript\number.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
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
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，将输入的值传递给 resolve 函数
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
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

// 主程序，使用 async 函数定义
async function main()
{
    // 输出标题和介绍信息
    print(tab(33) + "NUMBER\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU\n");
    print("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO\n");
    print("A RANDOM NUMBER SELECTED BY THE COMPUTER.\n");
    print("\n");
    print("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)\n");
    print("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.\n");
    print("\n");
    // 初始化变量 p
    p = 0;
    // 循环进行游戏
    while (1) {
        do {
            // 提示用户猜一个数字
            print("GUESS A NUMBER FROM 1 TO 5");
            // 获取用户输入的数字
            g = parseInt(await input());
        } while (g < 1 || g > 5) ;
        // 生成随机数
        r = Math.floor(5 * Math.random() + 1);
        s = Math.floor(5 * Math.random() + 1);
        t = Math.floor(5 * Math.random() + 1);
        u = Math.floor(5 * Math.random() + 1);
        v = Math.floor(5 * Math.random() + 1);
        // 根据用户猜的数字和随机数进行判断，更新得分
        if (g == r) {
            p -= 5;
        } else if (g == s) {
            p += 5;
        } else if (g == t) {
            p += p;
            print("YOU HIT THE JACKPOT!!!\n");
        } else if (g == u) {
            p += 1;
        } else if (g == v) {
            p -= p * 0.5;
        }
        // 判断得分是否达到 500 分，如果是则结束游戏
        if (p <= 500) {
            print("YOU HAVE " + p + " POINTS.\n");
            print("\n");
        } else {
            print("!!!!YOU WIN!!!! WITH " + p + " POINTS.\n");
            break;
        }
    }
}

// 调用主程序
main();

```