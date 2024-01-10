# `basic-computer-games\72_Queen\javascript\queen.js`

```
// 定义一个打印函数，将字符串添加到输出元素中
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

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入框的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
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
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一些变量
var sa = [,81,  71,  61,  51,  41,  31,  21,  11,
           92,  82,  72,  62,  52,  42,  32,  22,
          103,  93,  83,  73,  63,  53,  43,  33,
          114, 104,  94,  84,  74,  64,  54,  44,
          125, 115, 105,  95,  85,  75,  65,  55,
          136, 126, 116, 106,  96,  86,  76,  66,
          147, 137, 127, 117, 107,  97,  87,  77,
          158, 148, 138, 128, 118, 108,  98,  88];

var m;
var m1;
var u;
var t;
var u1;
var t1;
# 显示游戏说明
function show_instructions()
{
    # 打印游戏说明
    print("WE ARE GOING TO PLAY A GAME BASED ON ONE OF THE CHESS\n");
    print("MOVES.  OUR QUEEN WILL BE ABLE TO MOVE ONLY TO THE LEFT,\n");
    print("DOWN, OR DIAGONALLY DOWN AND TO THE LEFT.\n");
    print("\n");
    print("THE OBJECT OF THE GAME IS TO PLACE THE QUEEN IN THE LOWER\n");
    print("LEFT HAND SQUARE BY ALTERNATING MOVES BETWEEN YOU AND THE\n");
    print("COMPUTER.  THE FIRST ONE TO PLACE THE QUEEN THERE WINS.\n");
    print("\n");
    print("YOU GO FIRST AND PLACE THE QUEEN IN ANY ONE OF THE SQUARES\n");
    print("ON THE TOP ROW OR RIGHT HAND COLUMN.\n");
    print("THAT WILL BE YOUR FIRST MOVE.\n");
    print("WE ALTERNATE MOVES.\n");
    print("YOU MAY FORFEIT BY TYPING '0' AS YOUR MOVE.\n");
    print("BE SURE TO PRESS THE RETURN KEY AFTER EACH RESPONSE.\n");
    print("\n");
    print("\n");
}

# 显示游戏地图
function show_map()
{
    # 打印游戏地图
    print("\n");
    for (var a = 0; a <= 7; a++) {
        for (var b = 1; b <= 8; b++) {
            i = 8 * a + b;
            print(" " + sa[i] + " ");
        }
        print("\n");
        print("\n");
        print("\n");
    }
    print("\n");
}

# 检查移动是否有效
function test_move()
{
    m = 10 * t + u;
    if (m == 158 || m == 127 || m == 126 || m == 75 || m == 73)
        return true;
    return false;
}

# 随机移动
function random_move()
{
    # 随机移动
    z = Math.random();
    if (z > 0.6) {
        u = u1 + 1;
        t = t1 + 1;
    } else if (z > 0.3) {
        u = u1 + 1;
        t = t1 + 2;
    } else {
        u = u1;
        t = t1 + 1;
    }
    m = 10 * t + u;
}

# 计算机移动
function computer_move()
{
    if (m1 == 41 || m1 == 44 || m1 == 73 || m1 == 75 || m1 == 126 || m1 == 127) {
        random_move();
        return;
    }
    for (k = 7; k >= 1; k--) {
        u = u1;
        t = t1 + k;
        if (test_move())
            return;
        u += k;
        if (test_move())
            return;
        t += k;
        if (test_move())
            return;
    }
    random_move();
}

# 主程序
// 异步函数，程序的入口
async function main()
{
    // 打印QUEEN，并在前面加上33个空格
    print(tab(33) + "QUEEN\n");
    // 打印CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY，并在前面加上15个空格
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印三个空行
    print("\n");
    print("\n");
    print("\n");

    // 进入循环，直到用户输入YES或NO
    while (1) {
        // 提示用户是否需要说明
        print("DO YOU WANT INSTRUCTIONS");
        // 等待用户输入
        str = await input();
        // 如果用户输入YES或NO，则跳出循环
        if (str == "YES" || str == "NO")
            break;
        // 如果用户输入不是YES或NO，则提示用户重新输入
        print("PLEASE ANSWER 'YES' OR 'NO'.\n");
    }
    // 如果用户输入YES，则显示说明
    if (str == "YES")
        show_instructions();
    }
    // 打印一个空行
    print("\n");
    // 打印OK --- THANKS AGAIN.
    print("OK --- THANKS AGAIN.\n");
}

// 调用main函数
main();
```