# `basic-computer-games\51_Hurkle\javascript\hurkle.js`

```py
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
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，将输入的值添加到输出元素中，并解析 Promise
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

// 定义一个制表符函数，返回指定数量的空格字符串
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
    // 打印游戏名称和信息
    print(tab(33) + "HURKLE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化变量 n 和 g
    n = 5;
    g = 10;
    print("\n");
    // 打印游戏提示信息
    print("A HURKLE IS HIDING ON A " + g + " BY " + g + " GRID. HOMEBASE\n");
    print("ON THE GRID IS POINT 0,0 IN THE SOUTHWEST CORNER,\n");
    print("AND ANY POINT ON THE GRID IS DESIGNATED BY A\n");
}
    # 打印游戏规则提示信息
    print("PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. THE FIRST\n");
    print("NUMBER IS THE HORIZONTAL POSITION AND THE SECOND NUMBER\n");
    print("IS THE VERTICAL POSITION. YOU MUST TRY TO\n");
    print("GUESS THE HURKLE'S GRIDPOINT. YOU GET " + n + " TRIES.\n");
    print("AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE\n");
    print("DIRECTION TO GO TO LOOK FOR THE HURKLE.\n");
    print("\n");
    # 进入游戏循环
    while (1) {
        # 随机生成目标点的横纵坐标
        a = Math.floor(g * Math.random());
        b = Math.floor(g * Math.random());
        # 循环进行猜测
        for (k = 1; k <= n; k++) {
            # 打印猜测次数提示
            print("GUESS #" + k + " ");
            # 获取用户输入的猜测坐标
            str = await input();
            x = parseInt(str);
            y = parseInt(str.substr(str.indexOf(",") + 1));
            # 判断猜测结果
            if (x == a && y == b) {
                # 猜中目标点，打印提示信息并结束游戏
                print("\n");
                print("YOU FOUND HIM IN " + k + " GUESSES!\n");
                break;
            }
            # 未猜中目标点，根据猜测坐标和目标坐标打印方向提示信息
            print("GO ");
            if (y < b) {
                print("NORTH");
            } else if (y > b) {
                print("SOUTH");
            }
            if (x < a) {
                print("EAST\n");
            } else {
                print("WEST\n");
            }
        }
        # 判断是否超过最大猜测次数
        if (k > n) {
            # 超过最大猜测次数，打印提示信息并展示目标点坐标
            print("\n");
            print("SORRY, THAT'S " + n + " GUESSES.\n");
            print("THE HURKLE IS AT " + a + "," + b + "\n");
        }
        # 打印提示信息，重新开始游戏
        print("\n");
        print("LET'S PLAY AGAIN, HURKLE IS HIDING.\n");
        print("\n");
    }
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```