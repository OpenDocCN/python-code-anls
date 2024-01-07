# `basic-computer-games\51_Hurkle\javascript\hurkle.js`

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

                       // 打印提示符
                       print("? ");
                       // 设置输入框属性
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素中
                       document.getElementById("output").appendChild(input_element);
                       // 输入框获取焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的值并返回
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
    // 打印标题
    print(tab(33) + "HURKLE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    n = 5;
    g = 10;
    print("\n");
    // 打印游戏规则
    print("A HURKLE IS HIDING ON A " + g + " BY " + g + " GRID. HOMEBASE\n");
    print("ON THE GRID IS POINT 0,0 IN THE SOUTHWEST CORNER,\n");
    print("AND ANY POINT ON THE GRID IS DESIGNATED BY A\n");
    print("PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. THE FIRST\n");
    print("NUMBER IS THE HORIZONTAL POSITION AND THE SECOND NUMBER\n");
    print("IS THE VERTICAL POSITION. YOU MUST TRY TO\n");
    print("GUESS THE HURKLE'S GRIDPOINT. YOU GET " + n + " TRIES.\n");
    print("AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE\n");
    print("DIRECTION TO GO TO LOOK FOR THE HURKLE.\n");
    print("\n");
    // 游戏循环
    while (1) {
        a = Math.floor(g * Math.random());
        b = Math.floor(g * Math.random());
        for (k = 1; k <= n; k++) {
            print("GUESS #" + k + " ");
            // 等待用户输入
            str = await input();
            x = parseInt(str);
            y = parseInt(str.substr(str.indexOf(",") + 1));
            if (x == a && y == b) {
                print("\n");
                print("YOU FOUND HIM IN " + k + " GUESSES!\n");
                break;
            }
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
        if (k > n) {
            print("\n");
            print("SORRY, THAT'S " + n + " GUESSES.\n");
            print("THE HURKLE IS AT " + a + "," + b + "\n");
        }
        print("\n");
        print("LET'S PLAY AGAIN, HURKLE IS HIDING.\n");
        print("\n");
    }
}

// 调用主程序
main();

```