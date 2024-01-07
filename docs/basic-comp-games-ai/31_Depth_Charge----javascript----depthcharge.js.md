# `basic-computer-games\31_Depth_Charge\javascript\depthcharge.js`

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

// 主程序，使用 async 函数定义，使用 await 来等待输入
async function main()
{
    // 输出标题
    print(tab(30) + "DEPTH CHARGE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 输出提示信息并获取输入
    print("DIMENSION OF THE SEARCH AREA");
    g = Math.floor(await input());
    n = Math.floor(Math.log(g) / Math.log(2)) + 1;
    print("YOU ARE THE CAPTAIN OF THE DESTROYER USS COMPUTER\n");
    print("AN ENEMY SUB HAS BEEN CAUSING YOU TROUBLE.  YOUR\n");
    print("MISSION IS TO DESTROY IT.  YOU HAVE " + n + " SHOTS.\n");
    print("SPECIFY DEPTH CHARGE EXPLOSION POINT WITH A\n");
    print("TRIO OF NUMBERS -- THE FIRST TWO ARE THE\n");
    print("SURFACE COORDINATES; THE THIRD IS THE DEPTH.\n");
    // 开始游戏循环
    do {
        print("\n");
        print("GOOD LUCK !\n");
        print("\n");
        // 生成随机的目标坐标
        a = Math.floor(Math.random() * g);
        b = Math.floor(Math.random() * g);
        c = Math.floor(Math.random() * g);
        // 进行游戏逻辑
        for (d = 1; d <= n; d++) {
            print("\n");
            print("TRIAL #" + d + " ");
            str = await input();
            x = parseInt(str);
            y = parseInt(str.substr(str.indexOf(",") + 1));
            z = parseInt(str.substr(str.lastIndexOf(",") + 1));
            if (Math.abs(x - a) + Math.abs(y - b) + Math.abs(z - c) == 0)
                break;
            if (y > b)
                print("NORTH");
            if (y < b)
                print("SOUTH");
            if (x > a)
                print("EAST");
            if (x < a)
                print("WEST");
            if (y != b || x != a)
                print(" AND");
            if (z > c)
                print(" TOO LOW.\n");
            if (z < c)
                print(" TOO HIGH.\n");
            if (z == c)
                print(" DEPTH OK.\n");
            print("\n");
        }
        // 根据游戏结果输出信息
        if (d <= n) {
            print("\n");
            print("B O O M ! ! YOU FOUND IT IN " + d + " TRIES!\n");
        } else {
            print("\n");
            print("YOU HAVE BEEN TORPEDOED!  ABANDON SHIP!\n");
            print("THE SUBMARINE WAS AT " + a + "," + b + "," + c + "\n");
        }
        print("\n");
        print("\n");
        print("ANOTHER GAME (Y OR N)");
        str = await input();
    } while (str.substr(0, 1) == "Y") ;
    print("OK.  HOPE YOU ENJOYED YOURSELF.\n");
}

// 调用主程序
main();

```