# `basic-computer-games\31_Depth_Charge\javascript\depthcharge.js`

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
                       // 设置输入框类型为文本，长度为50
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise 对象
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

// 主程序
async function main()
{
    // 打印游戏标题
    print(tab(30) + "DEPTH CHARGE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印提示信息
    print("DIMENSION OF THE SEARCH AREA");
    // 获取输入的搜索区域维度
    g = Math.floor(await input());
    // 计算搜索区域的维度对应的 n 值
    n = Math.floor(Math.log(g) / Math.log(2)) + 1;
    // 打印提示信息
    print("YOU ARE THE CAPTAIN OF THE DESTROYER USS COMPUTER\n");
}
    # 打印提示信息，通知玩家敌方潜艇正在造成麻烦
    print("AN ENEMY SUB HAS BEEN CAUSING YOU TROUBLE.  YOUR\n");
    # 打印提示信息，通知玩家任务是摧毁敌方潜艇，并给出可用的射击次数
    print("MISSION IS TO DESTROY IT.  YOU HAVE " + n + " SHOTS.\n");
    # 打印提示信息，要求玩家输入深度炸弹爆炸点的坐标
    print("SPECIFY DEPTH CHARGE EXPLOSION POINT WITH A\n");
    print("TRIO OF NUMBERS -- THE FIRST TWO ARE THE\n");
    print("SURFACE COORDINATES; THE THIRD IS THE DEPTH.\n");
    # 进入游戏循环
    do {
        # 打印祝福信息
        print("\n");
        print("GOOD LUCK !\n");
        print("\n");
        # 生成随机的潜艇坐标
        a = Math.floor(Math.random() * g);
        b = Math.floor(Math.random() * g);
        c = Math.floor(Math.random() * g);
        # 循环进行玩家的射击操作
        for (d = 1; d <= n; d++) {
            # 打印当前射击的次数
            print("\n");
            print("TRIAL #" + d + " ");
            # 等待玩家输入坐标
            str = await input();
            # 解析玩家输入的坐标
            x = parseInt(str);
            y = parseInt(str.substr(str.indexOf(",") + 1));
            z = parseInt(str.substr(str.lastIndexOf(",") + 1));
            # 判断玩家是否击中潜艇
            if (Math.abs(x - a) + Math.abs(y - b) + Math.abs(z - c) == 0)
                break;
            # 根据玩家输入的坐标与潜艇坐标的关系，给出提示信息
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
        # 判断玩家是否成功摧毁潜艇
        if (d <= n) {
            print("\n");
            print("B O O M ! ! YOU FOUND IT IN " + d + " TRIES!\n");
        } else {
            print("\n");
            print("YOU HAVE BEEN TORPEDOED!  ABANDON SHIP!\n");
            print("THE SUBMARINE WAS AT " + a + "," + b + "," + c + "\n");
        }
        # 询问玩家是否继续游戏
        print("\n");
        print("\n");
        print("ANOTHER GAME (Y OR N)");
        str = await input();
    } while (str.substr(0, 1) == "Y") ;
    # 打印结束游戏的提示信息
    print("OK.  HOPE YOU ENJOYED YOURSELF.\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```