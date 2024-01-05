# `d:/src/tocomm/basic-computer-games\51_Hurkle\javascript\hurkle.js`

```
// BATNUM
// 
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    // 在页面上输出文本
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    // 返回一个 Promise 对象
    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型为文本
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
# 结束函数 tab 的定义
}

# 定义函数 tab，接受一个参数 space
function tab(space)
{
    # 初始化字符串 str 为空
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

}

// Main program
async function main()
{
    print(tab(33) + "HURKLE\n");  // 在第33列打印"HURKLE"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在第15列打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    n = 5;  // 初始化变量n为5
    g = 10;  // 初始化变量g为10
    print("\n");  // 打印空行
    print("A HURKLE IS HIDING ON A " + g + " BY " + g + " GRID. HOMEBASE\n");  // 打印包含变量g的字符串
    print("ON THE GRID IS POINT 0,0 IN THE SOUTHWEST CORNER,\n");  // 打印指定位置的信息
    print("AND ANY POINT ON THE GRID IS DESIGNATED BY A\n");  // 打印指定位置的信息
    print("PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. THE FIRST\n");  // 打印指定位置的信息
    print("NUMBER IS THE HORIZONTAL POSITION AND THE SECOND NUMBER\n");  // 打印指定位置的信息
    # 打印提示信息，垂直位置
    print("IS THE VERTICAL POSITION. YOU MUST TRY TO\n");
    # 打印提示信息，猜测次数
    print("GUESS THE HURKLE'S GRIDPOINT. YOU GET " + n + " TRIES.\n");
    # 打印提示信息，每次猜测后的指引
    print("AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE\n");
    print("DIRECTION TO GO TO LOOK FOR THE HURKLE.\n");
    print("\n");
    # 进入无限循环
    while (1) {
        # 生成随机的a和b
        a = Math.floor(g * Math.random());
        b = Math.floor(g * Math.random());
        # 循环n次，进行猜测
        for (k = 1; k <= n; k++) {
            # 打印猜测次数
            print("GUESS #" + k + " ");
            # 等待用户输入
            str = await input();
            # 解析用户输入的x和y坐标
            x = parseInt(str);
            y = parseInt(str.substr(str.indexOf(",") + 1));
            # 如果猜中了，打印提示信息并跳出循环
            if (x == a && y == b) {
                print("\n");
                print("YOU FOUND HIM IN " + k + " GUESSES!\n");
                break;
            }
            # 打印提示信息，根据y和b的大小关系给出方向指引
            print("GO ");
            if (y < b) {
# 如果玩家猜测的纵坐标小于实际位置的纵坐标，则打印"NORTH"
            if (y < b) {
# 如果玩家猜测的纵坐标大于实际位置的纵坐标，则打印"SOUTH"
                print("NORTH");
            } else if (y > b) {
                print("SOUTH");
            }
# 如果玩家猜测的横坐标小于实际位置的横坐标，则打印"EAST"并换行
            if (x < a) {
                print("EAST\n");
# 如果玩家猜测的横坐标大于等于实际位置的横坐标，则打印"WEST"并换行
            } else {
                print("WEST\n");
            }
        }
# 如果玩家猜测的次数大于实际位置的次数，则打印换行符和提示信息
        if (k > n) {
            print("\n");
            print("SORRY, THAT'S " + n + " GUESSES.\n");
            print("THE HURKLE IS AT " + a + "," + b + "\n");
        }
# 打印换行符和提示信息，表示游戏结束
        print("\n");
        print("LET'S PLAY AGAIN, HURKLE IS HIDING.\n");
        print("\n");
    }
}
# 调用名为main的函数，但是在给定的代码中并没有定义这个函数，所以这行代码会导致错误。
```