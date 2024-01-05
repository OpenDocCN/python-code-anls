# `62_Mugwump\javascript\mugwump.js`

```
# 定义函数print，用于在页面上输出字符串
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 定义函数input，用于获取用户输入
def input():
    # 声明变量
    var input_element
    var input_str

    # 返回一个Promise对象，用于异步处理用户输入
    return new Promise(function (resolve) {
        # 创建一个input元素
        input_element = document.createElement("INPUT")
        # 在页面上输出提示符
        print("? ")
        # 设置input元素的类型为文本
        input_element.setAttribute("type", "text")
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
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

var p = [];  // 创建一个空数组

// Main program
async function main()
{
    print(tab(33) + "MUGWUMP\n");  // 打印带有缩进的字符串"MUGWUMP"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有缩进的字符串"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    // Courtesy People's Computer Company
    print("THE OBJECT OF THIS GAME IS TO FIND FOUR MUGWUMPS\n");  // 打印游戏目标的说明
    print("HIDDEN ON A 10 BY 10 GRID.  HOMEBASE IS POSITION 0,0.\n");  // 打印游戏规则的说明
    print("ANY GUESS YOU MAKE MUST BE TWO NUMBERS WITH EACH\n");  // 打印游戏规则的说明
    print("NUMBER BETWEEN 0 AND 9, INCLUSIVE.  FIRST NUMBER\n");  // 打印游戏规则的说明
    print("IS DISTANCE TO RIGHT OF HOMEBASE AND SECOND NUMBER\n");  // 打印游戏规则的说明
    # 打印提示信息
    print("IS DISTANCE ABOVE HOMEBASE.\n");
    # 打印空行
    print("\n");
    # 打印提示信息
    print("YOU GET 10 TRIES.  AFTER EACH TRY, I WILL TELL\n");
    # 打印提示信息
    print("YOU HOW FAR YOU ARE FROM EACH MUGWUMP.\n");
    # 打印空行
    print("\n");
    # 进入无限循环
    while (1) {
        # 循环4次，初始化p数组
        for (i = 1; i <= 4; i++) {
            p[i] = [];
            # 循环2次，为p数组的每个元素赋随机值
            for (j = 1; j <= 2; j++) {
                p[i][j] = Math.floor(10 * Math.random());
            }
        }
        # 初始化t为0
        t = 0;
        # 进入do-while循环
        do {
            # t自增
            t++;
            # 打印提示信息
            print("\n");
            # 打印提示信息
            print("\n");
            # 打印提示信息
            print("TURN NO. " + t + " -- WHAT IS YOUR GUESS");
            # 获取用户输入的字符串
            str = await input();
            # 将用户输入的字符串转换为整数
            m = parseInt(str);
            n = parseInt(str.substr(str.indexOf(",") + 1)); // 从字符串中获取逗号后面的数字并转换为整数赋值给变量n
            for (i = 1; i <= 4; i++) { // 循环4次，i从1到4
                if (p[i][1] == -1) // 如果p[i][1]等于-1，跳过本次循环
                    continue;
                if (p[i][1] == m && p[i][2] == n) { // 如果p[i][1]等于m并且p[i][2]等于n
                    p[i][1] = -1; // 将p[i][1]赋值为-1
                    print("YOU HAVE FOUND MUGWUMP " + i + "\n"); // 打印找到MUGWUMP的消息
                } else {
                    d = Math.sqrt(Math.pow(p[i][1] - m, 2) + Math.pow(p[i][2] - n, 2)); // 计算距离并赋值给变量d
                    print("YOU ARE " + Math.floor(d * 10) / 10 + " UNITS FROM MUGWUMP " + i + "\n"); // 打印距离消息
                }
            }
            for (j = 1; j <= 4; j++) { // 循环4次，j从1到4
                if (p[j][1] != -1) // 如果p[j][1]不等于-1
                    break; // 退出循环
            }
            if (j > 4) { // 如果j大于4
                print("\n"); // 打印空行
                print("YOU GOT THEM ALL IN " + t + " TURNS!\n"); // 打印找到所有MUGWUMP的消息
                break; // 退出循环
        } while (t < 10) ;  // 使用 do-while 循环，最多循环10次
        if (t == 10) {  // 如果循环次数达到10次
            print("\n");  // 输出空行
            print("SORRY, THAT'S 10 TRIES.  HERE IS WHERE THEY'RE HIDING:\n");  // 输出提示信息
            for (i = 1; i <= 4; i++) {  // 使用 for 循环遍历数组
                if (p[i][1] != -1)  // 如果数组中的值不等于-1
                    print("MUGWUMP " + i + " IS AT (" + p[i][1] + "," + p[i][2] + ")\n");  // 输出提示信息
            }
        }
        print("\n");  // 输出空行
        print("THAT WAS FUN! LET'S PLAY AGAIN.......\n");  // 输出提示信息
        print("FOUR MORE MUGWUMPS ARE NOW IN HIDING.\n");  // 输出提示信息
    }
}

main();  // 调用主函数
```