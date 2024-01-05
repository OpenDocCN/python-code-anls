# `58_Love\javascript\love.js`

```
// 定义名为print的函数，用于向页面输出文本
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义名为input的函数，用于获取用户输入
function input()
{
    var input_element;
    var input_str;

    // 返回一个Promise对象，用于异步处理用户输入
    return new Promise(function (resolve) {
                       // 创建一个input元素
                       input_element = document.createElement("INPUT");

                       // 在页面上输出提示符
                       print("? ");

                       // 设置input元素的类型为文本输入
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

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串
}

var data = [60,1,12,26,9,12,3,8,24,17,8,4,6,23,21,6,4,6,22,12,5,6,5,
            4,6,21,11,8,6,4,4,6,21,10,10,5,4,4,6,21,9,11,5,4,
            4,6,21,8,11,6,4,4,6,21,7,11,7,4,4,6,21,6,11,8,4,
            4,6,19,1,1,5,11,9,4,4,6,19,1,1,5,10,10,4,4,6,18,2,1,6,8,11,4,
            4,6,17,3,1,7,5,13,4,4,6,15,5,2,23,5,1,29,5,17,8,
            1,29,9,9,12,1,13,5,40,1,1,13,5,40,1,4,6,13,3,10,6,12,5,1,
            5,6,11,3,11,6,14,3,1,5,6,11,3,11,6,15,2,1,
            6,6,9,3,12,6,16,1,1,6,6,9,3,12,6,7,1,10,
            7,6,7,3,13,6,6,2,10,7,6,7,3,13,14,10,8,6,5,3,14,6,6,2,10,
            8,6,5,3,14,6,7,1,10,9,6,3,3,15,6,16,1,1,
            9,6,3,3,15,6,15,2,1,10,6,1,3,16,6,14,3,1,10,10,16,6,12,5,1,
            11,8,13,27,1,11,8,13,27,1,60];  // 定义一个包含大量数字的数组

// Main program
async function main()
{
    # 打印一个包含33个空格和"LOVE"的字符串
    print(tab(33) + "LOVE\n");
    # 打印一个包含15个空格和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"的字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    # 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    # 打印关于艺术家Robert Indiana的致敬信息
    print("A TRIBUTE TO THE GREAT AMERICAN ARTIST, ROBERT INDIANA.\n");
    print("HIS GREATEST WORK WILL BE REPRODUCED WITH A MESSAGE OF\n");
    print("YOUR CHOICE UP TO 60 CHARACTERS.  IF YOU CAN'T THINK OF\n");
    print("A MESSAGE, SIMPLE TYPE THE WORD 'LOVE'\n");
    print("\n");
    # 提示用户输入消息
    print("YOUR MESSAGE, PLEASE");
    # 等待用户输入消息
    str = await input();
    # 获取输入消息的长度
    l = str.length;
    # 初始化一个空数组ts
    ts = [];
    # 打印十个空行
    for (i = 1; i <= 10; i++)
        print("\n");
    # 将ts设置为空字符串
    ts = "";
    # 将输入消息重复拼接到ts，直到ts的长度达到60
    do {
        ts += str;
    } while (ts.length < 60) ;
    pos = 0;  // 初始化变量 pos 为 0
    c = 0;  // 初始化变量 c 为 0
    while (++c < 37) {  // 循环条件，当 c 小于 37 时执行循环
        a1 = 1;  // 初始化变量 a1 为 1
        p = 1;  // 初始化变量 p 为 1
        print("\n");  // 打印换行符
        do {  // do-while 循环开始
            a = data[pos++];  // 从 data 数组中取出 pos 位置的值给变量 a，然后 pos 自增
            a1 += a;  // a1 加上 a 的值
            if (p != 1) {  // 判断 p 是否不等于 1
                for (i = 1; i <= a; i++)  // 循环 a 次
                    print(" ");  // 打印空格
                p = 1;  // 将 p 设为 1
            } else {  // 如果 p 等于 1
                for (i = a1 - a; i <= a1 - 1; i++)  // 循环从 a1-a 到 a1-1
                    print(ts[i]);  // 打印 ts 数组中下标为 i 的值
                p = 0;  // 将 p 设为 0
            }
        } while (a1 <= 60);  // 循环条件，当 a1 小于等于 60 时执行循环
    }
    for (i = 1; i <= 10; i++)  # 使用循环从1到10
        print("\n");  # 打印换行符10次
}

main();  # 调用主函数
```