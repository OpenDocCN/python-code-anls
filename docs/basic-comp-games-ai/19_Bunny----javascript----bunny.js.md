# `basic-computer-games\19_Bunny\javascript\bunny.js`

```
// 定义一个打印函数，将字符串添加到指定 id 的元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个制表符函数，返回指定数量空格的字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个包含字符'B','U','N','N','Y'的数组
var bunny_string = ["B","U","N","N","Y"];

// 定义一个包含ASCII码的数组
var bunny_data = [1,2,-1,0,2,45,50,-1,0,5,43,52,-1,0,7,41,52,-1,
    1,9,37,50,-1,2,11,36,50,-1,3,13,34,49,-1,4,14,32,48,-1,
    5,15,31,47,-1,6,16,30,45,-1,7,17,29,44,-1,8,19,28,43,-1,
    9,20,27,41,-1,10,21,26,40,-1,11,22,25,38,-1,12,22,24,36,-1,
    13,34,-1,14,33,-1,15,31,-1,17,29,-1,18,27,-1,
    19,26,-1,16,28,-1,13,30,-1,11,31,-1,10,32,-1,
    8,33,-1,7,34,-1,6,13,16,34,-1,5,12,16,35,-1,
    4,12,16,35,-1,3,12,15,35,-1,2,35,-1,1,35,-1,
    2,34,-1,3,34,-1,4,33,-1,6,33,-1,10,32,34,34,-1,
    14,17,19,25,28,31,35,35,-1,15,19,23,30,36,36,-1,
    14,18,21,21,24,30,37,37,-1,13,18,23,29,33,38,-1,
    12,29,31,33,-1,11,13,17,17,19,19,22,22,24,31,-1,
    10,11,17,18,22,22,24,24,29,29,-1,
    22,23,26,29,-1,27,29,-1,28,29,-1,4096];

// 在指定位置打印字符串'BUNNY'
print(tab(32) + "BUNNY\n");
// 在指定位置打印字符串'CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY'
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
// 打印多个空行
print("\n");
print("\n");
print("\n");

// 初始化变量l为64，表示ASCII码
var l = 64;    // ASCII letter code
var pos = 0;

// 打印一个空行
print("\n");

// 初始化一个空字符串
var str = "";
// 遍历bunny_data数组，直到遇到大于等于128的值
for (var pos = 0; bunny_data[pos] < 128; pos++) {
    // 如果数组元素小于0，打印当前字符串并重置为空字符串
    if (bunny_data[pos] < 0) {
        print(str + "\n");
        str = "";
        continue;
    }
    // 当前字符串长度小于数组元素值时，添加空格
    while (str.length < bunny_data[pos])
        str += " ";
    // 根据数组元素值和下一个元素值，添加对应的字符到字符串中
    for (var i = bunny_data[pos]; i <= bunny_data[pos + 1]; i++)
        str += bunny_string[i % 5];
    pos++;
}
```