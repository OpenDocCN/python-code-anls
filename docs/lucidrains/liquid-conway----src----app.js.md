# `.\lucidrains\liquid-conway\src\app.js`

```py
# 导入 RxJS 库
import Rx from 'rxjs';
# 导入 helpers 模块
import helpers from './helpers';
# 导入样式文件
require('./app.sass');

# 从 helpers 模块中解构出 range、initArray、cacheFn 函数
const {
  range,
  initArray,
  cacheFn
} = helpers;

# 获取 canvas 元素
const c = document.getElementById('canvas');

# 定义常量
const FPS = 30;
const WIDTH = 150;
const HEIGHT = 75;
const CELL_SIZE = 7;
const CANVAS_WIDTH = WIDTH * CELL_SIZE;
const CANVAS_HEIGHT = HEIGHT * CELL_SIZE;
const CELL_FILL_STYLE = 'rgb(22, 109, 175)';
const BACKGROUND_COLOR = 'rgba(255, 255, 255, 0.5)';
const NEIGHBOR_COORS_CACHE = {};

# 定义方向数组 DIR
const DIR = range(-1, 1)
  .reduce((acc, x) => acc.concat(range(-1, 1).map(y => [x, y])), [])
  .filter(([x, y]) => !(x === 0 && y === 0));

# 设置 canvas 元素的宽度和高度
c.setAttribute('width', CANVAS_WIDTH.toString());
c.setAttribute('height', CANVAS_HEIGHT.toString());
c.style.display = 'block';
# 获取 2D 绘图上下文
const ctx = c.getContext('2d');

# 初始化网格
function initGrid(x, y, init) {
  return initArray(x, init).map(() => initArray(y, init));
}

# 初始化 grid 和 buffer
let [
  grid,
  buffer
] = [
  initGrid(WIDTH, HEIGHT, 0),
  initGrid(WIDTH, HEIGHT, 0)
];

# 获取网格坐标
const GRID_COORS = grid.reduce((acc, row, x) => {
  acc = acc.concat(row.map((_, y) => [x, y]));
  return acc;
}, []);

# 在网格中随机生成初始状态
GRID_COORS.forEach(([x, y]) => {
  grid[x][y] = Math.round(Math.random());
});

# 创建 RxJS Observable，处理鼠标事件
Rx.Observable
  .fromEvent(c, 'mousedown')
  .flatMap((md) => {
    md.preventDefault();
    let ev = md;

    return Rx.Observable.merge(
        Rx.Observable.interval(10).map(() => null),
        Rx.Observable.fromEvent(c, 'mousemove')
      )
      .map((mm) => {
        ev = mm || ev;
        const { left, top } = ev.target.getBoundingClientRect();
        const x = ev.clientX - left;
        const y = ev.clientY - top;
        const [coorX, coorY] = [x, y].map(el => Math.floor(el / CELL_SIZE));
        return [coorX, coorY];
      })
      .takeUntil(Rx.Observable.fromEvent(c, 'mouseup'));
  })
  .throttleTime(10)
  .subscribe(([x, y]) => {
    grid[x][y] = 1;
  });

# 判断坐标是否在网格范围内
function withinBounds(grid, x, y) {
  return x >= 0 && x < grid.length && y >= 0 && y < grid[0].length;
}

# 获取邻居坐标
function getNeighborCoors(grid, x, y) {
  return DIR.reduce((acc, [dx, dy]) => {
    const [nx, ny] = [dx + x, dy + y];
    if (withinBounds(grid, nx, ny)) {
      acc.push([nx, ny]);
    }
    return acc;
  }, []);
}

# 使用缓存函数获取邻居坐标
const getCacheNeighborCoors = cacheFn(
  getNeighborCoors,
  NEIGHBOR_COORS_CACHE,
  (_, x, y) => `${x}:${y}`
);

# 计算邻居中存活细胞数量
function countNeighborsAlive(grid, x, y) {
  const neighbors = getCacheNeighborCoors(grid, x, y);

  return neighbors.reduce((acc, [nx, ny]) => {
    if (grid[nx][ny] === 1) {
      acc += 1;
    }
    return acc;
  }, 0);
}

# 计算下一个状态
function computeNextState(curr, neighbors) {
  return ((curr === 1 && neighbors === 2) || neighbors === 3) ? 1 : 0;
}

# 计算下一个状态
function nextState(grid, buffer) {
  GRID_COORS.forEach(([x, y]) => {
    const cell = grid[x][y];
    const count = countNeighborsAlive(grid, x, y);
    buffer[x][y] = computeNextState(cell, count);
  });
}

# 渲染函数
function render(ctx, grid) {
  ctx.fillStyle = BACKGROUND_COLOR;
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  GRID_COORS.forEach(([x, y]) => {
    const cell = grid[x][y];
    if (cell === 1) {
      ctx.fillStyle = CELL_FILL_STYLE;
      ctx.fillRect(
        (x * CELL_SIZE) + 1,
        (y * CELL_SIZE) + 1,
        CELL_SIZE - 1,
        CELL_SIZE - 1
      );
    }
  });
}

# 定义动画函数
let start;
const throttleDiff = (1000 / FPS);

function step() {
  const now = +new Date();
  start = start || now;
  const diff = now - start;
  start = now;

  render(ctx, grid);

  const callNextFrame = window.requestAnimationFrame.bind(null, step);
  if (diff > throttleDiff) {
    callNextFrame();
  } else {
    setTimeout(callNextFrame, throttleDiff - diff);
  }
}

# 启动动画
step();

# 定时更新状态
setInterval(() => {
  nextState(grid, buffer);
  [buffer, grid] = [grid, buffer];
}, 80);
```