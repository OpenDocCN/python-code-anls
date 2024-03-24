# `.\lucidrains\liquid-conway\src\liquid.js`

```
# 导入 Rx 模块
import Rx from 'rxjs';
# 导入 helpers 模块
import helpers from './helpers';

# 导入样式表
require('./app.sass');

# 从 helpers 模块中导入 initArray 函数
const { initArray } = helpers;

# 获取 canvas 元素
const c = document.getElementById('canvas');

# 阻止右键菜单弹出
c.oncontextmenu = (e) => {
  e.preventDefault();
};

# 设置常量
const FPS = 30;
const WIDTH = 80;
const HEIGHT = 60;
const CELL_SIZE = 10;
const CANVAS_WIDTH = WIDTH * CELL_SIZE;
const CANVAS_HEIGHT = HEIGHT * CELL_SIZE;

const CELL_COLOR_LIGHTEST = 'rgb(0, 204, 255)';
const CELL_COLOR_LIGHT = 'rgb(0, 153, 255)';
const CELL_COLOR = 'rgb(0, 102, 255)';
const CELL_COLOR_DARK = 'rgb(51, 102, 255)';
const CELL_COLOR_DARKEST = 'rgb(51, 51, 204)';

const BACKGROUND_COLOR = 'rgb(255, 255, 255)';

# 初始化网格
function initGrid(x, y, init) {
  return initArray(x, init).map(() => initArray(y, init));
}

# 创建网格
const GRID = initGrid(WIDTH, HEIGHT, { val: 0, diff: 0 });

# 获取网格坐标
const GRID_COORS = GRID.reduce((acc, row, x) =>
  acc.concat(row.map((_, y) => [x, y]))
, []);

# 检查坐标是否在网格内
function withinBounds(grid, x, y) {
  return x >= 0 && x < grid.length && y >= 0 && y < grid[0].length;
}

# 检查网格中的单元格是否为空
function isEmptyCell(grid, x, y) {
  return withinBounds(grid, x, y) && !grid[x][y].wall;
}

# 设置 canvas 元素的宽度和高度
c.setAttribute('width', CANVAS_WIDTH.toString());
c.setAttribute('height', CANVAS_HEIGHT.toString());
c.style.display = 'block';

# 获取 2D 上下文
const ctx = c.getContext('2d');

# 合并鼠标和触摸事件的 Observable 流
Rx.Observable.merge(
    Rx.Observable.fromEvent(c, 'mousedown'),
    Rx.Observable.fromEvent(c, 'touchstart')
  )
  .flatMap((md) => {
    md.preventDefault();
    let ev = md;

    return Rx.Observable.merge(
        Rx.Observable.interval(10).map(() => null),
        Rx.Observable.fromEvent(c, 'mousemove'),
        Rx.Observable.fromEvent(c, 'touchmove')
      )
      .map((mm) => {
        ev = mm || ev;
        return { ev, which: md.which };
      })
      .takeUntil(Rx.Observable.merge(
        Rx.Observable.fromEvent(c, 'mouseup'),
        Rx.Observable.fromEvent(c, 'mouseout'),
        Rx.Observable.fromEvent(c, 'touchend')
      ));
  })
  .throttleTime(10)
  .subscribe(({ ev, which }) => {
    const { target, touches, type } = ev;
    const isTouch = type === 'touchmove' || type === 'touchstart';

    const { left, top } = target.getBoundingClientRect();
    const { clientX, clientY } = isTouch ? touches[0] : ev;

    const x = clientX - left;
    const y = clientY - top;
    const [cx, cy] = [x, y].map(el => Math.floor(el / CELL_SIZE));

    if (!withinBounds(GRID, cx, cy)) {
      return;
    }

    const cell = GRID[cx][cy];

    if (which === 1 || isTouch) {
      delete cell.wall;
      cell.val += 100;
    } else if (which === 3) {
      cell.wall = true;
      cell.val = 0;
    }
  });

# 计算下一个状态
function nextState(grid) {
  const withinGrid = withinBounds.bind(null, grid);

  GRID_COORS.forEach(([x, y]) => {
    const cell = grid[x][y];
    const val = cell.val;

    if (cell.wall || val < 0) {
      return;
    }

    if (withinGrid(x, y + 1) && grid[x][y + 1].val < 100) {
      cell.diff -= val;
      grid[x][y + 1].diff += val;
      return;
    }

    let volume = val;

    const flowCoors = [[1, 0], [-1, 0]]
      .filter(([dx, dy]) => {
        const [nx, ny] = [x + dx, y + dy];
        return withinGrid(nx, ny) && val > grid[nx][ny].val;
      });

    const diffs = flowCoors.map(([dx, dy]) => {
      const [nx, ny] = [x + dx, y + dy];
      const diff = val - grid[nx][ny].val;
      return diff;
    });

    const totalDiff = diffs.reduce((acc, diff) => {
      acc += diff;
      return acc;
    }, 0);

    const finalDiff = Math.min(volume, totalDiff);

    diffs.forEach((diff, i) => {
      const [dx, dy] = flowCoors[i];
      const weightedDiff = Math.floor(finalDiff * (diff / totalDiff)) / 2;

      grid[x][y].diff -= weightedDiff;
      grid[x + dx][y + dy].diff += weightedDiff;
      volume -= weightedDiff;
    });

    if (volume < 0) {
      return;
    }
    # 如果当前单元格上方的单元格在网格内且数值小于当前单元格的数值，并且当前单元格的数值大于100
    if (withinGrid(x, y - 1) && grid[x][y - 1].val < cell.val && cell.val > 100) {
      # 计算数值差值，将差值的一部分分配给上方单元格
      const diff = Math.floor((val - grid[x][y - 1].val) / 20);
      grid[x][y - 1].diff += diff;
      # 减去分配的差值
      cell.diff -= diff;
      # 更新总体差值
      volume -= diff;
    }

    # 如果当前单元格下方的单元格在网格内且数值小于当前单元格的数值
    if (withinGrid(x, y + 1) && grid[x][y + 1].val < cell.val) {
      # 计算数值差值，将差值的一部分分配给下方单元格
      const diff = Math.floor((val - grid[x][y + 1].val) / 10);
      grid[x][y + 1].diff += diff;
      # 减去分配的差值
      cell.diff -= diff;
      # 更新总体差值
      volume -= diff;
    }
  });

  # 遍历所有网格坐标
  GRID_COORS.forEach(([x, y]) => {
    # 获取当前单元格
    const cell = grid[x][y];
    # 更新单元格数值，重置差值为0
    cell.val += cell.diff;
    cell.diff = 0;
  });
// 渲染函数，根据传入的上下文和网格对象进行绘制
function render(context, grid) {
  // 设置背景颜色并填充整个画布
  context.fillStyle = BACKGROUND_COLOR;
  context.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  // 遍历所有网格坐标
  GRID_COORS.forEach(([x, y]) => {
    // 获取当前坐标对应的单元格对象
    const cell = grid[x][y];

    // 如果单元格是墙壁
    if (cell.wall) {
      // 设置颜色为黑色并填充墙壁单元格
      context.fillStyle = 'black';
      context.fillRect(
        (x * CELL_SIZE) + 1,
        (y * CELL_SIZE) + 1,
        CELL_SIZE,
        CELL_SIZE
      );
    } else {
      // 如果单元格不是墙壁
      const val = cell.val;

      // 如果值小于等于0，则跳过
      if (val <= 0) {
        return;
      }

      let fillStyle = CELL_COLOR;
      let cellHeight = CELL_SIZE - 1;
      let cellY = (y * CELL_SIZE) + 1;

      // 检查是否有底部相邻单元格或者顶部无相邻单元格
      const hasBottomNeighbor = (!isEmptyCell(grid, x, y + 1) || grid[x][y + 1].val > 0);
      const hasNoTopNeighbor = (!isEmptyCell(grid, x, y - 1) || grid[x][y - 1].val <= 0);

      // 根据条件调整单元格高度和位置
      if (val < 100 && hasBottomNeighbor && hasNoTopNeighbor) {
        cellHeight *= parseFloat(val) / 100;
        cellY += (CELL_SIZE - cellHeight);
      }

      // 根据值的大小设置不同的颜色
      if (val < 50) {
        fillStyle = CELL_COLOR_LIGHTEST;
      } else if (val < 80) {
        fillStyle = CELL_COLOR_LIGHT;
      } else if (val > 150) {
        fillStyle = CELL_COLOR_DARKEST;
      } else if (val > 120) {
        fillStyle = CELL_COLOR_DARK;
      }

      // 设置颜色并填充单元格
      context.fillStyle = fillStyle;
      context.fillRect(
        (x * CELL_SIZE) + 1,
        cellY,
        CELL_SIZE - 1,
        cellHeight
      );
    }
  });
}

// 初始化时间变量和节流时间间隔
let start;
const throttleDiff = (1000 / FPS);

// 每一帧的处理函数
function step() {
  const now = +new Date();
  start = start || now;
  const diff = now - start;
  start = now;

  // 调用渲染函数
  render(ctx, GRID);

  // 请求下一帧动画
  const callNextFrame = window.requestAnimationFrame.bind(null, step);
  if (diff > throttleDiff) {
    callNextFrame();
  } else {
    setTimeout(callNextFrame, throttleDiff - diff);
  }
}

// 开始执行动画
step();

// 每50毫秒更新一次网格状态
setInterval(() => {
  nextState(GRID);
}, 50);
```