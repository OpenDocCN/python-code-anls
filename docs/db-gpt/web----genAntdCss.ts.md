# `.\DB-GPT-src\web\genAntdCss.ts`

```py
import { createHash } from 'crypto';
import fs from 'fs';
import path from 'path';
import { extractStyle } from '@ant-design/cssinjs';
import type Entity from '@ant-design/cssinjs/lib/Cache';

// 定义函数参数类型，cache 是一个缓存实体，dir 是输出目录，默认为 'antd-output'，baseFileName 是输出文件名，默认为 'antd.min'
export type DoExtraStyleOptions = {
  cache: Entity;
  dir?: string;
  baseFileName?: string;
};

// 执行额外的样式处理，根据传入的选项生成对应的样式文件路径
export function doExtraStyle({ cache, dir = 'antd-output', baseFileName = 'antd.min' }: DoExtraStyleOptions) {
  // 确定基础目录为 '../../static/css' 相对于当前文件所在目录的路径
  const baseDir = path.resolve(__dirname, '../../static/css');

  // 根据 baseDir 和 dir 构建输出的 CSS 文件路径
  const outputCssPath = path.join(baseDir, dir);

  // 如果输出路径不存在，则创建它及其父目录
  if (!fs.existsSync(outputCssPath)) {
    fs.mkdirSync(outputCssPath, { recursive: true });
  }

  // 提取样式内容
  const css = extractStyle(cache, true);

  // 如果提取不到样式内容，则返回空字符串
  if (!css) return '';

  // 使用 MD5 算法创建 CSS 内容的哈希值
  const md5 = createHash('md5');
  const hash = md5.update(css).digest('hex');

  // 构建最终的文件名，包含基础文件名、哈希的前8位、以及 '.css' 扩展名
  const fileName = `${baseFileName}.${hash.substring(0, 8)}.css`;

  // 构建完整的输出文件路径
  const fullpath = path.join(outputCssPath, fileName);

  // 构建输出的相对路径，用于返回给调用者
  const res = `_next/static/css/${dir}/${fileName}`;

  // 如果输出文件已经存在，则直接返回相对路径
  if (fs.existsSync(fullpath)) return res;

  // 否则，将 CSS 内容写入文件
  fs.writeFileSync(fullpath, css);

  // 返回生成的相对路径
  return res;
}
```