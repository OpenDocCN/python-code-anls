# `.\DB-GPT-src\web\components\common\FileStatusIcon.tsx`

```py
import { IDocument } from '@/types/knowledge';
import React from 'react';
import { FileDone, FileSync } from '../icons';
import FileError from '../icons/file-error';

interface IProps {
  document: IDocument;  // 定义组件属性接口，包含一个名为document的IDocument类型对象
}

export default function FileStatusIcon({ document }: IProps) {
  switch (document.status) {  // 根据document对象的status属性进行条件判断
    case 'RUNNING':  // 当status为'RUNNING'时
      return <FileSync />;  // 返回FileSync组件，表示运行中的状态图标
    case 'FINISHED':  // 当status为'FINISHED'时
      return <FileDone />;  // 返回FileDone组件，表示完成的状态图标
    case 'FAILED':  // 当status为'FAILED'时
      return <FileError />;  // 返回FileError组件，表示失败的状态图标
    default:  // 默认情况，如果status不匹配上述任何一种情况
      return <FileDone />;  // 返回FileDone组件，作为默认状态图标
  }
}
```