# `.\DB-GPT-src\web\components\chat\doc-list.tsx`

```py
import { IDocument } from '@/types/knowledge';
import { Button, Tooltip } from 'antd';
import { useRouter } from 'next/router';
import React from 'react';
import FileStatusIcon from '../common/FileStatusIcon';

interface IProps {
  documents: IDocument[];  // 定义组件接收的 props，包含一个 IDocument 类型的数组 documents 和一个可选的字符串 dbParam
}

export default function DocList(props: IProps) {
  const { documents, dbParam } = props;  // 从 props 中解构出 documents 和 dbParam
  const router = useRouter();  // 获取路由对象

  // 点击处理函数，根据文档 id 和 dbParam 跳转到指定的页面路径
  const handleClick = (id: number) => {
    router.push(`/knowledge/chunk/?spaceName=${dbParam}&id=${id}`);
  };

  // 如果 documents 为空数组或未定义，返回 null
  if (!documents?.length) return null;

  // 渲染文档列表组件
  return (
    <div className="absolute flex overflow-scroll h-12 top-[-35px] w-full z-10">
      {documents.map((doc) => {
        let color;
        // 根据文档的状态选择相应的颜色
        switch (doc.status) {
          case 'RUNNING':
            color = '#2db7f5';
            break;
          case 'FINISHED':
            color = '#87d068';
            break;
          case 'FAILED':
            color = '#f50';
            break;
          default:
            color = '#87d068';  // 默认状态为 FINISHED 的颜色
            break;
        }
        // 返回一个带有 Tooltip 的 Button 组件，显示文档状态图标和文档名称
        return (
          <Tooltip key={doc.id} title={doc.result}>
            <Button
              style={{ color }}  // 根据状态设置按钮的颜色样式
              onClick={() => {
                handleClick(doc.id);  // 点击按钮时调用 handleClick 函数
              }}
              className="shrink flex items-center mr-3"  // 按钮的样式类名
            >
              <FileStatusIcon document={doc} />  // 显示文档状态图标
              {doc.doc_name}  // 显示文档名称
            </Button>
          </Tooltip>
        );
      })}
    </div>
  );
}
```