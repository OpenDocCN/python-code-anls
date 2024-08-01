# `.\DB-GPT-src\web\components\chat\header\db-selector.tsx`

```py
import { ChatContext } from '@/app/chat-context';
import { apiInterceptors, postChatModeParamsList } from '@/client/api';
import { IDB } from '@/types/chat';
import { dbMapper } from '@/utils';
import { useAsyncEffect } from 'ahooks';
import { Select } from 'antd';
import { useContext, useEffect, useMemo, useState } from 'react';
import DBIcon from '@/components/common/db-icon';

function DBSelector() {
  // 从 ChatContext 中获取场景、数据库参数和设置数据库参数的函数
  const { scene, dbParam, setDbParam } = useContext(ChatContext);

  // 存储从 API 获取的数据库列表
  const [dbs, setDbs] = useState<IDB[]>([]);

  // 使用 useAsyncEffect 异步获取数据库列表
  useAsyncEffect(async () => {
    // 调用 API 获取聊天模式参数列表的拦截器
    const [, res] = await apiInterceptors(postChatModeParamsList(scene as string));
    // 更新数据库列表状态，如果 res 为 null 则设置为空数组
    setDbs(res ?? []);
  }, [scene]);

  // 使用 useMemo 根据数据库列表映射为选项数组
  const dbOpts = useMemo(
    () =>
      dbs.map?.((db: IDB) => {
        // 将每个数据库项映射为 Select 组件的选项对象，包括名称和图标等属性
        return { name: db.param, ...dbMapper[db.type] };
      }),
    [dbs],
  );

  // 当 dbOpts 变化或者 dbParam 为空时，设置默认数据库参数
  useEffect(() => {
    if (dbOpts?.length && !dbParam) {
      setDbParam(dbOpts[0].name);
    }
  }, [dbOpts, setDbParam, dbParam]);

  // 如果数据库选项为空，则返回 null
  if (!dbOpts?.length) return null;

  // 渲染带有图标和名称的数据库选择下拉框
  return (
    <Select
      value={dbParam} // 设置当前选中的数据库参数值
      className="w-36" // 设置 Select 组件的样式
      onChange={(val) => {
        setDbParam(val); // 当选择项变化时更新数据库参数
      }}
    >
      {/* 渲染每个数据库选项 */}
      {dbOpts.map((item) => (
        <Select.Option key={item.name}>
          {/* 渲染数据库图标和名称 */}
          <DBIcon width={24} height={24} src={item.icon} label={item.label} className="w-[1.5em] h-[1.5em] mr-1 inline-block mt-[-4px]" />
          {item.name}
        </Select.Option>
      ))}
    </Select>
  );
}

export default DBSelector;
```