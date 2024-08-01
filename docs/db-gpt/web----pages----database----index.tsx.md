# `.\DB-GPT-src\web\pages\database\index.tsx`

```py
import React, { useMemo, useState } from 'react';
import { useAsyncEffect } from 'ahooks';
import { Badge, Button, Card, Drawer, Empty, Modal, Spin, message } from 'antd';
import FormDialog from '@/components/database/form-dialog';
import { apiInterceptors, getDbList, getDbSupportType, postDbDelete, postDbRefresh } from '@/client/api';
import { DeleteFilled, EditFilled, PlusOutlined, RedoOutlined } from '@ant-design/icons';
import { DBOption, DBType, DbListResponse, DbSupportTypeResponse } from '@/types/db';
import MuiLoading from '@/components/common/loading';
import { dbMapper } from '@/utils';
import GPTCard from '@/components/common/gpt-card';
import { useTranslation } from 'react-i18next';

type DBItem = DbListResponse[0];

// 检查数据库类型是否为文件数据库
export function isFileDb(dbTypeList: DBOption[], dbType: DBType) {
  return dbTypeList.find((item) => item.value === dbType)?.isFileDb;
}

function Database() {
  const { t } = useTranslation();

  // 状态管理
  const [dbList, setDbList] = useState<DbListResponse>([]); // 数据库列表状态
  const [dbSupportList, setDbSupportList] = useState<DbSupportTypeResponse>([]); // 数据库支持类型列表状态
  const [loading, setLoading] = useState(false); // 加载状态
  const [modal, setModal] = useState<{ open: boolean; info?: DBItem; dbType?: DBType }>({ open: false }); // 模态框状态
  const [draw, setDraw] = useState<{ open: boolean; dbList?: DbListResponse; name?: string; type?: DBType }>({ open: false }); // 抽屉状态
  const [refreshLoading, setRefreshLoading] = useState(false); // 刷新加载状态

  // 获取数据库支持类型列表
  const getDbSupportList = async () => {
    const [, data] = await apiInterceptors(getDbSupportType());
    setDbSupportList(data ?? []);
  };

  // 刷新数据库列表
  const refreshDbList = async () => {
    setLoading(true);
    const [, data] = await apiInterceptors(getDbList());
    setDbList(data ?? []);
    setLoading(false);
  };

  // 计算数据库类型选项列表
  const dbTypeList = useMemo(() => {
    const supportDbList = dbSupportList.map((item) => {
      const { db_type, is_file_db } = item;
      return { ...dbMapper[db_type], value: db_type, isFileDb: is_file_db };
    }) as DBOption[];
    const unSupportDbList = Object.keys(dbMapper)
      .filter((item) => !supportDbList.some((db) => db.value === item))
      .map((item) => ({ ...dbMapper[item], value: dbMapper[item].label, disabled: true })) as DBOption[];
    return [...supportDbList, ...unSupportDbList];
  }, [dbSupportList]);

  // 修改数据库项
  const onModify = (item: DBItem) => {
    setModal({ open: true, info: item });
  };

  // 删除数据库项
  const onDelete = (item: DBItem) => {
    Modal.confirm({
      title: 'Tips',
      content: `Do you Want to delete the ${item.db_name}?`,
      onOk() {
        return new Promise<void>(async (resolve, reject) => {
          try {
            const [err] = await apiInterceptors(postDbDelete(item.db_name));
            if (err) {
              message.error(err.message);
              reject();
              return;
            }
            message.success('success');
            refreshDbList();
            resolve();
          } catch (e: any) {
            reject();
          }
        });
      },
      // 确认删除提示框
      onCancel() {},
    });
  },
  });

  // 定义刷新函数，用于刷新数据库列表
  const onRefresh = async (item: DBItem) => {
    // 设置刷新状态为 true
    setRefreshLoading(true);
    // 发送刷新数据库请求，并获取返回结果
    const [, res] = await apiInterceptors(postDbRefresh({ db_name: item.db_name, db_type: item.db_type }));
    // 如果返回结果存在，则显示刷新成功的消息
    if (res) message.success(t('refreshSuccess'));
    // 设置刷新状态为 false
    setRefreshLoading(false);
  };

  // 根据数据库类型列表和数据库连接列表生成数据库列表映射
  const dbListByType = useMemo(() => {
    const mapper = dbTypeList.reduce((acc, item) => {
      acc[item.value] = dbList.filter((dbConn) => dbConn.db_type === item.value);
      return acc;
    }, {} as Record<DBType, DbListResponse>);
    return mapper;
  }, [dbList, dbTypeList]);

  // 使用异步效果钩子，在组件挂载时刷新数据库列表和获取数据库支持列表
  useAsyncEffect(async () => {
    await refreshDbList();
    await getDbSupportList();
  }, []);

  // 处理数据库类型点击事件，根据点击的数据库类型筛选数据库连接列表并设置抽屉状态
  const handleDbTypeClick = (info: DBOption) => {
    const dbItems = dbList.filter((item) => item.db_type === info.value);
    setDraw({ open: true, dbList: dbItems, name: info.label, type: info.value });
  };

  // 返回组件的结束标签
  return (
    </div>
  );
}
// 结束 Database 类的定义

export default Database;
// 导出 Database 类作为模块的默认输出
```