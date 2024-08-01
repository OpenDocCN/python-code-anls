# `.\DB-GPT-src\web\components\model\model-card.tsx`

```py
import React, { useState } from 'react';
import { IModelData } from '@/types/model';
import { useTranslation } from 'react-i18next';
import { message } from 'antd';
import moment from 'moment';
import { apiInterceptors, stopModel } from '@/client/api';
import GptCard from '../common/gpt-card';
import { PauseCircleOutlined } from '@ant-design/icons';
import { MODEL_ICON_MAP } from '@/utils';

// 定义 Props 接口，包含 info 属性，类型为 IModelData
interface Props {
  info: IModelData;
}

// 定义 ModelCard 组件，接收 Props 参数
function ModelCard({ info }: Props) {
  // 使用 useTranslation hook 获取 t 函数
  const { t } = useTranslation();
  // 使用 useState hook 创建 loading 状态及其更新函数 setLoading
  const [loading, setLoading] = useState<boolean>(false);

  // 定义异步函数 stopTheModel，用于停止模型
  async function stopTheModel(info: IModelData) {
    // 如果 loading 为 true，则直接返回，不执行后续代码
    if (loading) {
      return;
    }
    // 设置 loading 为 true
    setLoading(true);
    // 发送停止模型的请求，并获取返回结果
    const [, res] = await apiInterceptors(
      stopModel({
        host: info.host,
        port: info.port,
        model: info.model_name,
        worker_type: info.model_type,
        params: {},
      }),
    );
    // 设置 loading 为 false
    setLoading(false);
    // 如果返回结果为 true，则显示停止模型成功的消息
    if (res === true) {
      message.success(t('stop_model_success'));
    }
  }

  // 返回 GptCard 组件，展示模型卡片信息
  return (
    <GptCard
      className="w-96"
      title={info.model_name}
      tags={[
        {
          text: info.healthy ? 'Healthy' : 'Unhealthy',
          color: info.healthy ? 'green' : 'red',
          border: true,
        },
        info.model_type,
      ]}
      icon={MODEL_ICON_MAP[info.model_name]?.icon || '/models/huggingface.svg'}
      operations={[
        {
          children: (
            <div>
              <PauseCircleOutlined className="mr-2" />
              <span className="text-sm">Stop Model</span>
            </div>
          ),
          onClick: () => {
            stopTheModel(info);
          },
        },
      ]}
    >
      <div className="flex flex-col gap-1 px-4 pb-4 text-xs">
        <div className="flex overflow-hidden">
          <p className="w-28 text-gray-500 mr-2">Host:</p>
          <p className="flex-1 text-ellipsis">{info.host}</p>
        </div>
        <div className="flex overflow-hidden">
          <p className="w-28 text-gray-500 mr-2">Manage Host:</p>
          <p className="flex-1 text-ellipsis">
            {info.manager_host}:{info.manager_port}
          </p>
        </div>
        <div className="flex overflow-hidden">
          <p className="w-28 text-gray-500 mr-2">Last Heart Beat:</p>
          <p className="flex-1 text-ellipsis">{moment(info.last_heartbeat).format('YYYY-MM-DD')}</p>
        </div>
      </div>
    </GptCard>
  );
}

// 导出 ModelCard 组件
export default ModelCard;
```