# `.\DB-GPT-src\web\utils\constants.ts`

```py
// 导入模型类型 ModelType 和数据库类型 DBType，这些类型位于 '@/types/chat' 和 '@/types/db' 中
import { ModelType } from '@/types/chat';
import { DBType } from '@/types/db';

// 导出一个名为 dbMapper 的常量，其类型为 Record<DBType, { label: string; icon: string; desc: string }>，
// 用于映射每种 DBType 到对应的数据库信息对象
export const dbMapper: Record<DBType, { label: string; icon: string; desc: string }> = {
  // 映射 mysql 数据库类型，包括其名称、图标路径和描述信息
  mysql: {
    label: 'MySQL',
    icon: '/icons/mysql.png',
    desc: 'Fast, reliable, scalable open-source relational database management system.',
  },
  // 映射 oceanbase 数据库类型，包括其名称、图标路径和描述信息
  oceanbase: {
    label: 'OceanBase',
    icon: '/icons/oceanbase.png',
    desc: 'An Ultra-Fast & Cost-Effective Distributed SQL Database.',
  },
  // 映射 mssql 数据库类型，包括其名称、图标路径和描述信息
  mssql: {
    label: 'MSSQL',
    icon: '/icons/mssql.png',
    desc: 'Powerful, scalable, secure relational database system by Microsoft.',
  },
  // 映射 duckdb 数据库类型，包括其名称、图标路径和描述信息
  duckdb: {
    label: 'DuckDB',
    icon: '/icons/duckdb.png',
    desc: 'In-memory analytical database with efficient query processing.',
  },
  // 映射 sqlite 数据库类型，包括其名称、图标路径和描述信息
  sqlite: {
    label: 'Sqlite',
    icon: '/icons/sqlite.png',
    desc: 'Lightweight embedded relational database with simplicity and portability.',
  },
  // 映射 clickhouse 数据库类型，包括其名称、图标路径和描述信息
  clickhouse: {
    label: 'ClickHouse',
    icon: '/icons/clickhouse.png',
    desc: 'Columnar database for high-performance analytics and real-time queries.',
  },
  // 映射 oracle 数据库类型，包括其名称、图标路径和描述信息
  oracle: {
    label: 'Oracle',
    icon: '/icons/oracle.png',
    desc: 'Robust, scalable, secure relational database widely used in enterprises.',
  },
  // 映射 access 数据库类型，包括其名称、图标路径和描述信息
  access: {
    label: 'Access',
    icon: '/icons/access.png',
    desc: 'Easy-to-use relational database for small-scale applications by Microsoft.',
  },
  // 映射 mongodb 数据库类型，包括其名称、图标路径和描述信息
  mongodb: {
    label: 'MongoDB',
    icon: '/icons/mongodb.png',
    desc: 'Flexible, scalable NoSQL document database for web and mobile apps.',
  },
  // 映射 doris 数据库类型，包括其名称、图标路径和描述信息
  doris: {
    label: 'ApacheDoris',
    icon: '/icons/doris.png',
    desc: 'A new-generation open-source real-time data warehouse.',
  },
  // 映射 starrocks 数据库类型，包括其名称、图标路径和描述信息
  starrocks: {
    label: 'StarRocks',
    icon: '/icons/starrocks.png',
    desc: 'An Open-Source, High-Performance Analytical Database.',
  },
  // 映射 db2 数据库类型，包括其名称、图标路径和描述信息
  db2: {
    label: 'DB2',
    icon: '/icons/db2.png',
    desc: 'Scalable, secure relational database system developed by IBM.',
  },
  // 映射 hbase 数据库类型，包括其名称、图标路径和描述信息
  hbase: {
    label: 'HBase',
    icon: '/icons/hbase.png',
    desc: 'Distributed, scalable NoSQL database for large structured/semi-structured data.',
  },
  // 映射 redis 数据库类型，包括其名称、图标路径和描述信息
  redis: {
    label: 'Redis',
    icon: '/icons/redis.png',
    desc: 'Fast, versatile in-memory data structure store as cache, DB, or broker.',
  },
  // 映射 cassandra 数据库类型，包括其名称、图标路径和描述信息
  cassandra: {
    label: 'Cassandra',
    icon: '/icons/cassandra.png',
    desc: 'Scalable, fault-tolerant distributed NoSQL database for large data.',
  },
  // 映射 couchbase 数据库类型，包括其名称、图标路径和描述信息
  couchbase: {
    label: 'Couchbase',
    icon: '/icons/couchbase.png',
    desc: 'High-performance NoSQL document database with distributed architecture.',
  },
  // 映射 postgresql 数据库类型，包括其名称、图标路径和描述信息
  postgresql: {
    label: 'PostgreSQL',
    icon: '/icons/postgresql.png',
    desc: 'Powerful open-source relational database with extensibility and SQL standards.',
  },
  // 映射 vertica 数据库类型，包括其名称、图标路径和描述信息
  vertica: {
    label: 'Vertica',
    icon: '/icons/vertica.png',
    desc: 'Columnar database optimized for analytics with high performance.',
  },
};
    {
      // 描述 Vertica 数据仓库系统，提供强一致性、ACID兼容的 SQL 数据仓库功能，设计用于处理当今数据驱动世界的规模和复杂性。
      vertica: {
        desc: 'Vertica is a strongly consistent, ACID-compliant, SQL data warehouse, built for the scale and complexity of today’s data-driven world.',
      },
      // 描述 Spark，一个统一的引擎，用于大规模数据分析。
      spark: {
        label: 'Spark',
        icon: '/icons/spark.png',
        desc: 'Unified engine for large-scale data analytics.'
      },
      // 描述 Hive，一个分布式容错数据仓库系统。
      hive: {
        label: 'Hive',
        icon: '/icons/hive.png',
        desc: 'A distributed fault-tolerant data warehouse system.'
      },
      // 描述 Space，知识分析系统。
      space: {
        label: 'Space',
        icon: '/icons/knowledge.png',
        desc: 'knowledge analytics.'
      },
      // 描述 TuGraph，由蚂蚁集团和清华大学联合开发的高性能图数据库。
      tugraph: {
        label: 'TuGraph',
        icon: '/icons/tugraph.png',
        desc: 'TuGraph is a high-performance graph database jointly developed by Ant Group and Tsinghua University.'
      }
    }
# 代码结尾处的分号，可能是一个语法错误，应该在代码中找到对应的起始位置进行修正
```