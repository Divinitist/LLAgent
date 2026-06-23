# LLAgent — 层级检索小说问答系统设计

## 概述

基于 RAG 的小说《圣王》（500 万字）问答 Agent。采用**层级 metadata 抽取 + 层级检索**方案，构造与阅读分离。

- **构造 (Offline Workflow)**：批处理 DAG，构建层级索引
- **阅读 (Online Agent)**：纯读不写，基于索引做问答

---

## 一、Agent 工具清单

| 工具 | 输入 | 输出 | 用途 |
|---|---|---|---|
| **`link_entities`** | query 文本 | `[{规范名, 类型, 置信度}]` | 从问题中抽实体，映射到规范名 |
| **`locate_entities`** | 实体名列表 | `{实体名 → [{层级, 节点ID, 出现次数}]}` | 查每个实体出现在哪些层级节点 |
| **`navigate`** | 层级(L1~L4) + 节点ID | `{父节点, 子节点列表, 本级metadata}` | 在层级树中上下导航 |
| **`get_metadata`** | 层级 + 节点ID | `{summary, entities, events, tags}` | 读取某节点的结构化元数据 |
| **`get_context`** | 层级(L0) + 节点ID + chunk 范围 | `[chunk_text]` | 取原文片段给生成用 |
| **`search`** | query + top_k | `[chunk_text]` | 全文/向量兜底搜索（实体链接失败时） |
| **`verify`** | 生成的答案 + 引用的节点ID | `{is_supported: bool, 疑点列表}` | 校验答案是否有原文依据 |

---

## 二、层级索引设计

### 层级结构

| 层级 | 粒度 | 数量（约） | Metadata 内容 |
|---|---|---|---|
| L4 | 每1000章 | 1-2 个 | `book_overview`, `main_characters` |
| L3 | 每100章 | 10+ 个 | `phase_summary`, `major_threads` |
| L2 | 每10章 | 100+ 个 | `arc_summary`, `key_events`, `new_entities` |
| L1 | 每章 | 1000+ 个 | `summary`(1-2句), `main_entities` |
| L0 | 自定义 chunk | 数千个 | 原文片段（无额外 metadata） |

### 检索策略（二选一或结合）

1. **自上而下硬路由**：先检 L1 确定范围 → 下钻 L2 → L3 → L4 → L0
2. **扁平化召回**：所有层级 metadata 统一索引，一次召回后按层级关系聚合排序

---

## 三、Workflow 产出的索引文件

Agent 的数据依赖：

```
数据文件                             给哪个工具
─────────────────────────────────────
entity_index.json                   → link_entities + locate_entities
  { "规范名": { "type": "人物",
      "aliases": ["杨奇","奇儿","小少爷"],
      "occurrences": { "L1": [1,5,9], "L2": [1], ... } } }

hierarchy_tree.json                 → navigate + get_metadata
  { "L4": [{ id:0, children:[0,1], summary:"..." }],
    "L3": [{ id:0, parent:0, children:[0..9], ... }],
    "L2": [...], "L1": [...] }

chunk_index.json                    → get_context
  { "L0": [{ id:0, chpt_id:1, chunk_id:0,
             text:"...", parent:"L1/1" }] }

raw_text.json (或分文件)             → search + verify
```

---

## 四、Workflow DAG 节点

### 阶段一：数据准备

```
节点1: 文本切分
  输入: 圣王.txt
  输出: L0 chunks (每 chunk ~512 tokens，按段落/章节边界切分)
  
节点2: 章节识别
  输入: 原始文本
  输出: L1 层级结构（章节列表 + 每章节起止位置）
  
节点3: 层级聚合
  输入: L1 章节列表
  输出: L2 (10章/组)、L3 (100章/组)、L4 (1000章/组) 的节点结构和父子关系
  
节点4: 实体抽取
  输入: 全量原文
  输出: 全量候选实体清单（人物/功法/丹药/地点/境界）
  
节点5: 实体清洗与对齐
  输入: 候选实体清单
  输出: entity_index（规范名、别名、类型、出现位置）
```

### 阶段二：Metadata 抽取（每级都需要 LLM 调用）

```
节点6a: L1 Metadata 抽取
  每章 → summary(1-2句)、main_entities(出现的主要人物)
  
节点6b: L2 Metadata 抽取
  每10章 → arc_summary(一段)、key_events(3-5条)、new_entities
  
节点6c: L3 Metadata 抽取
  每100章 → phase_summary、major_threads（贯穿百章的主线）
  
节点6d: L4 Metadata 抽取
  全书 → book_overview(几段)、main_characters(主要人物列表)
```

### 阶段三：导出

```
节点7: 索引打包
  输出: hierarchy_tree.json + chunk_index.json + entity_index.json
```

---

## 五、关键设计决策

### 构造 vs 阅读分离

| | 构造 (Workflow) | 阅读 (Agent) |
|---|---|---|
| 运行时机 | 一次性离线 | 每次在线问答 |
| 行为 | 批处理 DAG，确定性 | 动态检索 + 生成 |
| 数据权限 | 读写全量数据 | 只读索引 |
| 工具 | 任意（Python, LLM, shell） | 仅限上述 7 个工具 |

### Metadata 分层原则

- 下层继承上层的实体列表（L1 实体是 L2 实体的子集）
- summary 逐层抽象：L1 是具体情节 → L2 是故事弧线 → L3 是宏观阶段 → L4 是全书主题
- 实体倒排索引是跨层级快速定位的核心

### ChatGPT Memory (Dreaming) 架构参考

ChatGPT 的跨 session memory（~1500 words 纯文本摘要，注入 system prompt）在小量记忆场景有效，但不适用于 500 万字小说。本系统的**结构化层级索引 + 选择性检索**是其自然扩展。

---

## 六、后续待定

1. Metadata 抽取是否只做 Tier 1（摘要+实体）还是扩展到 Tier 2（情节标签/境界时间线）
2. 检索策略选：自上而下硬路由 / 扁平召回 + 聚合 / query routing 动态决定
3. 层级索引的物理存储方案（JSON 文件 / SQLite / 轻量嵌入式 DB）
4. 同义实体合并的策略（规则 + LLM 辅助 vs 全自动聚类）
