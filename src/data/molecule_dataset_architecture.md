# Molecule Dataset 数据管道架构解析

> 本文档解析 `src/data/molecule_dataset.py` 的代码架构。
>
> **学习路径**：先读 [§1 总览](#1-总览) 建立整体印象 → [§2 核心原理](#2-核心原理) 理解数学基础 → [§3 代码实现](#3-代码实现) 追踪数据流 → [§4 模块依赖](#4-模块依赖) 理解调用关系 → [§5 设计决策](#5-设计决策) 理解权衡取舍。

---

## 1. 总览

### 1.1 四个核心组件

```
MoleculeTokenizer              ← 分词器：SMILES 字符串 ↔ token 整数序列
    ↓
MoleculeDataset (Dataset)      ← 数据集：加载文件 + RDKit 验证 + 样本读取
    ↓
create_data_loaders()          ← 工厂函数：构建 PyTorch DataLoader
    ↓
validate_smiles_internal()      ← 工具函数：RDKit 验证 SMILES 合法性
```

### 1.2 数据流总图

```
SMILES 字符串: "CCO" (乙醇)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ MoleculeTokenizer.encode()                               │
│   "C" → 52, "C" → 52, "O" → 57, "<eos>" → 3, "<pad>" → 0│
│   最大长度 512，不足补 <pad>                             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Token IDs: (52, 52, 57, 3, 0, 0, ..., 0)  ← Tuple[int, ...]
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ MoleculeDataset.__getitem__(idx)                        │
│   → (input_ids: Tensor, labels_tensor: Tensor)          │
│   input_ids: shape (512,) dtype=torch.long              │
│   labels: shape (num_labels,) dtype=torch.float         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ create_data_loaders() → DataLoader                      │
│   batch: (B, 512) tensor                               │
└─────────────────────────────────────────────────────────┘
```

### 1.3 参数一览

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `max_length` | 512 | token 序列最大长度 |
| `validate_smiles` | True | 是否用 RDKit 验证 SMILES |
| `task_type` | "regression" | 任务类型（"regression" 或 "classification"）|
| `batch_size` | 32 | 每批样本数 |
| `num_workers` | 4 | DataLoader 并行加载进程数 |

### 1.4 词汇表结构

```python
# special_token_tuple: 特殊标记（indices 0-3）
0: "<pad>"   # 填充标记
1: ">"       # 未知字符（对应 "?" 在原文中，但实际 token 是 ">"）
2: "<bos>"   # 句子开始
3: "<eos>"   # 句子结束

# smiles_token_tuple: SMILES 字符（indices 4-67）
# 包含：括号 "()[]"、双键 "="、三键 "#"、数字 "0-9"、电荷 "+-"、元素符号 "C" "N" "O" "S" "P" "F" "Cl" "Br" "I" "Si" "Se" "Te" "At" 等

# 总词汇表大小: 4 (特殊) + 64 (SMILES) = 68 tokens
```

---

## 2. 核心原理

### 2.1 SMILES 分词策略

SMILES 字符串的**双字符 token** 需要优先匹配（如 `Br`=溴、`Cl`=氯、`Si`=硅），否则按单字符匹配。

**匹配规则**（伪代码）：

```python
for i in range(len(smiles)):
    if i+1 < len(smiles) and smiles[i:i+2] in vocab:   # 双字符优先
        tokens.append(vocab[smiles[i:i+2]])
        i += 2
    elif smiles[i] in vocab:                              # 单字符次之
        tokens.append(vocab[smiles[i]])
        i += 1
    else:
        tokens.append(vocab["<pad>"])                     # 未知字符 → <pad>
        i += 1
```

**为什么双字符优先？** 因为 `Br`（溴原子）是一个语义单元，不应被拆成 `B`（硼）+ `r`。

### 2.2 LRU 缓存机制

```python
@functools.lru_cache(maxsize=500000)
def tokenize_smiles_cached_internal(smiles: str, vocab_id: int, max_length: int) -> Tuple[int, ...]:
    ...
```

**缓存设计**：
- **缓存大小**：500,000 条 SMILES → 足够缓存大多数分子数据集
- **缓存 key**：`(smiles, vocab_id, max_length)` 三元组
- **返回值**：Tuple[int, ...]（不可变，可哈希，满足 lru_cache 要求）
- **vocab_id 的作用**：不同词汇表有不同的 vocab_id，缓存互不干扰

**典型命中率**：分子数据集通常 < 10,000 条 SMILES，缓存可完全覆盖。

### 2.3 RDKit SMILES 验证

```python
def validate_smiles_internal(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False
```

**验证逻辑**：RDKit 的 `MolFromSmiles` 尝试解析 SMILES：
- 合法 → 返回 `mol` 对象（非 None）
- 非法（如括号不匹配、手性标记错误）→ 返回 `None` 或抛出异常

**为什么需要验证？**
- SMILES 字符串在数据集中可能存在噪声/错误
- RDKit 无法解析的分子无法计算特征
- 过滤掉无效样本可以避免训练崩溃

### 2.4 多格式文件加载

```python
# CSV: 第一列是 SMILES，其余列是标签
smiles_col = df.columns[0]
label_cols = df.columns[1:].tolist() if len(df.columns) > 1 else []

# JSON: [{"smiles": "...", "labels": [...]}, ...]
# TXT: "smiles,label1,label2,..." （逗号分隔）
```

**Data 数据类**：

```python
@dataclass
class Data:
    smiles: str           # 分子 SMILES 字符串
    labels: List[float]  # 标签列表（单标签或多标签）
```

---

## 3. 代码实现

### 3.1 MoleculeTokenizer

```python
class MoleculeTokenizer:
    def __init__(self, given_vocab_dict: Optional[Dict[str, int]] = None):
        # 如果没有提供词汇表，使用默认词汇表
        self.vocab = given_vocab_dict or default_vocab
        # 构建反向词汇表：token_id → token 字符串
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, smiles: str, max_length: int = 512) -> Tuple[int, ...]:
        # 调用带缓存的分词器
        return tokenize_smiles_cached_internal(smiles, id(self.vocab), max_length)

    def decode(self, token_ids: List[int]) -> str:
        # 过滤特殊标记，还原为 SMILES 字符串
        tokens = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, "")
            if token not in ["<pad>", ">", "<bos>", "<eos>"]:
                tokens.append(token)
        return "".join(tokens)
```

**encode → decode 往返验证**：

```python
>>> tokenizer = MoleculeTokenizer()
>>> original = "CCO"  # 乙醇
>>> encoded = tokenizer.encode(original)  # (52, 52, 57, 3, 0, 0, ...)
>>> decoded = tokenizer.decode(encoded)
>>> assert "CCO" in decoded  # decode 可能丢失 <eos> 但 SMILES 主体保留
```

### 3.2 MoleculeDataset

```python
class MoleculeDataset(Dataset):
    def __init__(
        self,
        data_file_path: str,
        task_type: str = "regression",
        max_length: int = 512,
        validate_smiles: bool = True,
    ):
        self.task_type = task_type
        self.max_length = max_length
        self.validate_smiles = validate_smiles
        self.tokenizer = MoleculeTokenizer()
        self.vocab_id = id(default_vocab)
        self.data = self.load_data_internal(data_file_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        item = self.data[idx]
        token_ids = tokenize_smiles_cached_internal(
            item.smiles, self.vocab_id, self.max_length
        )
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels_tensor = torch.tensor(
            item.labels,
            dtype=torch.float if self.task_type == "regression" else torch.long,
        )
        return input_ids, labels_tensor
```

**关键设计**：
- `id(default_vocab)` 作为缓存 key，确保不同 Dataset 实例缓存独立
- `task_type` 决定 labels 的 dtype：回归 → float，分类 → long

### 3.3 create_data_loaders()

```python
def create_data_loaders(
    train_path: str,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    batch_size: int = 32,
    task_type: str = "regression",
    max_length: int = 512,
    num_workers: int = 4,
) -> Tuple:
    def make_loader(path: str) -> DataLoader:
        dataset = MoleculeDataset(
            data_file_path=path,
            task_type=task_type,
            max_length=max_length
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(path == train_path),  # 仅训练集打乱
            num_workers=num_workers,
            pin_memory=True,  # 锁页内存，加速 CPU→GPU
        )

    train_loader = make_loader(train_path)
    val_loader = make_loader(val_path) if val_path and os.path.exists(val_path) else None
    test_loader = make_loader(test_path) if test_path and os.path.exists(test_path) else None
    return train_loader, val_loader, test_loader
```

**返回类型注解**：`Tuple`（非具体化）—— 返回 `(train_loader, val_loader, test_loader)` 三元组，val/test 可能为 None。

### 3.4 完整前向示例

```python
# 1. 创建 tokenizer
tokenizer = MoleculeTokenizer()

# 2. 加载数据集
dataset = MoleculeDataset(
    data_file_path="data/ESOL.csv",
    task_type="regression",
    max_length=512,
    validate_smiles=True
)

# 3. 获取单个样本
input_ids, labels = dataset[0]
# input_ids:  shape (512,) dtype=torch.long
# labels:     shape (1,) dtype=torch.float

# 4. 创建 DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 5. 遍历一个 batch
for batch_input_ids, batch_labels in loader:
    # batch_input_ids: (32, 512)
    # batch_labels:    (32, 1)
    break
```

---

## 4. 模块依赖

### 4.1 依赖树

```
molecule_dataset.py (本模块)
├── rdkit.Chem              ← SMILES 验证
├── pandas                  ← CSV 加载
├── json                    ← JSON 加载
├── torch                   ← Tensor + DataLoader
├── functools               ← lru_cache
└── os                      ← 文件路径检查

调用方 (train.py, eval.py):
└── create_data_loaders()   ← 主要入口
```

### 4.2 调用关系图

```
train.py / eval.py
    │
    ├── create_data_loaders(train_path, val_path, test_path, ...)
    │       │
    │       ├── MoleculeDataset(train_path)
    │       │       │
    │       │       ├── load_csv_internal() / load_json_internal() / load_txt_internal()
    │       │       │       │
    │       │       │       └── Data(smiles, labels)  ← dataclass
    │       │       │
    │       │       ├── validate_smiles_internal()  ← RDKit 过滤
    │       │       │
    │       │       └── tokenize_smiles_cached_internal()  ← LRU 缓存分词
    │       │
    │       └── DataLoader(dataset, batch_size, shuffle, ...)
    │               │
    │               └── MoleculeDataset.__getitem__(idx)
    │                       │
    │                       ├── tokenize_smiles_cached_internal()
    │                       └── torch.tensor(..., dtype=...)
    │
    └── train_loop(batch_input_ids, batch_labels)
            │
            └── model.forward(input_ids)  → BiMambaForPropertyPrediction
```

### 4.3 与模型模块的接口

```
molecule_dataset.py
    │
    └── output: (input_ids: Tensor, labels: Tensor)
                │
                ▼
        BiMambaForPropertyPrediction
                │
                ├── self.encoder(x)  → hidden_states (B, L, d_model)
                └── self.classifier(hidden_states)  → predictions (B, num_labels)

关键接口：
- input_ids: Tensor (B, L) dtype=torch.long
- labels:    Tensor (B, num_labels) dtype=torch.float (regression) / torch.long (classification)
```

---

## 5. 设计决策

### 5.1 为什么用 Tuple 而非 List 存储 token_ids？

```python
# 返回 Tuple[int, ...] 而非 List[int]
return tuple(tokens + [pad_token_id] * (max_length - len(tokens)))
```

**Tuple 的优势**：
- **不可变**：可以作为 dict key，支持 lru_cache 哈希
- **内存高效**：不可变对象 Python 可能共享内存
- **语义正确**：token 序列长度固定，不应被修改

### 5.2 为什么不直接在 MoleculeTokenizer 中缓存？

```python
# 方案 A（当前）：全局函数缓存
@functools.lru_cache(maxsize=500000)
def tokenize_smiles_cached_internal(smiles: str, vocab_id: int, max_length: int) -> Tuple[int, ...]

# 方案 B（替代）：实例方法缓存
class MoleculeTokenizer:
    def __init__(self):
        self._cache = {}
    def encode(self, smiles):
        if smiles not in self._cache:
            self._cache[smiles] = ...
        return self._cache[smiles]
```

**选择方案 A 的理由**：
- **进程共享**：lru_cache 是进程级缓存，多个 MoleculeTokenizer 实例共享
- **配置简单**：不需要手动管理缓存字典
- **内存管理**：lru_cache 自动淘汰最少使用项

### 5.3 为什么 validate_smiles 是可选的？

```python
validate_smiles: bool = True  # 默认验证
```

**理由**：
- 验证有计算开销（每个 SMILES 需 RDKit 解析）
- 某些场景数据已预处理过验证（如 MoleculeNet 下载好的数据集）
- 设置 `validate_smiles=False` 可加速数据加载

### 5.4 为什么不同任务类型标签 dtype 不同？

```python
labels_tensor = torch.tensor(
    item.labels,
    dtype=torch.float if self.task_type == "regression" else torch.long,
)
```

- **回归任务**：标签是连续值（溶解度、毒性数值）→ float32
- **分类任务**：标签是离散类别（0/1 二分类、N 分类）→ long（int64）

**代价**：如果标签类型不匹配，训练时会报错或产生隐式类型转换开销。

### 5.5 为什么用 pin_memory=True？

```python
DataLoader(..., pin_memory=True)
```

**原因**：
- `pin_memory=True` 将数据加载到**锁页内存**（不被换出）
- CPU→GPU 传输时，`cudaMemcpyAsync` 可以直接从锁页内存传输，绕过 CPU 内存拷贝
- 加速数据传输，通常可提升 10-20% 训练速度

**代价**：锁页内存占用增加（但 DataLoader 批处理是流式的，峰值内存可控）。

---

## 6. 与其他模块的对比

### 6.1 三个模块职责对比

| 模块 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `molecule_dataset.py` | 数据加载与分词 | CSV/JSON/TXT 文件路径 | DataLoader (B, L) |
| `bimamba.py` | 序列建模 | Tensor (B, L, d_model) | Tensor (B, d_model) |
| `bimamba_with_mamba_ssm.py` | 序列建模（mamba_ssm 封装）| Tensor (B, L, d_model) | Tensor (B, d_model) |

### 6.2 数据管道在训练流程中的位置

```
数据文件 (CSV/JSON/TXT)
    │
    ▼
create_data_loaders()
    │
    ├── train_loader  → train.py  → model.forward() → loss.backward() → optimizer.step()
    ├── val_loader    → eval.py   → model.forward() → metrics.compute()
    └── test_loader   → eval.py   → model.forward() → metrics.compute()
```

---

## 7. 附录

### 7.1 SMILES 标记一览

SMILES 分词使用的字符集（共 64 个 token）：

```
括号类: ( ) [ ] 
键类: = # %
数字类: 0 1 2 3 4 5 6 7 8 9
电荷类: + -
其他: / . : ; < > @
元素符号: B Br C Cl F H I N O P S Si Te Se At
```

### 7.2 DataLoader 训练 vs 验证行为对比

| 设置 | 训练集 | 验证/测试集 |
|------|--------|------------|
| `shuffle` | True（打乱）| False（保持顺序）|
| `num_workers` | 4 | 4 |
| `pin_memory` | True | True |
| 目的 | 增加泛化能力 | 保证可复现性 |

### 7.3 典型 MoleculeNet 数据格式

```csv
# ESOL.csv（回归任务，单标签）
smiles,label
CCO,-0.77
CCC,-0.87
...

# BBBP.csv（二分类）
smiles,label
CC(C)NCCNC(=O)c1ccc(Cl)nc1,1
CC(=O)Nc1ccc(O)cc1,0
...

# 多标签数据集（如 ClinTox）
smiles,label1,label2
CCO,0.0,0.0
CCC,1.0,0.0
...
```
