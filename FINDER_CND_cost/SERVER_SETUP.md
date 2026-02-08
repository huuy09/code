# FINDER_CND_cost 服务器环境配置指南

## 系统要求

### 操作系统
- **推荐**: Linux (Ubuntu 18.04/20.04/22.04)
- **也支持**: CentOS 7+, macOS
- **不推荐**: Windows (编译复杂)

### 硬件要求
- **CPU**: 支持AVX2指令集（现代CPU均支持）
- **内存**: 至少 8GB RAM（推荐 16GB+）
- **磁盘**: 至少 10GB 可用空间
- **GPU**: 可选（TensorFlow可用GPU加速，但CPU版本也可以）

---

## 必需软件和库

### 1. Python 环境
```bash
Python 3.7 或 3.8 （推荐 3.8）
```

### 2. C++ 编译器
**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential g++ gcc

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install gcc-c++
```

**需要支持 C++11 标准**

### 3. Python包依赖

#### 核心依赖:
```bash
# Cython - Python到C/C++编译器
pip install Cython==0.29.32

# TensorFlow - 深度学习框架（1.x版本）
pip install tensorflow==1.15.5  # CPU版本
# 或
pip install tensorflow-gpu==1.15.5  # GPU版本

# NumPy - 数值计算
pip install numpy==1.19.5

# NetworkX - 图论库
pip install networkx==2.5

# tqdm - 进度条
pip install tqdm

# pandas - 数据处理
pip install pandas

# SciPy - 科学计算
pip install scipy
```

#### 可选依赖（用于可视化和测试）:
```bash
pip install matplotlib
pip install seaborn
```

### 4. 完整requirements.txt
创建文件 `requirements.txt`:
```
Cython==0.29.32
tensorflow==1.15.5
numpy==1.19.5
networkx==2.5
tqdm
pandas
scipy
matplotlib
```

安装:
```bash
pip install -r requirements.txt
```

---

## 环境配置步骤

### 方案1: Conda虚拟环境（推荐）
```bash
# 1. 创建虚拟环境
conda create -n finder_cnd python=3.8

# 2. 激活环境
conda activate finder_cnd

# 3. 安装依赖
pip install Cython==0.29.32
pip install tensorflow==1.15.5
pip install numpy==1.19.5 networkx==2.5 tqdm pandas scipy matplotlib

# 4. 验证安装
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import Cython; print(Cython.__version__)"
```

### 方案2: Python venv
```bash
# 1. 创建虚拟环境
python3.8 -m venv finder_env

# 2. 激活环境
source finder_env/bin/activate  # Linux/macOS

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 编译FINDER_CND_cost

### 1. 上传代码
```bash
# 上传整个FINDER_CND_cost目录到服务器
scp -r FINDER_CND_cost/ user@server:/path/to/workspace/
```

### 2. 编译
```bash
cd /path/to/workspace/FINDER_CND_cost

# 激活虚拟环境
conda activate finder_cnd  # 或 source finder_env/bin/activate

# 编译所有模块
python setup.py build_ext --inplace

# 验证编译
python -c "import FINDER; import mvc_env; import graph; print('✓ 编译成功')"
```

### 3. 常见编译问题

**问题1: `error: command 'gcc' failed`**
```bash
# 安装编译工具
sudo apt-get install build-essential python3-dev
```

**问题2: `fatal error: Python.h: No such file or directory`**
```bash
# 安装Python开发头文件
sudo apt-get install python3.8-dev
```

**问题3: `-std=c++11` 不支持**
```bash
# 更新gcc版本
sudo apt-get install gcc-7 g++-7
```

---

## 多目标优化配置

### 调整CN/ND权重

编辑 `src/lib/mvc_env.cpp` 第 307 行:
```cpp
double alpha = 0.5;  // 修改此值

// alpha = 0.0  -> 纯ND目标（最大连通分量）
// alpha = 0.5  -> 均衡（默认）
// alpha = 1.0  -> 纯CN目标（网络碎片化）
```

修改后需要重新编译:
```bash
python setup.py build_ext --inplace
```

---

## 训练配置

### 训练参数（FINDER.pyx 第30-60行）

关键参数:
```python
MAX_ITERATION = 1000000   # 最大训练迭代次数
NUM_MIN = 30              # 训练图最小节点数
NUM_MAX = 50              # 训练图最大节点数
BATCH_SIZE = 64           # 批次大小
LEARNING_RATE = 0.0001    # 学习率
MEMORY_SIZE = 500000      # 经验回放内存大小
N_STEP = 5                # N-step学习步数
```

### 修改训练参数
1. 编辑 `FINDER.pyx` 对应行
2. 重新编译: `python setup.py build_ext --inplace`

---

## 训练数据说明

### ✅ 无需准备数据集！

FINDER采用**在线生成合成图**的方式训练：

#### 自动生成机制
1. **训练图**: 每5000次迭代自动生成1000个新图
2. **验证图**: 初始化时生成200个验证图
3. **图类型**: Barabási-Albert (BA)模型（无标度网络）
4. **节点范围**: 30-50个节点（可配置）
5. **权重策略**: 基于节点度数的权重（degree-based）

#### 图生成配置

**FINDER.pyx 第70-72行:**
```python
self.g_type = 'barabasi_albert'  # 图类型
self.training_type = 'degree'    # 节点权重类型
```

**支持的图类型:**
- `'barabasi_albert'`: BA无标度网络（默认，推荐）
- `'erdos_renyi'`: ER随机图
- `'powerlaw'`: 幂律聚类图
- `'small-world'`: 小世界网络

**支持的权重类型:**
- `'degree'`: 节点权重 = 度数/最大度数（归一化）
- `'random'`: 随机权重 [0,1]
- `'degree_noise'`: 度数+高斯噪声

#### 训练数据流程

```python
# train.py 启动后自动执行：

# 1. 初始化时生成验证图（200个）
dqn.PrepareValidData()
# → 输出: Validation of HDA: 0.467...
# → 输出: Validation of HBA: 0.448...

# 2. 生成初始训练图（1000个）
dqn.gen_new_graphs(30, 50)  # 30-50节点

# 3. 训练过程中每5000次迭代自动更新训练图
# 代码中 if iter % 5000 == 0:
#     dqn.gen_new_graphs(NUM_MIN, NUM_MAX)
```

### 可选：使用真实网络数据

如需在真实网络上测试（不是训练），可以：

1. **准备.gml格式图文件**
```python
import networkx as nx
G = nx.read_gml('your_network.gml', destringizer=int)
```

2. **添加节点权重**（如果没有）
```python
# 基于度数的权重
degree = nx.degree(G)
max_degree = max(dict(degree).values())
for node in G.nodes():
    G.nodes[node]['weight'] = degree[node] / max_degree
```

3. **使用测试脚本**
```bash
# 修改 testReal.py 或 robustness_test.py
# 指定图文件路径进行测试
```

**注意**: 真实网络只用于**测试/评估**，训练仍使用合成图。

---

## 运行训练

### 启动训练
```bash
# 后台运行训练（数据会自动生成）
nohup python train.py > train.log 2>&1 &

# 查看日志
tail -f train.log

# 查看进程
ps aux | grep train.py
```

### 训练启动日志示例
```
generating validation graphs...
100%|██████████| 200/200 [00:06<00:00, 29.39it/s]
Validation of HDA: 0.4670382286557640
Validation of HBA: 0.4482178444140553

generating new training graphs...
100%|██████████| 1000/1000 [00:00<00:00, 2571.32it/s]

iter 0 eps 1.0 average size of vc: 23.45
...
```

### 训练输出
- **模型文件**: `./models/Model_barabasi_albert/nrange_30_50_iter_*.ckpt`
- **性能记录**: `./models/Model_barabasi_albert/ModelVC_30_50.csv`
- **日志**: 终端输出或 `train.log`

### 监控训练
```bash
# 查看最新模型
ls -lht models/Model_barabasi_albert/*.ckpt | head -n 5

# 查看训练进度（每300次迭代输出一次）
grep "iter" train.log | tail -n 20
```

---

## 测试验证

### 快速测试（在服务器上运行）
```bash
# 测试多目标优化功能
python test_multi_objective.py

# 预期输出
# ✓ 模块加载: 通过
# ✓ 初始化: 通过
# ✓ 训练流程: 通过
```

### 使用已训练模型测试
```python
from FINDER import FINDER

dqn = FINDER()
dqn.LoadModel('./models/Model_barabasi_albert/nrange_30_50_iter_300000.ckpt')
# 运行测试...
```

---

## 文件清单（需要上传）

### 必需文件
```
FINDER_CND_cost/
├── FINDER.pyx                    # 主模型代码
├── setup.py                      # 编译配置
├── train.py                      # 训练脚本
├── test_multi_objective.py       # 测试脚本
├── PrepareBatchGraph.pyx         # 批处理
├── graph.pyx, graph.pxd          # 图数据结构
├── mvc_env.pyx, mvc_env.pxd      # 环境（包含多目标reward）
├── nstep_replay_mem*.pyx/.pxd    # 经验回放
├── utils.pyx, utils.pxd          # 工具函数
├── graph_struct.pyx, graph_struct.pxd
└── src/lib/                      # C++源代码
    ├── mvc_env.cpp/.h            # ⭐ 多目标reward实现
    ├── graph.cpp/.h
    ├── PrepareBatchGraph.cpp/.h
    ├── nstep_replay_mem*.cpp/.h
    ├── utils.cpp/.h
    ├── graph_utils.cpp/.h
    ├── disjoint_set.cpp/.h
    └── decrease_strategy.cpp/.h
```

### 不需要上传的文件（会自动生成）
```
*.pyd          # Windows编译文件
*.so           # Linux编译文件
*.c, *.cpp     # Cython生成的C/C++代码（会自动生成）
build/         # 编译临时目录
__pycache__/   # Python缓存
```

---

## SSH服务器训练建议

### 1. 使用screen或tmux防止断线
```bash
# 使用screen
screen -S finder_training
python train.py
# Ctrl+A, D 分离会话

# 恢复会话
screen -r finder_training

# 或使用tmux
tmux new -s finder_training
python train.py
# Ctrl+B, D 分离会话
tmux attach -t finder_training
```

### 2. 定期保存检查点
代码已经配置为每300次迭代保存一次模型，无需额外配置。

### 3. 监控资源使用
```bash
# CPU和内存
htop

# GPU使用（如果有GPU）
nvidia-smi -l 1
```

### 4. 磁盘空间管理
```bash
# 检查模型文件大小
du -sh models/

# 定期清理旧模型（保留最近的N个）
ls -t models/Model_*/nrange_30_50_iter_*.ckpt.* | tail -n +50 | xargs rm
```

---

## 验证环境是否就绪

### 完整测试脚本
```bash
#!/bin/bash
echo "检查Python版本..."
python --version

echo "检查GCC版本..."
gcc --version

echo "检查依赖包..."
python -c "
import sys
packages = ['tensorflow', 'numpy', 'networkx', 'Cython', 'tqdm', 'pandas', 'scipy']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        print(f'✓ {pkg}: {ver}')
    except ImportError:
        print(f'✗ {pkg}: NOT FOUND')
        sys.exit(1)
"

echo "检查TensorFlow兼容性..."
python -c "
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print('✓ TensorFlow v1 兼容模式正常')
"

echo "✓ 环境检查完成！"
```

保存为 `check_env.sh`，运行:
```bash
chmod +x check_env.sh
./check_env.sh
```

---

## 快速开始总结

```bash
# 1. 创建环境
conda create -n finder_cnd python=3.8
conda activate finder_cnd

# 2. 安装依赖
pip install Cython==0.29.32 tensorflow==1.15.5 numpy==1.19.5 networkx==2.5 tqdm pandas scipy matplotlib

# 3. 上传代码
scp -r FINDER_CND_cost/ user@server:/path/to/workspace/

# 4. SSH登录服务器
ssh user@server
cd /path/to/workspace/FINDER_CND_cost
conda activate finder_cnd

# 5. 编译
python setup.py build_ext --inplace

# 6. 测试
python test_multi_objective.py

# 7. 训练
nohup python train.py > train.log 2>&1 &
tail -f train.log
```

---

## 故障排除

### TensorFlow 1.15安装失败
```bash
# 尝试清华镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.15.5
```

### 编译时内存不足
```bash
# 减少并行编译数
python setup.py build_ext --inplace -j1
```

### import错误: DLL load failed
```bash
# 确保在激活的虚拟环境中
which python
# 应该指向虚拟环境的python
```

---

## 多目标优化特性说明

**FINDER_CND_cost 的独特之处:**
- 同时优化 CN（Component Number Density）和 ND（Max Component）
- alpha参数控制两个目标的权重
- 适合探索不同攻击策略对网络的影响

**建议训练多个alpha值:**
- alpha=0.3: 偏重最大连通分量
- alpha=0.5: 均衡策略（默认）
- alpha=0.7: 偏重网络碎片化

每个alpha值需要单独训练一个模型。
