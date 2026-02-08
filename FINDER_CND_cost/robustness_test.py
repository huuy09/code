"""
网络鲁棒性测试脚本 - 修正版
===========================

⚠️ 重要说明：
FINDER的训练目标确实是网络鲁棒性攻击（reward = -最大连通分量）
但RF表现可能不如启发式方法的原因：

1. **训练数据不匹配**：
   - RF在degree_cost加权图上训练（节点权重∝度数）
   - 当前测试图是均匀权重（所有节点weight=1）
   - RF学到的策略针对加权图优化

2. **模型泛化能力**：
   - 30-50节点的模型测试30-50节点的图
   - 可能存在过拟合

功能说明：
1. 比较不同方法在网络攻击场景下的表现
   - RF: FINDER强化学习方法（针对加权图训练）
   - HDA/HBA/HCA/HPRA: 启发式中心性方法（通用）
2. 模拟攻击过程，计算网络鲁棒性指标（LCC和PC）
3. 生成对比图，展示不同方法的攻击效果

作者：优化版本
日期：2026-02-02
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy.linalg
from FINDER import FINDER
import datetime

# ==========================================
# 环境配置
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 黑体、微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# 全局参数设置
# ==========================================
ROOT_GML_DIR = 'saved_graphs/degree_cost_weighted'  # 加权图目录
OUT_ROOT = './attack_outputs/degree_cost_weighted'  # 输出目录
METHODS = ['RF', 'HDA', 'HBA', 'HCA', 'HPRA']  # 测试的攻击方法
MAX_GRAPHS = 10  # 测试模式：处理前10个图（改为None处理全部）
MODEL_FILE = './models/nrange_30_50_iter_134100.ckpt'  # 模型文件

os.makedirs(OUT_ROOT, exist_ok=True)

# ==========================================
# 初始化FINDER模型（全局只加载一次）
# ==========================================
print("="*60)
print("初始化 FINDER 模型...")
print("="*60)
dqn = FINDER()
dqn.LoadModel(MODEL_FILE)
print(f"✓ 已加载模型: {MODEL_FILE}\n")


# ==========================================
# 函数1: 生成攻击序列
# ==========================================
def generate_attack_sequences(gml_path):
    """为单个图生成所有方法的攻击序列（包括rob值）"""
    gid = os.path.basename(gml_path).replace('.gml', '')
    OUT_DIR = os.path.join(OUT_ROOT, gid)
    os.makedirs(OUT_DIR, exist_ok=True)

    G = nx.read_gml(gml_path, destringizer=int)
    print(f"  图规模: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # ⚠️ 不要修改权重！保留原始的degree_cost权重
    # 验证权重存在
    if not all('weight' in G.nodes[node] for node in G.nodes()):
        print("  ⚠️ 警告：图中缺少节点权重")
        for node in G.nodes():
            if 'weight' not in G.nodes[node]:
                G.nodes[node]['weight'] = 1
    
    sequences = {}
    rob_values = {}  # 保存每个方法的rob值
    
    for m in METHODS:
        if m == 'RF':
            # RF强化学习方法
            dqn.ClearTestGraphs()  # 清空测试集
            dqn.InsertGraph(G, is_test=True)  # 插入当前图
            rob, sol = dqn.GetSol(gid=0, step=1)  # 现在gid=0是当前图
            sequences[m] = sol
            rob_values[m] = rob  # 保存FINDER的加权rob值
            print(f"    {m:6s}: 序列长度={len(sol)}, rob={rob:.4f} ({rob*100:.2f}%)")
        else:
            # 启发式方法
            rob, sol = dqn.HXAWithSol(G, m)
            sequences[m] = sol
            rob_values[m] = rob  # 保存FINDER的加权rob值
            print(f"    {m:6s}: 序列长度={len(sol)}, rob={rob:.4f} ({rob*100:.2f}%)")
        
        # 保存序列
        pd.DataFrame({'Action': sequences[m]}).to_csv(
            os.path.join(OUT_DIR, f'{m}_seq.csv'), index=False)
    
    return G, sequences, rob_values


# ==========================================
# 函数2: 模拟攻击过程
# ==========================================
def simulate_attack(G, seq):
    """模拟攻击过程，计算LCC、PC和加权鲁棒性"""
    GG = G.copy()
    N0 = G.number_of_nodes()
    
    # 获取总权重
    total_weight = sum(G.nodes[node].get('weight', 1.0) for node in G.nodes())
    
    adj0 = nx.to_numpy_array(GG)
    pc0 = np.sum(scipy.linalg.expm(adj0)) - N0
    
    x_vals = [0.0]
    lcc_vals = [1.0]
    pc_vals = [1.0]
    rob_vals = [1.0]  # 加权鲁棒性：考虑节点权重的最大连通分量比例
    removed_count = 0
    removed_weight = 0.0
    
    for action in seq:
        if action < N0 and action in GG.nodes():
            # 记录删除节点的权重
            removed_weight += G.nodes[action].get('weight', 1.0)
            GG.remove_node(action)
            removed_count += 1
        else:
            continue
        
        if GG.number_of_nodes() == 0:
            lcc_vals.append(0.0)
            pc_vals.append(0.0)
            rob_vals.append(0.0)
        else:
            largest_cc = max(nx.connected_components(GG), key=len)
            lcc = len(largest_cc) / N0
            lcc_vals.append(lcc)
            
            # 计算加权鲁棒性：最大连通分量中节点的权重之和 / 总权重
            lcc_weight = sum(G.nodes[node].get('weight', 1.0) for node in largest_cc)
            weighted_rob = lcc_weight / total_weight if total_weight > 0 else 0.0
            rob_vals.append(weighted_rob)
            
            adj = nx.to_numpy_array(GG)
            curr_pc = np.sum(scipy.linalg.expm(adj)) - GG.number_of_nodes()
            pc = curr_pc / pc0 if pc0 > 1e-9 else 0.0
            pc_vals.append(max(pc, 1e-8))
        
        # X轴改为累积删除的节点权重比例（而不是数量比例）
        x_vals.append(removed_weight / total_weight)
    
    return x_vals, lcc_vals, pc_vals, rob_vals


# ==========================================
# 主流程
# ==========================================
def main():
    print("="*60)
    print("开始批量测试")
    print("="*60)
    
    # 检查图文件目录是否存在
    if not os.path.exists(ROOT_GML_DIR):
        print(f"\n❌ 错误：图文件目录不存在: {ROOT_GML_DIR}")
        print(f"\n请提供包含.gml文件的目录，例如：")
        print(f"  - saved_graphs/test_pl_20_40/")
        print(f"  - 或使用synthetic数据集生成GML文件\n")
        return
    
    gml_files = [os.path.join(ROOT_GML_DIR, f)
                 for f in os.listdir(ROOT_GML_DIR) if f.endswith('.gml')]
    gml_files = sorted(gml_files)
    
    if len(gml_files) == 0:
        print(f"\n❌ 错误：在 {ROOT_GML_DIR} 中未找到.gml文件")
        print(f"\n请确保目录中包含网络图文件（.gml格式）\n")
        return
    
    if MAX_GRAPHS is not None:
        gml_files = gml_files[:MAX_GRAPHS]
        print(f"⚠️ 测试模式：只处理前 {MAX_GRAPHS} 个图\n")
    
    print(f"找到 {len(gml_files)} 个图文件\n")
    
    record_x = {m: [] for m in METHODS}
    record_lcc = {m: [] for m in METHODS}
    record_pc = {m: [] for m in METHODS}
    record_rob = {m: [] for m in METHODS}  # 保存FINDER rob值（标量）
    record_rob_curve = {m: [] for m in METHODS}  # 保存加权鲁棒性曲线
    
    for idx, gml_path in enumerate(gml_files, 1):
        gid = os.path.basename(gml_path).replace('.gml', '')
        print(f"\n[{idx}/{len(gml_files)}] 处理: {gid}")
        
        try:
            G, sequences, rob_values = generate_attack_sequences(gml_path)
            
            for m in METHODS:
                x, lcc, pc, rob_curve = simulate_attack(G, sequences[m])
                record_x[m].append(x)
                record_lcc[m].append(lcc)
                record_pc[m].append(pc)
                record_rob[m].append(rob_values[m])  # 保存FINDER rob值
                record_rob_curve[m].append(rob_curve)  # 保存加权鲁棒性曲线
        except Exception as e:
            print(f"  ❌ 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("开始生成对比图")
    print("="*60)
    
    # ==========================================
    # 计算平均曲线（插值对齐）并绘图
    # ==========================================
    from scipy.interpolate import interp1d

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 统一的删除权重比例 q（横轴）
    # 限制到0.6，因为大部分方法在这之前就停止了
    NUM_Q = 50
    q_points = np.linspace(0, 0.6, NUM_Q)

    colors = {
        'RF': 'red',
        'HDA': 'blue',
        'HBA': 'green',
        'HCA': 'orange',
        'HPRA': 'purple'
    }

    # 先输出FINDER rob值统计
    print("\n各方法的FINDER rob值（加权累积指标，越低越好）：")
    print("-" * 50)
    method_rob_values = {}
    for m in METHODS:
        if len(record_rob[m]) > 0:
            rob_mean = np.mean(record_rob[m])
            rob_std = np.std(record_rob[m])
            method_rob_values[m] = rob_mean
            print(f"  {m:6s}: {rob_mean:.4f} ({rob_mean*100:.2f}%) ± {rob_std:.4f}")
    
    # 按rob值排序（rob值高的先画，rob值低的后画，这样rob值低的在视觉上更突出）
    methods_sorted = sorted(METHODS, key=lambda m: method_rob_values.get(m, 999), reverse=True)
    print(f"\n绘图顺序（按rob值从高到低）: {methods_sorted}")
    
    print("\n各方法的平均LCC值（简单平均，仅供参考）：")
    print("-" * 50)

    for m in methods_sorted:
        lcc_interp_all = []
        pc_interp_all = []
        rob_interp_all = []  # 加权鲁棒性曲线

        for x_vals, lcc_vals, pc_vals, rob_vals in zip(
            record_x[m], record_lcc[m], record_pc[m], record_rob_curve[m]
        ):
            x = np.array(x_vals)
            lcc = np.array(lcc_vals)
            pc = np.array(pc_vals)
            rob_curve = np.array(rob_vals)

            # 防止极端情况（长度不足）
            if len(x) < 2:
                continue

            # LCC 插值
            f_lcc = interp1d(
                x, lcc,
                kind='linear',
                bounds_error=False,
                fill_value=(lcc[0], lcc[-1])
            )
            lcc_q = f_lcc(q_points)
            lcc_interp_all.append(lcc_q)
            
            # 加权鲁棒性插值
            f_rob = interp1d(
                x, rob_curve,
                kind='linear',
                bounds_error=False,
                fill_value=(rob_curve[0], rob_curve[-1])
            )
            rob_q = f_rob(q_points)
            rob_interp_all.append(rob_q)

            # PC 插值
            f_pc = interp1d(
                x, pc,
                kind='linear',
                bounds_error=False,
                fill_value=(pc[0], pc[-1])
            )
            pc_q = f_pc(q_points)
            pc_interp_all.append(pc_q)

        if len(lcc_interp_all) == 0:
            print(f"  {m:6s}: 无有效数据")
            continue

        # 转为数组
        LCC_all = np.vstack(lcc_interp_all)
        ROB_all = np.vstack(rob_interp_all)
        PC_all = np.vstack(pc_interp_all)

        # 平均 & 标准差
        LCC_mean = np.mean(LCC_all, axis=0)
        LCC_std = np.std(LCC_all, axis=0)
        
        ROB_mean = np.mean(ROB_all, axis=0)
        ROB_std = np.std(ROB_all, axis=0)

        PC_mean = np.mean(PC_all, axis=0)
        PC_std = np.std(PC_all, axis=0)

        # 平均LCC（简单平均，仅供参考）
        avg_lcc = np.mean(LCC_mean)
        print(f"  {m:6s}: {avg_lcc:.4f}")

        color = colors.get(m, 'gray')
        
        # 获取FINDER rob值用于图例显示
        rob_mean_scalar = np.mean(record_rob[m]) if len(record_rob[m]) > 0 else 0.0

        # -------- GCC Size图（网络瓦解ND问题用LCC节点比例） --------
        axes[0].plot(
            q_points, LCC_mean,
            label=f'{m} (rob={rob_mean_scalar:.3f})',
            color=color,
            linewidth=2
        )
        axes[0].fill_between(
            q_points,
            LCC_mean - LCC_std,
            LCC_mean + LCC_std,
            color=color,
            alpha=0.2
        )

        # -------- PC 图 --------
        axes[1].plot(
            q_points, PC_mean,
            label=m,
            color=color,
            linewidth=2
        )
        axes[1].fill_between(
            q_points,
            np.maximum(PC_mean - PC_std, 1e-8),
            PC_mean + PC_std,
            color=color,
            alpha=0.2
        )

    # ==========================
    # 图表格式
    # ==========================
    axes[0].set_title("网络鲁棒性对比 - LCC指标", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("累积节点权重比例 (w)", fontsize=12)
    axes[0].set_ylabel("最大连通分量比例 (LCC/N)", fontsize=12)
    axes[0].set_xlim(0, 0.6)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("网络鲁棒性对比 - PC指标", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("累积节点权重比例 (w)", fontsize=12)
    axes[1].set_ylabel("网络鲁棒性对比 - PC指标", fontsize=12)
    axes[1].set_yscale("log")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(1e-3, 1.5)
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    # 保存
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    output_path = os.path.join(OUT_ROOT, f'robustness_comparison_interp_{timestamp}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # 同时保存PNG
    output_path_png = os.path.join(OUT_ROOT, f'robustness_comparison_interp_{timestamp}.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✓ 插值对齐完成")
    print(f"✓ 输出图表: {output_path}")
    print(f"✓ 输出图表: {output_path_png}")
    
    print("\n" + "="*60)
    print("✓ 完成！")
    print("="*60)
    print(f"  处理图数: {len(gml_files)}")
    print(f"  输出目录: {OUT_ROOT}")
    print(f"  对比图表: {output_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n\n❌ 程序出错: {str(e)}")
        import traceback
        traceback.print_exc()
