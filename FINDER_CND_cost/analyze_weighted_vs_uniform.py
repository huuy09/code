"""
详细分析：加权图vs均匀图上的RF表现
"""
import networkx as nx
import numpy as np
from FINDER import FINDER
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载模型
dqn = FINDER()
dqn.LoadModel('./models/nrange_30_50_iter_134100.ckpt')

# 测试同一个图的两个版本
print("="*60)
print("对比分析：加权图 vs 均匀权重图")
print("="*60)

# 1. 加权图（原始degree_cost）
G_weighted = nx.read_gml('saved_graphs/degree_cost_weighted/graph_000.gml', destringizer=int)
print(f"\n加权图（degree_cost）:")
print(f"  节点数: {G_weighted.number_of_nodes()}, 边数: {G_weighted.number_of_edges()}")
weights = [G_weighted.nodes[i]['weight'] for i in range(min(5, G_weighted.number_of_nodes()))]
print(f"  前5个节点权重: {[f'{w:.3f}' for w in weights]}")

# 2. 均匀权重图
G_uniform = nx.read_gml('saved_graphs/test_pl_20_40/graph_000.gml', destringizer=int)
print(f"\n均匀权重图:")
print(f"  节点数: {G_uniform.number_of_nodes()}, 边数: {G_uniform.number_of_edges()}")
weights_uniform = [G_uniform.nodes[i]['weight'] for i in range(min(5, G_uniform.number_of_nodes()))]
print(f"  前5个节点权重: {[f'{w:.3f}' for w in weights_uniform]}")

# 测试RF
print("\n" + "="*60)
print("RF方法测试")
print("="*60)

# 加权图
dqn.ClearTestGraphs()
dqn.InsertGraph(G_weighted, is_test=True)
rob_weighted, sol_weighted = dqn.GetSol(gid=0, step=1)

# 均匀图
dqn.ClearTestGraphs()
dqn.InsertGraph(G_uniform, is_test=True)
rob_uniform, sol_uniform = dqn.GetSol(gid=0, step=1)

print(f"\n加权图: rob={rob_weighted:.4f}, 序列长度={len(sol_weighted)}")
print(f"  前10个节点: {sol_weighted[:10]}")
print(f"\n均匀图: rob={rob_uniform:.4f}, 序列长度={len(sol_uniform)}")
print(f"  前10个节点: {sol_uniform[:10]}")

# 计算实际LCC
def compute_lcc_curve(G, sol):
    GG = G.copy()
    N0 = G.number_of_nodes()
    lcc_vals = [1.0]
    
    for action in sol:
        if action < N0 and action in GG.nodes():
            GG.remove_node(action)
            if GG.number_of_nodes() > 0:
                largest_cc = max(nx.connected_components(GG), key=len)
                lcc_vals.append(len(largest_cc) / N0)
            else:
                lcc_vals.append(0.0)
    
    return lcc_vals

lcc_weighted = compute_lcc_curve(G_weighted, sol_weighted)
lcc_uniform = compute_lcc_curve(G_uniform, sol_uniform)

print(f"\n加权图平均LCC: {np.mean(lcc_weighted):.4f}")
print(f"均匀图平均LCC: {np.mean(lcc_uniform):.4f}")

# HDA对比
print("\n" + "="*60)
print("HDA方法对比")
print("="*60)

rob_hda_w, sol_hda_w = dqn.HXAWithSol(G_weighted, 'HDA')
rob_hda_u, sol_hda_u = dqn.HXAWithSol(G_uniform, 'HDA')

lcc_hda_w = compute_lcc_curve(G_weighted, sol_hda_w)
lcc_hda_u = compute_lcc_curve(G_uniform, sol_hda_u)

print(f"\n加权图HDA: rob={rob_hda_w:.4f}, 平均LCC={np.mean(lcc_hda_w):.4f}")
print(f"均匀图HDA: rob={rob_hda_u:.4f}, 平均LCC={np.mean(lcc_hda_u):.4f}")

# 绘图对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：加权图
x_rf_w = np.linspace(0, 1, len(lcc_weighted))
x_hda_w = np.linspace(0, 1, len(lcc_hda_w))

axes[0].plot(x_rf_w, lcc_weighted, 'r-', linewidth=2, marker='o', markersize=4,
             label=f'RF (LCC={np.mean(lcc_weighted):.3f})')
axes[0].plot(x_hda_w, lcc_hda_w, 'b-', linewidth=2, marker='s', markersize=4,
             label=f'HDA (LCC={np.mean(lcc_hda_w):.3f})')
axes[0].set_xlabel('移除节点比例', fontsize=12)
axes[0].set_ylabel('LCC / N₀', fontsize=12)
axes[0].set_title('加权图（degree_cost）\nRF仍然表现较差', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：均匀图
x_rf_u = np.linspace(0, 1, len(lcc_uniform))
x_hda_u = np.linspace(0, 1, len(lcc_hda_u))

axes[1].plot(x_rf_u, lcc_uniform, 'r-', linewidth=2, marker='o', markersize=4,
             label=f'RF (LCC={np.mean(lcc_uniform):.3f})')
axes[1].plot(x_hda_u, lcc_hda_u, 'b-', linewidth=2, marker='s', markersize=4,
             label=f'HDA (LCC={np.mean(lcc_hda_u):.3f})')
axes[1].set_xlabel('移除节点比例', fontsize=12)
axes[1].set_ylabel('LCC / N₀', fontsize=12)
axes[1].set_title('均匀权重图\nRF表现仍然较差', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./attack_outputs/weighted_vs_uniform_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ 对比图已保存: ./attack_outputs/weighted_vs_uniform_comparison.png")

print("\n" + "="*60)
print("结论")
print("="*60)
print("即使在训练使用的加权图（degree_cost）上测试，")
print("RF的表现仍然不如简单的HDA启发式方法。")
print("\n可能的原因:")
print("1. 模型可能过拟合或训练不充分")
print("2. 测试图与训练图分布不同")
print("3. DQN的探索-利用平衡问题")
print("4. 特征表示或网络结构的局限性")
print("="*60)

import os
os.system('start ./attack_outputs/weighted_vs_uniform_comparison.png')
