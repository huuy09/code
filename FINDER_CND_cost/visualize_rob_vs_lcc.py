"""
可视化：对比FINDER rob值与实际LCC曲线
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from FINDER import FINDER

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载模型
dqn = FINDER()
MODEL_FILE = './models/nrange_30_50_iter_134100.ckpt'
dqn.LoadModel(MODEL_FILE)

# 测试一个图
G = nx.read_gml('saved_graphs/test_pl_20_40/graph_000.gml', destringizer=int)
for node in G.nodes():
    if 'weight' not in G.nodes[node]:
        G.nodes[node]['weight'] = 1

N0 = G.number_of_nodes()

# RF方法
dqn.ClearTestGraphs()
dqn.InsertGraph(G, is_test=True)
rob_rf, sol_rf = dqn.GetSol(gid=0, step=1)

GG = G.copy()
lcc_rf = [1.0]
x_rf = [0.0]
removed = 0

for action in sol_rf:
    if action < N0 and action in GG.nodes():
        GG.remove_node(action)
        removed += 1
        if GG.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(GG), key=len)
            lcc_rf.append(len(largest_cc) / N0)
        else:
            lcc_rf.append(0.0)
        x_rf.append(removed / N0)

# HDA方法
rob_hda, sol_hda = dqn.HXAWithSol(G, 'HDA')

GG = G.copy()
lcc_hda = [1.0]
x_hda = [0.0]
removed = 0

for action in sol_hda:
    if action < N0 and action in GG.nodes():
        GG.remove_node(action)
        removed += 1
        if GG.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(GG), key=len)
            lcc_hda.append(len(largest_cc) / N0)
        else:
            lcc_hda.append(0.0)
        x_hda.append(removed / N0)

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_rf, lcc_rf, 'r-', linewidth=2, marker='o', markersize=4,
        label=f'RF (FINDER rob={rob_rf:.3f}, 平均LCC={np.mean(lcc_rf):.3f})')
ax.plot(x_hda, lcc_hda, 'b-', linewidth=2, marker='s', markersize=4,
        label=f'HDA (FINDER rob={rob_hda:.3f}, 平均LCC={np.mean(lcc_hda):.3f})')

ax.set_xlabel('移除节点比例 q', fontsize=12)
ax.set_ylabel('LCC / N₀', fontsize=12)
ax.set_title('FINDER rob值 ≠ 网络攻击LCC\n(rob值低但LCC高 = 攻击效果差)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 添加注释
ax.text(0.5, 0.7, 
        '⚠️ 注意:\n'
        'FINDER的rob值是MVC鲁棒性\n'
        '不是网络攻击的LCC!\n\n'
        'RF: rob低(0.33)但LCC高(0.54)→攻击差\n'
        'HDA: rob高(0.45)但LCC低(0.45)→攻击好',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('./attack_outputs/pl_20_40_fixed/rob_vs_lcc_explanation.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: ./attack_outputs/pl_20_40_fixed/rob_vs_lcc_explanation.png")

import os
os.system('start ./attack_outputs/pl_20_40_fixed/rob_vs_lcc_explanation.png')
