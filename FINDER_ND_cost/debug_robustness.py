"""
调试脚本：对比FINDER的rob值和实际LCC
"""
import networkx as nx
import numpy as np
from FINDER import FINDER

# 加载模型
dqn = FINDER()
MODEL_FILE = './models/nrange_30_50_iter_134100.ckpt'
dqn.LoadModel(MODEL_FILE)

# 测试一个图
G = nx.read_gml('saved_graphs/test_pl_20_40/graph_000.gml', destringizer=int)

# 添加weight
for node in G.nodes():
    if 'weight' not in G.nodes[node]:
        G.nodes[node]['weight'] = 1

N0 = G.number_of_nodes()
print(f"图规模: {N0} 节点, {G.number_of_edges()} 边\n")

# RF方法
dqn.ClearTestGraphs()
dqn.InsertGraph(G, is_test=True)
rob_rf, sol_rf = dqn.GetSol(gid=0, step=1)

print("="*60)
print("RF方法")
print("="*60)
print(f"FINDER的rob值: {rob_rf:.4f}")
print(f"序列长度: {len(sol_rf)}")
print(f"前10个节点: {sol_rf[:10]}")

# 手动计算LCC曲线
GG = G.copy()
lcc_values = [1.0]  # 初始LCC=1
removed_count = 0

for action in sol_rf:
    if action < N0 and action in GG.nodes():
        GG.remove_node(action)
        removed_count += 1
        
        if GG.number_of_nodes() == 0:
            lcc_values.append(0.0)
        else:
            largest_cc = max(nx.connected_components(GG), key=len)
            lcc = len(largest_cc) / N0
            lcc_values.append(lcc)

avg_lcc_rf = np.mean(lcc_values)
print(f"平均LCC值: {avg_lcc_rf:.4f}")
print(f"LCC曲线: {lcc_values[:10]}")

# HDA方法
rob_hda, sol_hda = dqn.HXAWithSol(G, 'HDA')

print("\n" + "="*60)
print("HDA方法")
print("="*60)
print(f"FINDER的rob值: {rob_hda:.4f}")
print(f"序列长度: {len(sol_hda)}")
print(f"前10个节点: {sol_hda[:10]}")

# 手动计算LCC曲线
GG = G.copy()
lcc_values = [1.0]
removed_count = 0

for action in sol_hda:
    if action < N0 and action in GG.nodes():
        GG.remove_node(action)
        removed_count += 1
        
        if GG.number_of_nodes() == 0:
            lcc_values.append(0.0)
        else:
            largest_cc = max(nx.connected_components(GG), key=len)
            lcc = len(largest_cc) / N0
            lcc_values.append(lcc)

avg_lcc_hda = np.mean(lcc_values)
print(f"平均LCC值: {avg_lcc_hda:.4f}")
print(f"LCC曲线: {lcc_values[:10]}")

print("\n" + "="*60)
print("对比总结")
print("="*60)
print(f"{'方法':<10} {'FINDER rob值':<15} {'平均LCC':<15} {'攻击效果'}")
print("-"*60)
print(f"{'RF':<10} {rob_rf:<15.4f} {avg_lcc_rf:<15.4f} {'rob低但LCC高=差'}")
print(f"{'HDA':<10} {rob_hda:<15.4f} {avg_lcc_hda:<15.4f} {'rob高但LCC低=好'}")
print("\n结论: FINDER的rob值是MVC鲁棒性，不是网络攻击LCC！")
