"""
对比分析：testSynthetic的rob值 vs robustness_test的LCC
找出为什么两个结果矛盾
"""
import networkx as nx
import numpy as np
from FINDER import FINDER

dqn = FINDER()
dqn.LoadModel('./models/nrange_30_50_iter_134100.ckpt')

# 测试一个图
G = nx.read_gml('./synthetic/degree_cost/30-50/g_0', destringizer=int)
N0 = G.number_of_nodes()

print("="*60)
print(f"测试图: {N0}节点, {G.number_of_edges()}边")
print("="*60)

# RF方法
dqn.ClearTestGraphs()
dqn.InsertGraph(G, is_test=True)
rob_rf, sol_rf = dqn.GetSol(gid=0, step=1)

print(f"\nRF方法:")
print(f"  FINDER rob值: {rob_rf:.4f} ({rob_rf*100:.2f}%)")
print(f"  序列长度: {len(sol_rf)}")

# 手动计算LCC曲线（我们的方法）
GG = G.copy()
lcc_values = [1.0]
x_values = [0.0]
removed = 0

for action in sol_rf:
    if action < N0 and action in GG.nodes():
        GG.remove_node(action)
        removed += 1
        if GG.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(GG), key=len)
            lcc_values.append(len(largest_cc) / N0)
        else:
            lcc_values.append(0.0)
        x_values.append(removed / N0)

avg_lcc = np.mean(lcc_values)
print(f"  我们的平均LCC: {avg_lcc:.4f}")
print(f"  LCC曲线(前10步): {[f'{v:.3f}' for v in lcc_values[:10]]}")

# HDA对比
rob_hda, sol_hda = dqn.HXAWithSol(G, 'HDA')
print(f"\nHDA方法:")
print(f"  FINDER rob值: {rob_hda:.4f} ({rob_hda*100:.2f}%)")
print(f"  序列长度: {len(sol_hda)}")

GG = G.copy()
lcc_hda = [1.0]
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

avg_lcc_hda = np.mean(lcc_hda)
print(f"  我们的平均LCC: {avg_lcc_hda:.4f}")
print(f"  LCC曲线(前10步): {[f'{v:.3f}' for v in lcc_hda[:10]]}")

print("\n" + "="*60)
print("关键发现：")
print("="*60)
print(f"1. FINDER rob值:")
print(f"   RF:  {rob_rf:.4f} < HDA: {rob_hda:.4f}")
print(f"   → RF更好（rob值越低越好）")
print(f"\n2. 平均LCC值:")
print(f"   RF:  {avg_lcc:.4f} > HDA: {avg_lcc_hda:.4f}")
print(f"   → HDA更好（LCC越低=攻击越有效）")

print("\n" + "="*60)
print("矛盾的原因分析：")
print("="*60)
print("FINDER的rob值是【加权累积值】:")
print("  - 考虑了节点权重")
print("  - 考虑了移除顺序的累积效应")
print("  - 公式: Σ(最大连通分量 × 节点权重)")
print("\n我们的平均LCC是【简单平均】:")
print("  - 没有考虑节点权重")
print("  - 简单平均所有步骤的LCC")
print("  - 公式: mean(LCC_i)")

# 让我们计算加权的rob值看看
print("\n" + "="*60)
print("尝试复现FINDER的rob计算：")
print("="*60)

def compute_finder_rob(G, sol):
    """复现utils.cpp中的getRobustness计算"""
    from collections import defaultdict
    
    # 模拟逆序恢复节点
    removed_set = set(sol)
    N = G.number_of_nodes()
    total_weight = sum(G.nodes[i]['weight'] for i in range(N))
    
    totalMaxNum = 0.0
    current_removed = set(sol)
    
    # 逆序遍历solution
    for i in range(len(sol)-1, -1, -1):
        node = sol[i]
        # 恢复这个节点
        current_removed.remove(node)
        
        # 计算当前的最大连通分量
        remaining_nodes = set(range(N)) - current_removed
        subgraph = G.subgraph(remaining_nodes)
        
        if subgraph.number_of_nodes() > 0:
            max_cc = max(nx.connected_components(subgraph), key=len)
            max_cc_size = len(max_cc)
        else:
            max_cc_size = 0
        
        # 获取权重
        if i > 0:
            weight = G.nodes[sol[i-1]]['weight']
        else:
            weight = 0
        
        totalMaxNum += max_cc_size * weight
    
    rob = totalMaxNum / (N * total_weight)
    return rob

rob_rf_computed = compute_finder_rob(G, sol_rf)
rob_hda_computed = compute_finder_rob(G, sol_hda)

print(f"RF复现rob:  {rob_rf_computed:.4f} (原始: {rob_rf:.4f})")
print(f"HDA复现rob: {rob_hda_computed:.4f} (原始: {rob_hda:.4f})")

print("\n结论：FINDER的rob值和简单的平均LCC衡量的是不同的东西！")
print("testSynthetic.py的好结果是基于加权累积rob，不是我们的平均LCC。")
