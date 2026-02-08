"""
网络鲁棒性测试脚本 - 基于节点权重比例的横坐标
==================================================

功能说明：
1. 使用多种方法生成攻击序列（RF强化学习 + 4种启发式方法）
2. 模拟攻击过程，计算网络鲁棒性指标（LCC和PC）
3. **横坐标使用累积节点权重比例（而非节点数量比例）**
4. 生成对比图，展示不同方法的攻击效果

作者：基于fifty-test-fixed.py修改
日期：2026-02-02
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy.linalg
import FINDER
import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==========================================
# 环境配置
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# 全局参数设置
# ==========================================
ROOT_GML_DIR = '../FINDER_CN_cost/saved_graphs/degree_cost_weighted'  # 图文件目录（使用ND_cost的带权重图）
OUT_ROOT = './attack_outputs/weighted_axis_test_cn_cost'  # 输出目录
METHODS = ['RF', 'HDA', 'HBA', 'HCA', 'HPRA']  # 测试的攻击方法
MAX_GRAPHS = 10  # 测试模式：处理前10个图（改为None处理全部）

os.makedirs(OUT_ROOT, exist_ok=True)

# ==========================================
# 初始化FINDER模型（全局只加载一次）
# ==========================================
print("="*60)
print("初始化 FINDER_CN_cost 模型...")
print("="*60)
dqn = FINDER.FINDER()

# 使用CN-cost模型
best_iter = 122100
best_model = f'./models/nrange_30_50_iter_{best_iter}.ckpt'
print(f"✓ 使用CN-cost模型: iter={best_iter}")

dqn.LoadModel(best_model)
print(f"✓ 已加载模型: {best_model}\n")


# ==========================================
# 函数1: 生成攻击序列
# ==========================================
def generate_attack_sequences(gml_path, graph_idx=0):
    """为单个图生成所有方法的攻击序列"""
    gid = os.path.basename(gml_path).replace('.gml', '')
    OUT_DIR = os.path.join(OUT_ROOT, gid)
    os.makedirs(OUT_DIR, exist_ok=True)

    G = nx.read_gml(gml_path, destringizer=int)
    # print(f"  图规模: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    dqn.InsertGraph(G, is_test=True)
    sequences = {}
    
    for m in METHODS:
        if m == 'RF':
            sol = dqn.GetSolution(gid=graph_idx, step=1)
            sequences[m] = sol
            # print(f"    {m:6s}: 序列长度={len(sol)}")
        else:
            # CN_cost的HXA只返回robustness，需要手动生成序列
            sol = []
            G_copy = G.copy()
            while nx.number_of_edges(G_copy) > 0:
                if m == 'HDA':
                    dc = nx.degree_centrality(G_copy)
                elif m == 'HBA':
                    dc = nx.betweenness_centrality(G_copy)
                elif m == 'HCA':
                    dc = nx.closeness_centrality(G_copy)
                elif m == 'HPRA':
                    dc = nx.pagerank(G_copy)
                keys = list(dc.keys())
                values = list(dc.values())
                maxTag = np.argmax(values)
                node = keys[maxTag]
                sol.append(int(node))
                G_copy.remove_node(node)
            sequences[m] = sol
            # print(f"    {m:6s}: 序列长度={len(sol)}")
        
        pd.DataFrame({'Action': sequences[m]}).to_csv(
            os.path.join(OUT_DIR, f'{m}_seq.csv'), index=False)
    
    return G, sequences


# ==========================================
# 函数2: 模拟攻击过程（使用权重比例作为横坐标）
# ==========================================
def simulate_attack_weighted(G, seq):
    """模拟攻击过程，横坐标使用累积权重比例
    
    终止条件：当覆盖边数 >= 初始边数*0.8 时停止（与RF的isTerminal一致）
    
    返回：x_vals, lcc_vals, ccd_vals, cumulative_reward
    """
    GG = G.copy()
    N0 = G.number_of_nodes()
    E0 = G.number_of_edges()  # 初始边数
    
    # 计算初始CCD Score（连通分量密度分数）
    # CCD = sum(n_i * (n_i - 1) / 2) for all connected components
    ccd0 = N0 * (N0 - 1) / 2.0  # 初始时整个图是一个连通分量
    
    # 计算总权重
    total_weight = sum(G.nodes[node].get('weight', 1.0) for node in G.nodes())
    
    # 初始化记录
    x_vals = [0.0]  # 累积权重比例
    lcc_vals = [1.0]
    ccd_vals = [1.0]  # CCD Score归一化值
    accumulated_weight = 0.0
    removed_edges_set = set()  # 跟踪已移除的唯一边
    edge_threshold = E0 * 0.8  # 80%边覆盖阈值
    
    # 用于计算cumulative reward（训练目标：min Σ(CCD_after × weight)）
    cumulative_reward = 0.0
    
    for action in seq:
        if action < N0 and action in GG.nodes():
            # 累加被移除节点的权重
            node_weight = G.nodes[action].get('weight', 1.0)
            accumulated_weight += node_weight
            
            # 正确计算新移除的边数
            neighbors = list(GG.neighbors(action))
            for neighbor in neighbors:
                edge = tuple(sorted([action, neighbor]))
                removed_edges_set.add(edge)
            
            # 移除节点
            GG.remove_node(action)
            num_covered_edges = len(removed_edges_set)
        else:
            continue
        
        # 计算当前网络状态
        if GG.number_of_nodes() == 0:
            lcc_vals.append(0.0)
            ccd_vals.append(0.0)
            curr_ccd = 0.0
            ccd_ratio = 0.0
        else:
            # 计算LCC
            largest_cc = max(nx.connected_components(GG), key=len)
            lcc = len(largest_cc) / N0
            lcc_vals.append(lcc)
            
            # 计算CCD Score（连通分量密度分数）
            # CCD = sum(n_i * (n_i - 1) / 2) for all connected components
            connected_components = list(nx.connected_components(GG))
            curr_ccd = 0.0
            for component in connected_components:
                n_i = len(component)
                curr_ccd += n_i * (n_i - 1) / 2.0
            
            # 归一化
            ccd_ratio = curr_ccd / ccd0 if ccd0 > 1e-9 else 0.0
            ccd_vals.append(max(ccd_ratio, 1e-8))
        
        # 计算cumulative reward（与训练一致）
        # 训练用的是: reward = -current_CCD × weight_ratio
        step_reward = -ccd_ratio * (node_weight / total_weight)
        cumulative_reward += step_reward
        
        # 横坐标：累积权重比例
        x_vals.append(accumulated_weight / total_weight)
        
        # 检查终止条件：覆盖边数 >= 80%
        if num_covered_edges >= edge_threshold:
            break
    
    return x_vals, lcc_vals, ccd_vals, cumulative_reward


# ==========================================
# 主流程
# ==========================================
def main():
    print("\n开始批量测试...")
    
    # 读取所有图文件（.gml格式）
    gml_files = [os.path.join(ROOT_GML_DIR, f)
                 for f in os.listdir(ROOT_GML_DIR) 
                 if f.endswith('.gml')]
    gml_files = sorted(gml_files, key=lambda x: int(os.path.basename(x).replace('graph_', '').replace('.gml', '')))
    
    if MAX_GRAPHS is not None:
        gml_files = gml_files[:MAX_GRAPHS]
    
    print(f"处理 {len(gml_files)} 个图文件")
    
    record_x = {m: [] for m in METHODS}
    record_lcc = {m: [] for m in METHODS}
    record_ccd = {m: [] for m in METHODS}
    record_reward = {m: [] for m in METHODS}  # 新增：记录cumulative reward
    
    for idx, gml_path in enumerate(gml_files, 1):
        gid = os.path.basename(gml_path).replace('.gml', '')
        print(f"[{idx}/{len(gml_files)}] {gid}", end='  ', flush=True)
        
        # 传递图索引 (从0开始)
        G, sequences = generate_attack_sequences(gml_path, graph_idx=idx-1)
        
        for m in METHODS:
            x, lcc, ccd, reward = simulate_attack_weighted(G, sequences[m])
            record_x[m].append(x)
            record_lcc[m].append(lcc)
            record_ccd[m].append(ccd)
            record_reward[m].append(reward)
    
    print("\n\n生成对比图...")
    
    # ==========================================
    # 计算平均曲线（插值对齐）并绘图
    # ==========================================
    from scipy.interpolate import interp1d

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 动态确定横轴范围：找出所有方法实际达到的最大权重比例
    max_w_reached = 0.0
    for m in METHODS:
        for x_vals in record_x[m]:
            if len(x_vals) > 0:
                max_w_reached = max(max_w_reached, x_vals[-1])
    
    # 横轴范围略大于实际最大值（增加5%余量）
    w_max = min(max_w_reached * 1.05, 1.0)  # 不超过1.0
    NUM_W = 50
    w_points = np.linspace(0, w_max, NUM_W)

    colors = {
        'RF': 'red',
        'HDA': 'blue',
        'HBA': 'green',
        'HCA': 'orange',
        'HPRA': 'purple'
    }

    print("\n各方法的平均LCC鲁棒性（前80%边覆盖阶段）：")
    print("-" * 50)
    
    # 用于存储CCD指标
    ccd_auc_results = {}

    for m in METHODS:
        lcc_interp_all = []
        ccd_interp_all = []

        for x_vals, lcc_vals, ccd_vals in zip(
            record_x[m], record_lcc[m], record_ccd[m]
        ):
            x = np.array(x_vals)
            lcc = np.array(lcc_vals)
            ccd = np.array(ccd_vals)

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
            lcc_w = f_lcc(w_points)
            lcc_interp_all.append(lcc_w)

            # CCD 插值
            f_ccd = interp1d(
                x, ccd,
                kind='linear',
                bounds_error=False,
                fill_value=(ccd[0], ccd[-1])
            )
            ccd_w = f_ccd(w_points)
            ccd_interp_all.append(ccd_w)

        # 转为数组
        LCC_all = np.vstack(lcc_interp_all)
        CCD_all = np.vstack(ccd_interp_all)

        # 平均 & 标准差
        LCC_mean = np.mean(LCC_all, axis=0)
        LCC_std = np.std(LCC_all, axis=0)

        CCD_mean = np.mean(CCD_all, axis=0)
        CCD_std = np.std(CCD_all, axis=0)

        # 平均鲁棒性（AUC近似）
        lcc_auc = np.mean(LCC_mean)
        ccd_auc = np.mean(CCD_mean)
        avg_robustness = lcc_auc
        ccd_auc_results[m] = ccd_auc
        print(f"  {m:6s}: {avg_robustness:.4f}")

        color = colors.get(m, 'gray')

        # -------- LCC 图 --------
        axes[0].plot(
            w_points, LCC_mean,
            label=f'{m} (R={avg_robustness:.3f})',
            color=color,
            linewidth=2
        )
        axes[0].fill_between(
            w_points,
            LCC_mean - LCC_std,
            LCC_mean + LCC_std,
            color=color,
            alpha=0.2
        )

        # -------- CCD 图 --------
        axes[1].plot(
            w_points, CCD_mean,
            label=m,
            color=color,
            linewidth=2
        )
        axes[1].fill_between(
            w_points,
            np.maximum(CCD_mean - CCD_std, 1e-8),
            CCD_mean + CCD_std,
            color=color,
            alpha=0.2
        )

    # ==========================
    # 图表格式
    # ==========================
    axes[0].set_xlabel('累积节点权重比例 (w)', fontsize=13)
    axes[0].set_ylabel('最大连通分量比例 (LCC/N)', fontsize=13)
    axes[0].set_title('网络鲁棒性对比 - LCC指标', fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes[0].grid(True, alpha=0.3)

    # CCD子图
    axes[1].set_xlabel('累积节点权重比例 (w)', fontsize=13)
    axes[1].set_ylabel('pairwise connectivity (PC/PC0)', fontsize=13)
    axes[1].set_title('网络鲁棒性对比 - pc指标', fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)

    # 保存
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    output_path = os.path.join(OUT_ROOT, f'robustness_weighted_axis_{timestamp}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # 也保存PNG
    output_png = output_path.replace('.pdf', '.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 保存: {output_path}")
    
    # 输出CP指标评估（使用cumulative reward）
    print("\nPC 指标评估（cumulative reward）：")
    
    # 计算每个方法的平均reward
    avg_rewards = {}
    for m in METHODS:
        avg_rewards[m] = np.mean(record_reward[m])
    
    # 按reward排序（cumulative reward是负数，越大越好）
    # 等价于 min Σ(PC_after × weight)，越小越好
    sorted_rewards = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
    for rank, (m, reward_val) in enumerate(sorted_rewards, 1):
        mark = " ← 最优" if rank == 1 else ""
        print(f"  {rank}. {m:6s}: {reward_val:.6f}{mark}")
    
    print(f"\n✓ 完成！处理了 {len(gml_files)} 个图\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n\n❌ 程序出错: {str(e)}")
        import traceback
        traceback.print_exc()
