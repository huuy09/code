#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版测试脚本 - 记录最大联通组件大小等详细信息
"""
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER
from tqdm import tqdm
import networkx as nx
import numpy as np


def get_max_component_size(g, removed_nodes):
    """计算移除节点后的最大联通组件大小"""
    g_copy = g.copy()
    g_copy.remove_nodes_from(removed_nodes)
    if len(g_copy.nodes()) == 0:
        return 0
    
    # 获取所有联通组件
    components = list(nx.connected_components(g_copy))
    if len(components) == 0:
        return 0
    
    # 返回最大组件的大小
    max_component_size = max(len(c) for c in components)
    return max_component_size


def evaluate_with_components(dqn, data_test, model_file):
    """带有联通组件分析的评估"""
    dqn.LoadModel(model_file)
    
    n_test = 100
    result_scores = []
    result_times = []
    result_removed_nodes = []
    result_max_components = []
    result_num_components = []
    
    print(f"开始测试 {data_test}...")
    
    for i in tqdm(range(n_test)):
        g_path = f'{data_test}/g_{i}'
        g = nx.read_gml(g_path)
        original_nodes = g.number_of_nodes()
        
        # 插入图并获取解
        dqn.InsertGraph(g, is_test=True)
        val, sol = dqn.GetSol(i)
        
        # 计算联通组件信息
        max_comp_size = get_max_component_size(g, sol)
        
        # 计算联通组件数量
        g_remaining = g.copy()
        g_remaining.remove_nodes_from(sol)
        num_components = nx.number_connected_components(g_remaining)
        
        result_scores.append(val)
        result_removed_nodes.append(len(sol))
        result_max_components.append(max_comp_size)
        result_num_components.append(num_components)
    
    dqn.ClearTestGraphs()
    
    return {
        'score_mean': np.mean(result_scores),
        'score_std': np.std(result_scores),
        'removed_mean': np.mean(result_removed_nodes),
        'removed_std': np.std(result_removed_nodes),
        'max_comp_mean': np.mean(result_max_components),
        'max_comp_std': np.std(result_max_components),
        'num_comp_mean': np.mean(result_num_components),
        'num_comp_std': np.std(result_num_components),
    }


def main():
    dqn = FINDER()
    cost_types = ['degree_cost', 'random_cost']
    
    for cost in cost_types:
        data_test_path = './synthetic/%s/' % cost
        data_test_name = ['30-50', '50-100']
        model_file = './models/nrange_30_50_iter_134100.ckpt'
        
        file_path = '../results/FINDER_ND_cost/synthetic'
        
        if not os.path.exists('../results/FINDER_ND_cost'):
            os.makedirs('../results/FINDER_ND_cost')
        if not os.path.exists('../results/FINDER_ND_cost/synthetic'):
            os.makedirs('../results/FINDER_ND_cost/synthetic')
        
        results_file = f'{file_path}/{cost}_detailed_results.txt'
        
        with open(results_file, 'w') as fout:
            fout.write(f"详细测试结果 - {cost}\n")
            fout.write("=" * 80 + "\n\n")
            
            for i in tqdm(range(len(data_test_name)), desc=f"Testing {cost}"):
                data_test = data_test_path + data_test_name[i]
                
                print(f"\n正在测试: {data_test_name[i]}")
                results = evaluate_with_components(dqn, data_test, model_file)
                
                fout.write(f"数据集: {data_test_name[i]}\n")
                fout.write("-" * 80 + "\n")
                fout.write(f"近似率: {results['score_mean']*100:.2f}% ± {results['score_std']*100:.2f}%\n")
                fout.write(f"移除节点数: {results['removed_mean']:.2f} ± {results['removed_std']:.2f}\n")
                fout.write(f"最大联通组件大小: {results['max_comp_mean']:.2f} ± {results['max_comp_std']:.2f}\n")
                fout.write(f"联通组件数量: {results['num_comp_mean']:.2f} ± {results['num_comp_std']:.2f}\n")
                fout.write("\n")
                
                print(f"✓ {data_test_name[i]} 测试完成!")
                print(f"  近似率: {results['score_mean']*100:.2f}%")
                print(f"  最大联通组件: {results['max_comp_mean']:.2f} 节点")
        
        print(f"\n{cost} 详细结果已保存至: {results_file}\n")


if __name__=="__main__":
    main()
