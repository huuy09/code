"""
网络鲁棒性测试脚本 - 基于testSynthetic.py的简化版
====================================================

功能说明：
1. 测试FINDER算法在合成图数据集上的性能
2. 计算最大连通分量大小
3. 生成可视化对比图

作者：优化版本
日期：2026-02-02
"""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from FINDER import FINDER
from tqdm import tqdm

# ==========================================
# 环境配置
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 全局参数设置
# ==========================================
DATA_PATH = './synthetic'  # 合成数据集路径
MODEL_FILE = './models/nrange_30_50_iter_134100.ckpt'  # 模型文件
OUT_ROOT = '../results/FINDER_ND_cost/robustness'  # 输出目录

os.makedirs(OUT_ROOT, exist_ok=True)

# ==========================================
# 初始化FINDER模型
# ==========================================
print("="*60)
print("初始化 FINDER 模型...")
print("="*60)
dqn = FINDER()
print(f"✓ 将使用模型: {MODEL_FILE}\n")


# ==========================================
# 主函数
# ==========================================
def main():
    """运行FINDER测试并生成可视化报告"""
    print("="*60)
    print("开始测试 FINDER 算法")
    print("="*60)
    
    cost_types = ['degree_cost', 'random_cost']
    data_test_names = ['30-50', '50-100']
    
    all_results = {}
    
    for cost in cost_types:
        print(f"\n测试成本类型: {cost}")
        print("-" * 40)
        
        data_test_path = os.path.join(DATA_PATH, cost)
        
        if not os.path.exists(data_test_path):
            print(f"⚠️ 路径不存在: {data_test_path}")
            continue
        
        results_file = os.path.join(OUT_ROOT, f'{cost}_results.txt')
        
        with open(results_file, 'w', encoding='utf-8') as fout:
            fout.write(f"FINDER 测试结果 - {cost}\n")
            fout.write("="*60 + "\n\n")
            
            for size_range in data_test_names:
                data_test = os.path.join(data_test_path, size_range)
                
                if not os.path.exists(data_test):
                    print(f"  ⚠️ 数据集不存在: {size_range}")
                    continue
                
                print(f"  测试数据集: {size_range}")
                
                try:
                    score_mean, score_std, time_mean, time_std = dqn.Evaluate(
                        data_test, MODEL_FILE
                    )
                    
                    result_str = f'{size_range}: {score_mean*100:.2f}±{score_std*100:.2f}%'
                    print(f"    结果: {result_str}")
                    
                    fout.write(f"{result_str}\n")
                    fout.write(f"  平均时间: {time_mean:.4f}±{time_std:.4f} 秒\n\n")
                    
                    # 保存结果用于后续可视化
                    key = f"{cost}_{size_range}"
                    all_results[key] = {
                        'score_mean': score_mean,
                        'score_std': score_std,
                        'time_mean': time_mean,
                        'time_std': time_std
                    }
                    
                except Exception as e:
                    print(f"    ❌ 测试失败: {str(e)}")
                    fout.write(f"{size_range}: 测试失败 - {str(e)}\n\n")
        
        print(f"  ✓ 结果已保存至: {results_file}")
    
    # ==========================================
    # 生成可视化图表
    # ==========================================
    if all_results:
        print("\n" + "="*60)
        print("生成可视化图表")
        print("="*60)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 准备数据
        cost_labels = []
        scores = []
        errors = []
        times = []
        time_errors = []
        
        for key in sorted(all_results.keys()):
            cost_labels.append(key.replace('_', '\n'))
            scores.append(all_results[key]['score_mean'] * 100)
            errors.append(all_results[key]['score_std'] * 100)
            times.append(all_results[key]['time_mean'])
            time_errors.append(all_results[key]['time_std'])
        
        x_pos = np.arange(len(cost_labels))
        
        # 图1: 近似率
        ax1.bar(x_pos, scores, yerr=errors, capsize=5, alpha=0.7,
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_xlabel('数据集', fontsize=12)
        ax1.set_ylabel('近似率 (%)', fontsize=12)
        ax1.set_title('FINDER 近似率', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(cost_labels, fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 图2: 运行时间
        ax2.bar(x_pos, times, yerr=time_errors, capsize=5, alpha=0.7,
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_xlabel('数据集', fontsize=12)
        ax2.set_ylabel('运行时间 (秒)', fontsize=12)
        ax2.set_title('FINDER 运行时间', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(cost_labels, fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUT_ROOT, f'finder_results_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 图表已保存至: {output_path}")
    
    print("\n" + "="*60)
    print("✓ 测试完成！")
    print("="*60)
    print(f"输出目录: {OUT_ROOT}")
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
