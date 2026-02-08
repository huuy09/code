"""
将synthetic数据集转换为GML格式（保留原始节点权重）
"""
import networkx as nx
import os
import shutil

# 创建输出目录（使用新名称，保留原始权重）
os.makedirs('saved_graphs/degree_cost_weighted', exist_ok=True)

print("="*60)
print("复制 degree_cost 数据集（保留节点权重）")
print("="*60)

# 只使用degree_cost，因为它有有意义的权重
data_path = './synthetic/degree_cost/30-50'

converted_count = 0
file_index = 0

if not os.path.exists(data_path):
    print(f"\n⚠️ 路径不存在: {data_path}")
else:
    print(f"\n处理目录: {data_path}")
    
    files = [f for f in os.listdir(data_path) if f.startswith('g_')]
    print(f"  找到 {len(files)} 个图文件")
    
    for i, file in enumerate(files[:20]):  # 取20个
        input_path = os.path.join(data_path, file)
        
        try:
            # 读取GML文件（保留原始权重！）
            g = nx.read_gml(input_path, destringizer=int)
            
            # ⚠️ 不要修改权重！保持原始的degree_cost权重
            # 检查是否有权重
            has_weight = all('weight' in g.nodes[node] for node in g.nodes())
            if not has_weight:
                print(f"  ⚠️ {file} 没有权重属性，跳过")
                continue
            
            # 保存为GML
            output_file = f'saved_graphs/degree_cost_weighted/graph_{file_index:03d}.gml'
            nx.write_gml(g, output_file)
            
            file_index += 1
            converted_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"    已转换: {i+1}/{len(files[:20])}")
        
        except Exception as e:
            print(f"  ❌ 转换失败 {file}: {str(e)}")
            continue

print("\n" + "="*60)
print("✓ 转换完成！")
print("="*60)
print(f"  总共转换: {converted_count} 个加权图")
print(f"  输出目录: saved_graphs/degree_cost_weighted/")
print("="*60)

# 验证转换结果
print("\n验证前5个GML文件的权重...")
for i in range(min(5, converted_count)):
    gml_file = f'saved_graphs/degree_cost_weighted/graph_{i:03d}.gml'
    try:
        g = nx.read_gml(gml_file, destringizer=int)
        weights = [g.nodes[node]['weight'] for node in list(g.nodes())[:5]]
        print(f"  graph_{i:03d}.gml: {g.number_of_nodes()} 节点, 前5个权重={[f'{w:.3f}' for w in weights]}")
    except Exception as e:
        print(f"  ❌ 读取失败 graph_{i:03d}.gml: {str(e)}")

print("\n✓ 现在可以运行加权图测试: python robustness_test_weighted.py\n")
