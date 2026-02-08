#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试FINDER_CND_cost多目标优化功能
- 验证模块加载
- 测试训练流程（极短时间，只验证能运行）
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
from FINDER import FINDER

def test_module_loading():
    """测试模块加载"""
    print("="*60)
    print("测试1: 验证模块加载")
    print("="*60)
    
    try:
        import mvc_env
        import graph
        import PrepareBatchGraph
        print("✓ mvc_env 加载成功")
        print("✓ graph 加载成功")
        print("✓ PrepareBatchGraph 加载成功")
        print("\n✓ 所有核心模块加载正常\n")
        return True
    except Exception as e:
        print(f"❌ 模块加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_initialization():
    """测试FINDER初始化"""
    print("="*60)
    print("测试2: FINDER初始化和图生成")
    print("="*60)
    
    try:
        print("创建FINDER实例...")
        dqn = FINDER()
        
        print("生成训练图...")
        dqn.gen_new_graphs(30, 35)  # 生成30-35节点的小图
        
        print("✓ 训练图生成完成")
        
        print("\n准备验证数据...")
        dqn.PrepareValidData()
        print("✓ 验证数据准备完成")
        
        print("\n✓ FINDER初始化正常\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_minimal_training():
    """最小训练测试（只运行几个step）"""
    print("="*60)
    print("测试3: 最小训练流程")
    print("="*60)
    
    try:
        print("创建FINDER实例...")
        dqn = FINDER()
        
        print("准备数据...")
        dqn.PrepareValidData()
        dqn.gen_new_graphs(30, 35)
        
        print("初始化游戏环境（运行5局）...")
        for i in range(5):
            dqn.PlayGame(5, 1.0)
            if i == 0:
                print("  第1局完成")
        print("  所有初始化游戏完成")
        
        print("\n创建快照...")
        dqn.TakeSnapShot()
        
        print("\n运行训练step（3次）...")
        for i in range(3):
            dqn.PlayGame(5, 0.5)
            dqn.Fit()
            print(f"  训练step {i+1}/3 完成")
        
        print("\n✓ 训练流程正常工作\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n")
    print("="*60)
    print("FINDER_CND_cost 多目标优化测试")
    print("="*60)
    print("说明: 测试多目标reward是否能正常训练")
    print("      alpha=0.5 (CN和ND各占50%权重)")
    print("="*60)
    print("\n")
    
    # 测试1: 模块加载
    success1 = test_module_loading()
    
    # 测试2: 初始化
    success2 = False
    if success1:
        success2 = test_initialization()
    
    # 测试3: 最小训练
    success3 = False
    if success2:
        success3 = test_minimal_training()
    
    # 总结
    print("="*60)
    print("测试总结")
    print("="*60)
    print(f"模块加载: {'✓ 通过' if success1 else '❌ 失败'}")
    print(f"初始化:   {'✓ 通过' if success2 else '❌ 失败'}")
    print(f"训练流程: {'✓ 通过' if success3 else '❌ 失败'}")
    print("\n")
    
    if success1 and success2 and success3:
        print("✓ 所有测试通过！FINDER_CND_cost可以开始完整训练")
        print("\n运行完整训练:")
        print("  python train.py")
        print("\n调整多目标权重:")
        print("  编辑 src/lib/mvc_env.cpp 第307行")
        print("  double alpha = 0.5;  // 0.0=纯ND, 0.5=均衡, 1.0=纯CN")
        print("\n训练说明:")
        print("  - 默认alpha=0.5, 平衡CN和ND两个目标")
        print("  - 修改alpha后需要重新编译: python setup.py build_ext --inplace")
        print("  - 训练完成后模型保存在 ./models/ 目录")
    else:
        print("❌ 存在问题，请检查错误信息")
    
    print("\n")


if __name__ == "__main__":
    main()

