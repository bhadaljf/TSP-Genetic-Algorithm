#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSP问题遗传算法可视化模块
实现五张图表：路径图、收敛曲线、柱状图、双轴性能图、箱线图
"""

import argparse
import json
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 论文级配色方案
COLORS = {
    'primary': '#1f77b4',      # 蓝色
    'secondary': '#ff7f0e',    # 橙色
    'success': '#2ca02c',      # 绿色
    'danger': '#d62728',       # 红色
    'warning': '#ff7f0e',      # 橙色
    'info': '#17a2b8',         # 青色
    'light': '#f8f9fa',        # 浅灰
    'dark': '#343a40',         # 深灰
    'path': '#e63946',         # 路径红色
    'city': '#1d3557',         # 城市深蓝
    'grid': '#dee2e6'          # 网格浅灰
}


def load_run_data(prefix: str):
    """加载单次运行数据"""
    cities = []
    with open(f"{prefix}_cities.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cities.append({'x': float(row['x']), 'y': float(row['y'])})
    
    best_tour = []
    with open(f"{prefix}_best_tour.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            best_tour.append(int(row['index']))
    
    convergence = []
    with open(f"{prefix}_convergence.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            convergence.append({
                'generation': int(row['generation']),
                'best_length': float(row['best_length'])
            })
    
    meta = {}
    meta_path = f"{prefix}_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    
    return cities, best_tour, convergence, meta


def plot_tsp_path(cities, best_tour, n, function_num, save=False, trial_num=None, folder=None):
    """1. 路径可视化图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置坐标轴范围
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    
    # 绘制路径
    tour_x = [cities[i]['x'] for i in best_tour] + [cities[best_tour[0]]['x']]
    tour_y = [cities[i]['y'] for i in best_tour] + [cities[best_tour[0]]['y']]
    ax.plot(tour_x, tour_y, '-', color=COLORS['path'], linewidth=1.5, alpha=0.8, label='最优路径')
    
    # 绘制城市
    city_x = [city['x'] for city in cities]
    city_y = [city['y'] for city in cities]
    ax.scatter(city_x, city_y, s=30, c=COLORS['city'], edgecolors='#333333', 
               linewidth=0.5, label='城市', zorder=5)
    
    # 设置网格
    ax.grid(True, color=COLORS['grid'], linestyle=':', alpha=0.7, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 设置标签和标题
    ax.set_xlabel('X坐标', fontsize=12)
    ax.set_ylabel('Y坐标', fontsize=12)
    
    if trial_num is not None:
        title = f"TSP_function_{function_num}({trial_num})_n{n}_map"
    else:
        title = f"TSP_function_{function_num}_n{n}_map"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加图例
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # 设置坐标轴刻度
    ax.set_xticks(range(-500, 501, 100))
    ax.set_yticks(range(-500, 501, 100))
    
    plt.tight_layout()
    
    if save:
        if folder:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, f"{title}.png")
        else:
            filepath = f"{title}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"已保存: {filepath}")
    else:
        # 在非交互式环境中，即使不保存也要保存到临时文件
        if folder:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, f"{title}.png")
        else:
            filepath = f"{title}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"已保存: {filepath}")
    plt.close()


def plot_convergence_curve(convergence, save=False, folder=None):
    """2. 收敛曲线图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = [c['generation'] for c in convergence]
    best_lengths = [c['best_length'] for c in convergence]
    
    ax.plot(generations, best_lengths, '-', color=COLORS['success'], 
            linewidth=2, marker='o', markersize=3, label='最优路径长度')
    
    # 设置网格
    ax.grid(True, color=COLORS['grid'], linestyle=':', alpha=0.7, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 设置标签和标题
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('当代最优路径长度', fontsize=12)
    ax.set_title('Convergence_Curve', fontsize=14, fontweight='bold')
    
    # 添加图例
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    if save:
        if folder:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, "Convergence_Curve.png")
        else:
            filepath = "Convergence_Curve.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"已保存: {filepath}")
    else:
        # 在非交互式环境中，即使不保存也要保存到临时文件
        if folder:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, "Convergence_Curve.png")
        else:
            filepath = "Convergence_Curve.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"已保存: {filepath}")
    plt.close()


def plot_optimal_distance_bar_chart(compare_data, save=False, folder=None):
    """3. 不同n值对应最优距离柱状图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 计算每个n的平均最优距离
    n_values = sorted(compare_data.keys())
    avg_distances = [compare_data[n]['avg_distance'] for n in n_values]
    
    bars = ax.bar(n_values, avg_distances, color=COLORS['primary'], 
                  alpha=0.8, edgecolor='white', linewidth=1)
    
    # 添加数值标签
    for bar, avg_dist in zip(bars, avg_distances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{avg_dist:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 设置网格
    ax.grid(True, color=COLORS['grid'], linestyle=':', alpha=0.7, linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # 设置标签和标题
    ax.set_xlabel('城市数量 N', fontsize=12)
    ax.set_ylabel('平均最优路径长度', fontsize=12)
    ax.set_title('Optimal_Distance_vs_N_Bar_Chart', fontsize=14, fontweight='bold')
    
    # 设置x轴刻度
    ax.set_xticks(n_values)
    
    plt.tight_layout()
    
    if save:
        if folder:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, "Optimal_Distance_vs_N_Bar_Chart.png")
        else:
            filepath = "Optimal_Distance_vs_N_Bar_Chart.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"已保存: {filepath}")
    else:
        # 在非交互式环境中，即使不保存也要保存到临时文件
        if folder:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, "Optimal_Distance_vs_N_Bar_Chart.png")
        else:
            filepath = "Optimal_Distance_vs_N_Bar_Chart.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"已保存: {filepath}")
    plt.close()


def plot_performance_analysis_dual_axis(compare_data, save=False, folder=None):
    """4. 性能分析双轴图"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    n_values = sorted(compare_data.keys())
    avg_distances = [compare_data[n]['avg_distance'] for n in n_values]
    avg_times = [compare_data[n]['avg_time'] for n in n_values]
    
    # 左轴：最优路径长度
    color1 = COLORS['success']
    ax1.set_xlabel('城市数量 N', fontsize=12)
    ax1.set_ylabel('平均最优路径长度', color=color1, fontsize=12)
    line1 = ax1.plot(n_values, avg_distances, '-o', color=color1, 
                     linewidth=2, markersize=6, label='平均最优路径长度')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, color=COLORS['grid'], linestyle=':', alpha=0.7, linewidth=0.5)
    
    # 右轴：运行时间
    ax2 = ax1.twinx()
    color2 = COLORS['danger']
    ax2.set_ylabel('平均运行时间 (ms)', color=color2, fontsize=12)
    line2 = ax2.plot(n_values, avg_times, '-s', color=color2, 
                     linewidth=2, markersize=6, label='平均运行时间')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    ax1.set_title('Performance_Analysis_Dual_Axis', fontsize=14, fontweight='bold')
    ax1.set_xticks(n_values)
    
    plt.tight_layout()
    
    if save:
        if folder:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, "Performance_Analysis_Dual_Axis.png")
        else:
            filepath = "Performance_Analysis_Dual_Axis.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"已保存: {filepath}")
    else:
        # 在非交互式环境中，即使不保存也要保存到临时文件
        if folder:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, "Performance_Analysis_Dual_Axis.png")
        else:
            filepath = "Performance_Analysis_Dual_Axis.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"已保存: {filepath}")
    plt.close()


def plot_box_plot_distribution(compare_data, save=False, folder=None):
    """5. 箱线图分布"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_values = sorted(compare_data.keys())
    data_for_box = [compare_data[n]['distances'] for n in n_values]
    
    # 创建箱线图
    box_plot = ax.boxplot(data_for_box, tick_labels=[str(n) for n in n_values], 
                         patch_artist=True, showfliers=True)
    
    # 设置箱体颜色
    for patch in box_plot['boxes']:
        patch.set_facecolor(COLORS['primary'])
        patch.set_alpha(0.7)
    
    # 设置网格
    ax.grid(True, color=COLORS['grid'], linestyle=':', alpha=0.7, linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # 设置标签和标题
    ax.set_xlabel('城市数量 N', fontsize=12)
    ax.set_ylabel('最优路径长度', fontsize=12)
    ax.set_title('Box_Plot_Optimal_Distance_Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        if folder:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, "Box_Plot_Optimal_Distance_Distribution.png")
        else:
            filepath = "Box_Plot_Optimal_Distance_Distribution.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"已保存: {filepath}")
    else:
        # 在非交互式环境中，即使不保存也要保存到临时文件
        if folder:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, "Box_Plot_Optimal_Distance_Distribution.png")
        else:
            filepath = "Box_Plot_Optimal_Distance_Distribution.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"已保存: {filepath}")
    plt.close()


def load_compare_data(csv_file):
    """加载比较数据"""
    compare_data = {}
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['n'])
            if n not in compare_data:
                compare_data[n] = {
                    'distances': [],
                    'times': [],
                    'generations': []
                }
            
            compare_data[n]['distances'].append(float(row['best_length']))
            compare_data[n]['times'].append(float(row['elapsed_ms']))
            compare_data[n]['generations'].append(int(row['generations']))
    
    # 计算平均值
    for n in compare_data:
        data = compare_data[n]
        data['avg_distance'] = np.mean(data['distances'])
        data['avg_time'] = np.mean(data['times'])
        data['avg_generations'] = np.mean(data['generations'])
    
    return compare_data


def main():
    parser = argparse.ArgumentParser(description='TSP遗传算法可视化工具')
    parser.add_argument('--path', action='store_true', help='绘制路径图')
    parser.add_argument('--conv', action='store_true', help='绘制收敛曲线')
    parser.add_argument('--prefix', type=str, default='run_manual', help='数据文件前缀')
    parser.add_argument('--compare', action='store_true', help='绘制比较图表')
    parser.add_argument('--input', type=str, default='compare_results.csv', help='比较数据CSV文件')
    parser.add_argument('--save', action='store_true', help='保存为PNG格式')
    parser.add_argument('--function', type=int, default=1, help='功能编号')
    parser.add_argument('--n', type=int, help='城市数量')
    parser.add_argument('--trial', type=int, help='实验次数（用于功能3）')
    parser.add_argument('--folder', type=str, help='保存文件夹路径')
    
    args = parser.parse_args()
    
    if args.compare:
        # 功能3：绘制所有比较图表
        compare_data = load_compare_data(args.input)
        
        # 绘制五张图表
        plot_optimal_distance_bar_chart(compare_data, args.save, args.folder)
        plot_performance_analysis_dual_axis(compare_data, args.save, args.folder)
        plot_box_plot_distribution(compare_data, args.save, args.folder)
        
        # 为每个n值绘制路径图
        for n in sorted(compare_data.keys()):
            # 这里需要从实际运行数据中获取路径信息
            # 由于比较模式下没有具体的路径数据，这里跳过路径图
            pass
        
        print("比较图表绘制完成！")
        
    else:
        # 功能1、2：绘制路径图和收敛曲线
        if args.path or args.conv:
            cities, best_tour, convergence, meta = load_run_data(args.prefix)
            
            if args.path:
                n = args.n if args.n else len(cities)
                trial = args.trial if args.trial else None
                plot_tsp_path(cities, best_tour, n, args.function, args.save, trial, args.folder)
            
            if args.conv:
                plot_convergence_curve(convergence, args.save, args.folder)
            
            print("路径图和收敛曲线绘制完成！")


if __name__ == "__main__":
    main()