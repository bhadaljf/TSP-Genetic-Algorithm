#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSP问题遗传算法实验平台
功能：
1) 随机生成N个二维坐标作为城市
2) 采用GA搜索TSP近似最优路径（精英保留+锦标赛选择+OX交叉+交换变异）
3) 输出最优路径长度，并生成SVG文件可视化
4) 支持三种实验模式：手动指定N、随机生成N、多N对比分析
"""

import random
import math
import time
import json
import csv
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class Point:
    x: float
    y: float


@dataclass
class GAParams:
    num_cities: int = 50
    population_size: int = 300
    generations: int = 1000
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    elites: int = 2
    seed: int = 0  # 0 表示自动按时间种子


@dataclass
class GAResult:
    cities: List[Point]
    best_tour: List[int]
    convergence_best_length: List[float]
    best_length: float
    current_generation: int
    elapsed_ms: float


def compute_euclidean_distance(a: Point, b: Point) -> float:
    """计算两点间欧几里得距离"""
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def compute_path_length(path: List[int], cities: List[Point]) -> float:
    """计算路径总长度"""
    total = 0.0
    n = len(path)
    for i in range(n):
        u = cities[path[i]]
        v = cities[path[(i + 1) % n]]  # 回到起点
        total += compute_euclidean_distance(u, v)
    return total


def generate_random_cities(n: int, rng: random.Random) -> List[Point]:
    """在 [-500, 500] 范围生成N个随机城市"""
    cities = []
    for i in range(n):
        x = rng.uniform(-500.0, 500.0)
        y = rng.uniform(-500.0, 500.0)
        cities.append(Point(x, y))
    return cities


def make_initial_tour(n: int) -> List[int]:
    """生成初始路径 [0, 1, 2, ..., n-1]"""
    return list(range(n))


def initialize_population(pop_size: int, n: int, rng: random.Random) -> List[List[int]]:
    """初始化种群"""
    population = []
    for i in range(pop_size):
        tour = make_initial_tour(n)
        rng.shuffle(tour)
        population.append(tour)
    return population


def evaluate_population(population: List[List[int]], cities: List[Point]) -> List[float]:
    """评估种群适应度"""
    fitness = []
    for tour in population:
        length = compute_path_length(tour, cities)
        fitness.append(1.0 / (length + 1e-9))
    return fitness


def tournament_select(fitness: List[float], rng: random.Random, tour_size: int = 3) -> int:
    """锦标赛选择"""
    pop_size = len(fitness)
    best_idx = rng.randint(0, pop_size - 1)
    best_fit = fitness[best_idx]
    
    for i in range(1, tour_size):
        idx = rng.randint(0, pop_size - 1)
        if fitness[idx] > best_fit:
            best_fit = fitness[idx]
            best_idx = idx
    return best_idx


def order_crossover_ox(p1: List[int], p2: List[int], rng: random.Random) -> List[int]:
    """顺序交叉（OX）"""
    n = len(p1)
    a = rng.randint(0, n - 1)
    b = rng.randint(0, n - 1)
    if a > b:
        a, b = b, a
    
    child = [-1] * n
    for i in range(a, b + 1):
        child[i] = p1[i]
    
    used = [False] * n
    for i in range(a, b + 1):
        used[p1[i]] = True
    
    pos = (b + 1) % n
    for i in range(n):
        gene = p2[(b + 1 + i) % n]
        if not used[gene]:
            child[pos] = gene
            pos = (pos + 1) % n
    
    return child


def mutate_swap(tour: List[int], rng: random.Random, mutation_rate: float) -> None:
    """交换变异"""
    if rng.random() < mutation_rate:
        n = len(tour)
        if n >= 2:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i != j:
                tour[i], tour[j] = tour[j], tour[i]


def run_ga(params: GAParams, brief_output: bool = False, detail_file: str = None) -> GAResult:
    """运行遗传算法"""
    # 初始化随机数生成器
    if params.seed == 0:
        rng = random.Random()
    else:
        rng = random.Random(params.seed)
    
    start_time = time.time()
    
    # 生成城市和初始种群
    cities = generate_random_cities(params.num_cities, rng)
    population = initialize_population(params.population_size, params.num_cities, rng)
    fitness = evaluate_population(population, cities)
    
    # 找到初始最优解
    best_idx = 0
    best_fit = fitness[0]
    for i in range(1, params.population_size):
        if fitness[i] > best_fit:
            best_fit = fitness[i]
            best_idx = i
    
    best_tour = population[best_idx].copy()
    best_length = compute_path_length(best_tour, cities)
    
    convergence = []
    detail_fp = None
    if detail_file:
        detail_fp = open(detail_file, 'w', newline='', encoding='utf-8')
        detail_fp.write("generation,best_length\n")
    
    # 主循环
    for gen in range(params.generations):
        # 精英保留
        idx = list(range(len(population)))
        idx.sort(key=lambda i: fitness[i], reverse=True)
        
        new_population = []
        for e in range(min(params.elites, len(population))):
            new_population.append(population[idx[e]].copy())
        
        # 生成新个体
        while len(new_population) < params.population_size:
            pa = tournament_select(fitness, rng)
            pb = tournament_select(fitness, rng)
            p1 = population[pa]
            p2 = population[pb]
            
            if rng.random() < params.crossover_rate:
                child = order_crossover_ox(p1, p2, rng)
            else:
                child = p1.copy()
            
            mutate_swap(child, rng, params.mutation_rate)
            new_population.append(child)
        
        # 更新种群
        population = new_population
        fitness = evaluate_population(population, cities)
        
        # 更新最优解
        cur_best_idx = 0
        cur_best_fit = fitness[0]
        for i in range(1, params.population_size):
            if fitness[i] > cur_best_fit:
                cur_best_fit = fitness[i]
                cur_best_idx = i
        
        cur_length = 1.0 / cur_best_fit - 1e-9
        if cur_length < best_length:
            best_length = cur_length
            best_tour = population[cur_best_idx].copy()
        
        convergence.append(best_length)
        
        # 输出进度
        should_print = not brief_output and ((gen + 1) % max(1, params.generations // 10) == 0)
        if should_print:
            print(f"[Gen {gen + 1}/{params.generations}] best length = {best_length:.6f}")
        
        if detail_fp:
            detail_fp.write(f"{gen + 1},{best_length:.6f}\n")
    
    if detail_fp:
        detail_fp.close()
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    
    return GAResult(
        cities=cities,
        best_tour=best_tour,
        convergence_best_length=convergence,
        best_length=best_length,
        current_generation=params.generations,
        elapsed_ms=elapsed_ms
    )




def write_run_files(prefix: str, result: GAResult, folder: str = None) -> None:
    """写入运行数据到文件（仅用于display.py读取）"""
    # 如果指定了文件夹，将文件保存到文件夹内
    if folder:
        os.makedirs(folder, exist_ok=True)
        base_path = os.path.join(folder, prefix)
    else:
        base_path = prefix
    
    # cities.csv - 仅用于display.py读取
    with open(f"{base_path}_cities.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for city in result.cities:
            writer.writerow([city.x, city.y])
    
    # best_tour.csv - 仅用于display.py读取
    with open(f"{base_path}_best_tour.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index'])
        for idx in result.best_tour:
            writer.writerow([idx])
    
    # convergence.csv - 仅用于display.py读取
    with open(f"{base_path}_convergence.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'best_length'])
        for i, length in enumerate(result.convergence_best_length):
            writer.writerow([i + 1, length])
    
    # meta.json - 仅用于display.py读取
    with open(f"{base_path}_meta.json", 'w', encoding='utf-8') as f:
        json.dump({
            'best_length': result.best_length,
            'elapsed_ms': result.elapsed_ms,
            'generations': result.current_generation
        }, f, indent=2)


def create_function_folder(function_num: int) -> str:
    """创建功能文件夹"""
    now = datetime.now()
    folder_name = f"{now.year}-{now.month:02d}-{now.day:02d}-{now.hour:02d}-{now.minute:02d}-function{function_num}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name




def feature_manual_n() -> None:
    """功能1：手动指定N"""
    print("请输入城市数量N：", end='')
    try:
        n = int(input())
        if n < 3:
            print("城市数量至少为3")
            return
    except ValueError:
        print("请输入有效的数字")
        return
    
    params = GAParams(num_cities=n)
    
    print("是否只显示关键步骤？（详细过程将保存至文件）（y/n）", end='')
    brief_choice = input().strip().lower()
    brief_output = (brief_choice == 'y')
    
    # 创建功能文件夹
    folder = create_function_folder(1)
    print(f"创建文件夹: {folder}")
    
    detail_file = os.path.join(folder, "run_manual_detail.csv") if brief_output else None
    result = run_ga(params, brief_output, detail_file)
    
    print(f"运行时间(ms)：{result.elapsed_ms:.2f}")
    print(f"运行步数(代数)：{result.current_generation}")
    
    # 临时保存数据文件用于生成图片
    write_run_files("run_manual", result, folder)
    
    # 直接生成路径与收敛图PNG
    cmd = f'py display.py --path --conv --prefix "{os.path.join(folder, "run_manual")}" --function 1 --n {n} --folder "{folder}" --save'
    print(f"生成可视化图表...")
    result_cmd = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    print(f"命令输出: {result_cmd.stdout}")
    if result_cmd.stderr:
        print(f"错误输出: {result_cmd.stderr}")
    
    # 删除临时数据文件，只保留PNG图片
    temp_files = [
        os.path.join(folder, "run_manual_cities.csv"),
        os.path.join(folder, "run_manual_best_tour.csv"),
        os.path.join(folder, "run_manual_convergence.csv"),
        os.path.join(folder, "run_manual_meta.json")
    ]
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)


def feature_random_n() -> None:
    """功能2：随机生成N"""
    # 随机生成N (5-60)
    rng = random.Random()
    n = rng.randint(5, 60)
    print(f"随机生成的城市数量N：{n}")
    
    params = GAParams(num_cities=n)
    
    print("是否只显示关键步骤？（详细过程将保存至文件）（y/n）", end='')
    brief_choice = input().strip().lower()
    brief_output = (brief_choice == 'y')
    
    # 创建功能文件夹
    folder = create_function_folder(2)
    print(f"创建文件夹: {folder}")
    
    detail_file = os.path.join(folder, "run_random_detail.csv") if brief_output else None
    result = run_ga(params, brief_output, detail_file)
    
    print(f"运行时间(ms)：{result.elapsed_ms:.2f}")
    print(f"运行步数(代数)：{result.current_generation}")
    
    # 临时保存数据文件用于生成图片
    write_run_files("run_random", result, folder)
    
    # 直接生成路径与收敛图PNG
    cmd = f'py display.py --path --conv --prefix "{os.path.join(folder, "run_random")}" --function 2 --n {n} --folder "{folder}" --save'
    print(f"生成可视化图表...")
    result_cmd = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    print(f"命令输出: {result_cmd.stdout}")
    if result_cmd.stderr:
        print(f"错误输出: {result_cmd.stderr}")
    
    # 删除临时数据文件，只保留PNG图片
    temp_files = [
        os.path.join(folder, "run_random_cities.csv"),
        os.path.join(folder, "run_random_best_tour.csv"),
        os.path.join(folder, "run_random_convergence.csv"),
        os.path.join(folder, "run_random_meta.json")
    ]
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)


def feature_compare() -> None:
    """功能3：比较不同N的性能"""
    group1 = [5, 8, 10]
    group2 = [15, 20, 25, 30]
    group3 = [40, 50, 60]
    
    print("预设N分组如下：")
    print("(1) 小规模：5, 8, 10")
    print("(2) 中规模：15, 20, 25, 30")
    print("(3) 大规模：40, 50, 60")
    print("请选择组别 [1-3]：", end='')
    
    try:
        group = int(input())
        if group == 1:
            n_list = group1
        elif group == 2:
            n_list = group2
        elif group == 3:
            n_list = group3
        else:
            print("无效选择")
            return
    except ValueError:
        print("请输入有效数字")
        return
    
    print("是否只显示关键步骤？（详细过程将保存至文件）（y/n）", end='')
    brief_choice = input().strip().lower()
    brief_output = (brief_choice == 'y')
    
    # 每个N的实验次数
    print("请输入每个N的实验次数(建议>=5，默认10)：", end='')
    trials_input = input().strip()
    try:
        trials = max(1, int(trials_input)) if trials_input else 10
    except ValueError:
        trials = 10
    
    # 创建功能3文件夹
    folder = create_function_folder(3)
    print(f"创建文件夹: {folder}")
    
    # 直接在文件夹内创建比较结果文件
    compare_file = os.path.join(folder, "compare_results.csv")
    with open(compare_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'trial', 'best_length', 'elapsed_ms', 'generations'])
        
        for n in n_list:
            for trial in range(1, trials + 1):
                params = GAParams(num_cities=n, generations=1000)
                detail_file = os.path.join(folder, "compare_detail.csv") if brief_output else None
                result = run_ga(params, brief_output, detail_file)
                
                writer.writerow([n, trial, result.best_length, result.elapsed_ms, result.current_generation])
                print(f"N={n} 第{trial}次：best={result.best_length:.6f}, time(ms)={result.elapsed_ms:.2f}")
    
    print("实验完成。已输出数据表格到文件夹内。")
    
    # 直接生成对比类图表PNG
    cmd = f'py display.py --compare --input "{compare_file}" --folder "{folder}" --save'
    print(f"生成对比图表...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    print(f"命令输出: {result.stdout}")
    if result.stderr:
        print(f"错误输出: {result.stderr}")
    
    # 删除比较结果CSV文件，只保留PNG图片
    if os.path.exists(compare_file):
        os.remove(compare_file)


def menu_loop() -> None:
    """主菜单循环"""
    while True:
        print("\n=== TSP问题遗传算法实验平台 ===")
        print("1. 手动指定城市数量N")
        print("2. 随机生成城市数量N")
        print("3. 比较不同N对算法性能的影响")
        print("4. 退出")
        print("请选择 [1-4]：", end='')
        
        try:
            choice = int(input())
            if choice == 1:
                feature_manual_n()
            elif choice == 2:
                feature_random_n()
            elif choice == 3:
                feature_compare()
            elif choice == 4:
                print("再见！")
                break
            else:
                print("无效选项，请重新选择")
        except ValueError:
            print("请输入有效数字")
        except KeyboardInterrupt:
            print("\n程序被中断")
            break
        except EOFError:
            print("\n输入结束，程序退出")
            break


def main():
    """主函数"""
    # 设置控制台编码
    import sys
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        # 如果无法设置编码，继续运行
        pass
    
    print("TSP问题遗传算法实验平台 - Python版本")
    menu_loop()


if __name__ == "__main__":
    main()

