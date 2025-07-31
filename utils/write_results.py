# -*- coding: gbk -*-

import os
import re
import pandas as pd
from collections import defaultdict


def parse_log_file(log_path):
    """解析日志文件，兼容新旧所有指标"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(log_path, 'r', encoding='gbk') as f:
            content = f.read()

    # 初始化结果字典（包含所有字段）
    results = {
        'First_OA': 'N/A',
        # 原有精度指标
        'OA': 'N/A', 'AA': 'N/A', 'Kappa': 'N/A',
        # 新增参数
        'trainable_params': 'N/A',
        'total_mult_adds_M': 'N/A',
        'model_size_MB': 'N/A',
        'training_time_s': 'N/A',
        'inference_time_s': 'N/A',
        # 类别精度
        'class_acc': {},
        'Total_training_time_s': 'N/A',
        'Total_inference_time_s': 'N/A',
    }
    # =================================================================
    # 五次实验的总共时间==============================
    # =================================================================
    # 正则表达式
    # 训练时间（五次实验的总共）
    training_match = re.search(
        r'Total 5 Training time:\s*([\d.]+)\s*s',
        content,
        re.IGNORECASE
    )
    if training_match:
        results['Total_training_time_s'] = training_match.group(1)

    # 推理时间（五次实验的总共）
    inference_match = re.search(
        r'Total 5 Inference time:\s*([\d.]+)\s*s',
        content,
        re.IGNORECASE
    )
    if inference_match:
        results['Total_inference_time_s'] = inference_match.group(1)


    # =================================================================
    # 最高OA及其轮次==============================
    # =================================================================
    # 正则表达式
    pattern = r"Accuracy\s*:\s*(\d+(?:\.\d+)?)%"
    # 查找所有匹配项
    matches = re.findall(pattern, content)
    # 提取最高 Accuracy 的轮次数和值
    if matches:
        # 将匹配结果转换为 (轮次数, Accuracy 值) 的列表
        matches = [(int(round_num), float(accuracy)) for round_num, accuracy in enumerate(matches)]

        # 找到 Accuracy 值最高的那一项
        Max_run, Max_accuracy = max(matches, key=lambda x: x[1])
        results['Max_run'] = Max_run
        results['Max_accuracy'] = Max_accuracy

        # print(f"最高 Accuracy 出现在 Round {Max_run}, 值为 {Max_accuracy}%")

    First_OA = re.search(
        r"Accuracy\s*:\s*(\d+(?:\.\d+)?)%",
        content,
        re.IGNORECASE
    )
    if First_OA:
        # print("+++First_OA+++")
        results['First_OA'] = First_OA.group(1)

    # =================================================================
    # Part 1: 精准匹配三个参数（允许分散出现和跨行）
    # =================================================================
    # 1. Trainable Params
    trainable_match = re.search(
        r'Trainable\s+params:\s*([\d,]+)',  # 匹配 "Trainable params: 4,070"
        content,
        re.IGNORECASE | re.DOTALL
    )
    if trainable_match:
        results['trainable_params'] = trainable_match.group(1).replace(',', '')

    blocks = re.split(r'=+\n', content)  # 以 "====...\n" 分割区块
    for block in blocks:
        # 提取每个区块的内容行
        lines = [line.strip() for line in block.split('\n') if line.strip()]

        # =================================================================
        # Step 2: 解析第一区块参数（Total mult-adds 等）
        # =================================================================
        if 'Total params:' in block and (
                'Total mult-adds (M):' in block or 'Total mult-adds (Units.MEGABYTES):' in block):
            for line in lines:
                # 匹配两种格式：
                # 1. "Total mult-adds (M): "
                # 2. "Total mult-adds (Units.MEGABYTES): "
                if line.startswith('Total mult-adds'):
                    match = re.search(r'Total mult-adds \((M|Units\.MEGABYTES)\):\s*([\d.]+)', line)
                    if match:
                        value = match.group(2)  # 提取数值（如 9.26）
                        results['total_mult_adds_M'] = value  # 仍然用原来的 key 存储
                # 其他参数（按需添加，保持不变）
                elif line.startswith('Trainable params:'):
                    value = line.split(':')[1].strip().replace(',', '')
                    results['trainable_params'] = value

        # =================================================================
        # Step 3: 解析第二区块参数（Estimated Total Size 等）
        # =================================================================
        if 'Input size (MB):' in block and 'Estimated Total Size (MB):' in block:
            for line in lines:
                if line.startswith('Estimated Total Size (MB):'):
                    value = re.search(r':\s*([\d.]+)', line).group(1)
                    results['model_size_MB'] = value

    # =================================================================
    # Part 2: 解析时间指标（独立匹配，不要求连续）
    # =================================================================
    # 训练时间（全局首次出现）
    training_match = re.search(
        r'Training time:\s*([\d.]+)\s*s',
        content,
        re.IGNORECASE
    )
    if training_match:
        results['training_time_s'] = training_match.group(1)

    # 推理时间（全局首次出现）
    inference_match = re.search(
        r'Inference time:\s*([\d.]+)\s*s',
        content,
        re.IGNORECASE
    )
    if inference_match:
        results['inference_time_s'] = inference_match.group(1)

    # =================================================================
    # Part 3: 解析原有精度指标（OA/AA/Kappa/类别精度）
    # =================================================================
    agregated_section = re.search(
        r'Agregated results\s*:\n(.*?)(?=\nAgregated|\Z)',
        content,
        re.DOTALL
    )
    if agregated_section:
        section_text = agregated_section.group(1)

        # 精度指标（OA/AA/Kappa）
        metric_pattern = re.compile(
            r'^(OA|AA|Kappa|Accuracy):\s*([\d.]+)\s*[±+]\-\s*([\d.]+)',
            re.MULTILINE
        )
        for match in metric_pattern.finditer(section_text):
            metric_name = 'OA' if match.group(1) == 'Accuracy' else match.group(1)
            results[metric_name] = f"{match.group(2)} ± {match.group(3)}"

        # 类别精度
        class_pattern = re.compile(
            r'\t(.+?):\s*([\d.]+)\s*[±+]\-\s*([\d.]+)\n',
            re.MULTILINE
        )
        for match in class_pattern.finditer(section_text):
            class_name = match.group(1)
            results['class_acc'][class_name] = f"{match.group(2)} ± {match.group(3)}"

    return results


def main(base_dir, out_dir):
    dataset_records = defaultdict(lambda: defaultdict(dict))

    # 遍历所有模型和数据集
    for model in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model)
        if not os.path.isdir(model_path):
            continue

        for dataset in os.listdir(model_path):
            log_dir = os.path.join(model_path, dataset, 'logs')
            if not os.path.exists(log_dir):
                continue

            # 处理每个日志文件
            for log_file in os.listdir(log_dir):
                if not log_file.endswith('.log'):
                    continue

                log_path = os.path.join(log_dir, log_file)
                parsed = parse_log_file(log_path)
                if not parsed:
                    continue

                # 合并所有指标到记录中
                model_data = dataset_records[dataset][model]

                model_data['First_OA'] = parsed['First_OA']
                model_data['Max_run'] = parsed['Max_run']
                model_data['Max_accuracy'] = parsed['Max_accuracy']

                # 1. 原有精度指标
                model_data['OA'] = parsed['OA']
                model_data['AA'] = parsed['AA']
                model_data['Kappa'] = parsed['Kappa']

                # 2. 新增参数
                model_data['Trainable Params'] = parsed['trainable_params']
                model_data['Mult-Adds (M)'] = parsed['total_mult_adds_M']
                model_data['Model Size (MB)'] = parsed['model_size_MB']
                model_data['Training Time (s)'] = parsed['training_time_s']
                model_data['Inference Time (s)'] = parsed['inference_time_s']

                model_data['Total_Training Time (s)'] = parsed['Total_training_time_s']
                model_data['Total_Inference Time (s)'] = parsed['Total_inference_time_s']

                # 3. 类别精度（动态列）
                for class_name, acc in parsed['class_acc'].items():
                    model_data[class_name] = acc

    # 生成Excel文件
    with pd.ExcelWriter(out_dir) as writer:
        for dataset, models in dataset_records.items():
            # 构建DataFrame，确保列顺序
            df = pd.DataFrame.from_dict(models, orient='index')

            # +++ 新增自定义排序代码开始 +++
            all_models = df.index.tolist()
            # 定义你想要的模型顺序（按实际模型名修改！）
            custom_model_order = ['A2S2K', 'SSFTT', 'MorphFormer', 'LRDTN', 'CSCANet', 'S2VNet', 'LS2CM', 'ELS2T', 'LMSS', 'CLOLN', 'ACB']  # 替换为你的模型名

            ordered_index = []
            # 首先添加所有在自定义顺序中的模型
            for model in custom_model_order:
                if model in all_models:
                    ordered_index.append(model)
            # 然后添加所有不在自定义顺序中的其他模型
            for model in all_models:
                if model not in custom_model_order:
                    ordered_index.append(model)
            # +++ 新增自定义排序代码结束 +++

            # 列排序：类别精度 -> 新增参数 -> 原有精度
            class_columns = [col for col in df.columns if col not in [
                'First_OA',
                'Max_run',
                'Max_accuracy',
                'OA', 'AA', 'Kappa',
                'Trainable Params', 'Mult-Adds (M)',
                'Model Size (MB)',
                'Training Time (s)',
                'Inference Time (s)',
                'Total_Training Time (s)',
                'Total_Inference Time (s)'
            ]]
            ordered_columns = class_columns + [
                'OA', 'AA', 'Kappa',
                'Total_Training Time (s)',
                'Total_Inference Time (s)',
                'Trainable Params',
                'Mult-Adds (M)',
                'Training Time (s)',
                'Inference Time (s)',
                # 'Model Size (MB)',
                'First_OA',
                'Max_run',
                'Max_accuracy'
            ]
            # 使用自定义顺序重新索引数据框
            df = df.reindex(ordered_index)

            df[ordered_columns].T.to_excel(writer, sheet_name=dataset[:31])


if __name__ == "__main__":
    # main(base_dir="../checkpoints/", out_dir='full_metrics3.xlsx')
    base_path = r"F:\KDL\AAA_DelonKong\MyHSIC2\checkpoints\Datasets_1%"
    main(base_dir=base_path, out_dir='full_metrics_Datasets_1%.xlsx')
