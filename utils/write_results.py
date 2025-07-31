# -*- coding: gbk -*-

import os
import re
import pandas as pd
from collections import defaultdict


def parse_log_file(log_path):
    """������־�ļ��������¾�����ָ��"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(log_path, 'r', encoding='gbk') as f:
            content = f.read()

    # ��ʼ������ֵ䣨���������ֶΣ�
    results = {
        'First_OA': 'N/A',
        # ԭ�о���ָ��
        'OA': 'N/A', 'AA': 'N/A', 'Kappa': 'N/A',
        # ��������
        'trainable_params': 'N/A',
        'total_mult_adds_M': 'N/A',
        'model_size_MB': 'N/A',
        'training_time_s': 'N/A',
        'inference_time_s': 'N/A',
        # ��𾫶�
        'class_acc': {},
        'Total_training_time_s': 'N/A',
        'Total_inference_time_s': 'N/A',
    }
    # =================================================================
    # ���ʵ����ܹ�ʱ��==============================
    # =================================================================
    # ������ʽ
    # ѵ��ʱ�䣨���ʵ����ܹ���
    training_match = re.search(
        r'Total 5 Training time:\s*([\d.]+)\s*s',
        content,
        re.IGNORECASE
    )
    if training_match:
        results['Total_training_time_s'] = training_match.group(1)

    # ����ʱ�䣨���ʵ����ܹ���
    inference_match = re.search(
        r'Total 5 Inference time:\s*([\d.]+)\s*s',
        content,
        re.IGNORECASE
    )
    if inference_match:
        results['Total_inference_time_s'] = inference_match.group(1)


    # =================================================================
    # ���OA�����ִ�==============================
    # =================================================================
    # ������ʽ
    pattern = r"Accuracy\s*:\s*(\d+(?:\.\d+)?)%"
    # ��������ƥ����
    matches = re.findall(pattern, content)
    # ��ȡ��� Accuracy ���ִ�����ֵ
    if matches:
        # ��ƥ����ת��Ϊ (�ִ���, Accuracy ֵ) ���б�
        matches = [(int(round_num), float(accuracy)) for round_num, accuracy in enumerate(matches)]

        # �ҵ� Accuracy ֵ��ߵ���һ��
        Max_run, Max_accuracy = max(matches, key=lambda x: x[1])
        results['Max_run'] = Max_run
        results['Max_accuracy'] = Max_accuracy

        # print(f"��� Accuracy ������ Round {Max_run}, ֵΪ {Max_accuracy}%")

    First_OA = re.search(
        r"Accuracy\s*:\s*(\d+(?:\.\d+)?)%",
        content,
        re.IGNORECASE
    )
    if First_OA:
        # print("+++First_OA+++")
        results['First_OA'] = First_OA.group(1)

    # =================================================================
    # Part 1: ��׼ƥ�����������������ɢ���ֺͿ��У�
    # =================================================================
    # 1. Trainable Params
    trainable_match = re.search(
        r'Trainable\s+params:\s*([\d,]+)',  # ƥ�� "Trainable params: 4,070"
        content,
        re.IGNORECASE | re.DOTALL
    )
    if trainable_match:
        results['trainable_params'] = trainable_match.group(1).replace(',', '')

    blocks = re.split(r'=+\n', content)  # �� "====...\n" �ָ�����
    for block in blocks:
        # ��ȡÿ�������������
        lines = [line.strip() for line in block.split('\n') if line.strip()]

        # =================================================================
        # Step 2: ������һ���������Total mult-adds �ȣ�
        # =================================================================
        if 'Total params:' in block and (
                'Total mult-adds (M):' in block or 'Total mult-adds (Units.MEGABYTES):' in block):
            for line in lines:
                # ƥ�����ָ�ʽ��
                # 1. "Total mult-adds (M): "
                # 2. "Total mult-adds (Units.MEGABYTES): "
                if line.startswith('Total mult-adds'):
                    match = re.search(r'Total mult-adds \((M|Units\.MEGABYTES)\):\s*([\d.]+)', line)
                    if match:
                        value = match.group(2)  # ��ȡ��ֵ���� 9.26��
                        results['total_mult_adds_M'] = value  # ��Ȼ��ԭ���� key �洢
                # ����������������ӣ����ֲ��䣩
                elif line.startswith('Trainable params:'):
                    value = line.split(':')[1].strip().replace(',', '')
                    results['trainable_params'] = value

        # =================================================================
        # Step 3: �����ڶ����������Estimated Total Size �ȣ�
        # =================================================================
        if 'Input size (MB):' in block and 'Estimated Total Size (MB):' in block:
            for line in lines:
                if line.startswith('Estimated Total Size (MB):'):
                    value = re.search(r':\s*([\d.]+)', line).group(1)
                    results['model_size_MB'] = value

    # =================================================================
    # Part 2: ����ʱ��ָ�꣨����ƥ�䣬��Ҫ��������
    # =================================================================
    # ѵ��ʱ�䣨ȫ���״γ��֣�
    training_match = re.search(
        r'Training time:\s*([\d.]+)\s*s',
        content,
        re.IGNORECASE
    )
    if training_match:
        results['training_time_s'] = training_match.group(1)

    # ����ʱ�䣨ȫ���״γ��֣�
    inference_match = re.search(
        r'Inference time:\s*([\d.]+)\s*s',
        content,
        re.IGNORECASE
    )
    if inference_match:
        results['inference_time_s'] = inference_match.group(1)

    # =================================================================
    # Part 3: ����ԭ�о���ָ�꣨OA/AA/Kappa/��𾫶ȣ�
    # =================================================================
    agregated_section = re.search(
        r'Agregated results\s*:\n(.*?)(?=\nAgregated|\Z)',
        content,
        re.DOTALL
    )
    if agregated_section:
        section_text = agregated_section.group(1)

        # ����ָ�꣨OA/AA/Kappa��
        metric_pattern = re.compile(
            r'^(OA|AA|Kappa|Accuracy):\s*([\d.]+)\s*[��+]\-\s*([\d.]+)',
            re.MULTILINE
        )
        for match in metric_pattern.finditer(section_text):
            metric_name = 'OA' if match.group(1) == 'Accuracy' else match.group(1)
            results[metric_name] = f"{match.group(2)} �� {match.group(3)}"

        # ��𾫶�
        class_pattern = re.compile(
            r'\t(.+?):\s*([\d.]+)\s*[��+]\-\s*([\d.]+)\n',
            re.MULTILINE
        )
        for match in class_pattern.finditer(section_text):
            class_name = match.group(1)
            results['class_acc'][class_name] = f"{match.group(2)} �� {match.group(3)}"

    return results


def main(base_dir, out_dir):
    dataset_records = defaultdict(lambda: defaultdict(dict))

    # ��������ģ�ͺ����ݼ�
    for model in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model)
        if not os.path.isdir(model_path):
            continue

        for dataset in os.listdir(model_path):
            log_dir = os.path.join(model_path, dataset, 'logs')
            if not os.path.exists(log_dir):
                continue

            # ����ÿ����־�ļ�
            for log_file in os.listdir(log_dir):
                if not log_file.endswith('.log'):
                    continue

                log_path = os.path.join(log_dir, log_file)
                parsed = parse_log_file(log_path)
                if not parsed:
                    continue

                # �ϲ�����ָ�굽��¼��
                model_data = dataset_records[dataset][model]

                model_data['First_OA'] = parsed['First_OA']
                model_data['Max_run'] = parsed['Max_run']
                model_data['Max_accuracy'] = parsed['Max_accuracy']

                # 1. ԭ�о���ָ��
                model_data['OA'] = parsed['OA']
                model_data['AA'] = parsed['AA']
                model_data['Kappa'] = parsed['Kappa']

                # 2. ��������
                model_data['Trainable Params'] = parsed['trainable_params']
                model_data['Mult-Adds (M)'] = parsed['total_mult_adds_M']
                model_data['Model Size (MB)'] = parsed['model_size_MB']
                model_data['Training Time (s)'] = parsed['training_time_s']
                model_data['Inference Time (s)'] = parsed['inference_time_s']

                model_data['Total_Training Time (s)'] = parsed['Total_training_time_s']
                model_data['Total_Inference Time (s)'] = parsed['Total_inference_time_s']

                # 3. ��𾫶ȣ���̬�У�
                for class_name, acc in parsed['class_acc'].items():
                    model_data[class_name] = acc

    # ����Excel�ļ�
    with pd.ExcelWriter(out_dir) as writer:
        for dataset, models in dataset_records.items():
            # ����DataFrame��ȷ����˳��
            df = pd.DataFrame.from_dict(models, orient='index')

            # +++ �����Զ���������뿪ʼ +++
            all_models = df.index.tolist()
            # ��������Ҫ��ģ��˳�򣨰�ʵ��ģ�����޸ģ���
            custom_model_order = ['A2S2K', 'SSFTT', 'MorphFormer', 'LRDTN', 'CSCANet', 'S2VNet', 'LS2CM', 'ELS2T', 'LMSS', 'CLOLN', 'ACB']  # �滻Ϊ���ģ����

            ordered_index = []
            # ��������������Զ���˳���е�ģ��
            for model in custom_model_order:
                if model in all_models:
                    ordered_index.append(model)
            # Ȼ��������в����Զ���˳���е�����ģ��
            for model in all_models:
                if model not in custom_model_order:
                    ordered_index.append(model)
            # +++ �����Զ������������� +++

            # ��������𾫶� -> �������� -> ԭ�о���
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
            # ʹ���Զ���˳�������������ݿ�
            df = df.reindex(ordered_index)

            df[ordered_columns].T.to_excel(writer, sheet_name=dataset[:31])


if __name__ == "__main__":
    # main(base_dir="../checkpoints/", out_dir='full_metrics3.xlsx')
    base_path = r"F:\KDL\AAA_DelonKong\MyHSIC2\checkpoints\Datasets_1%"
    main(base_dir=base_path, out_dir='full_metrics_Datasets_1%.xlsx')
