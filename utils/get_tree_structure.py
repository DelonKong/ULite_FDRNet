# -*- coding: utf-8 -*-

import os
import sys


def generate_tree_structure(start_path, max_depth=4):
    """
    生成完整的目录树结构（包括文件和文件夹）
    :param start_path: 起始目录路径
    :param max_depth: 最大递归深度 (0=根目录)
    """
    tree = []
    ignored = {'__pycache__', '.git', '.vscode'}  # 要忽略的文件夹列表

    def add_tree(path, prefix='', depth=0):
        if depth > max_depth:
            return

        try:
            # 获取所有条目（包括文件和文件夹）
            entries = os.listdir(path)
            # 过滤隐藏文件和特殊文件夹
            entries = [e for e in entries if not e.startswith('.') and e not in ignored]
            entries.sort(key=lambda e: (not os.path.isdir(os.path.join(path, e)), e))  # 文件夹在前
        except Exception as e:
            tree.append(f"{prefix}[Error: {str(e)}]")
            return

        if not entries:
            return

        for i, entry in enumerate(entries):
            full_path = os.path.join(path, entry)
            is_last = i == len(entries) - 1

            # 决定当前前缀符号
            connector = '└── ' if is_last else '├── '

            # 如果是文件夹
            if os.path.isdir(full_path):
                tree.append(f"{prefix}{connector}{entry}/")
                # 计算下一级的前缀
                new_prefix = prefix + ('    ' if is_last else '│   ')
                add_tree(full_path, new_prefix, depth + 1)
            # 如果是文件
            else:
                tree.append(f"{prefix}{connector}{entry}")

    # 添加根目录
    base_name = os.path.basename(os.path.normpath(start_path))
    tree.append(f"{base_name}/")

    # 开始生成树结构
    add_tree(start_path)
    return tree



if __name__ == "__main__":

    root_dir = r"F:\KDL\AAA_DelonKong\02_ULite_FDRNet_github\wandbtest"
    if not os.path.isdir(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist")
        sys.exit(1)

    print()
    # 生成并打印目录树
    for line in generate_tree_structure(root_dir, max_depth=4):
        print(line)