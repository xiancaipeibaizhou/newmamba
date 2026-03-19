import os
import re
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parse existing metrics.txt files and aggregate them into a CSV.")
    parser.add_argument('--results_dir', type=str, default='results', help="Root directory containing model results")
    parser.add_argument('--output_csv', type=str, default='aggregated_metrics_results.csv', help="Output CSV filename")
    args = parser.parse_args()

    # 1. 查找所有的 metrics.txt 文件
    search_pattern = os.path.join(args.results_dir, "**", "metrics.txt")
    metrics_files = glob.glob(search_pattern, recursive=True)
    
    if not metrics_files:
        print(f"❌ No 'metrics.txt' files found in '{args.results_dir}'.")
        return
        
    print(f"🎯 Found {len(metrics_files)} metrics files. Parsing...\n")

    # 2. 准备写入 CSV
    with open(args.output_csv, "w", encoding="utf-8") as csv_f:
        # 定义 CSV 表头
        csv_f.write("Dataset,Variant,Group,Hidden,Seq,Threshold,ACC,APR,RE,F1_Macro,F1_Weighted,AUC,ASA,FAR\n")

        # 3. 遍历解析每个文件
        for idx, filepath in enumerate(metrics_files, 1):
            print(f"[{idx}/{len(metrics_files)}] Parsing: {filepath}")
            
            # --- 核心修改：动态、稳健地获取 Dataset 名称 ---
            # 路径示例: results/unsw_nb15/WoLocal_NB_EXP1_BASE_dim128_seq5/20260227-014836/metrics.txt
            parts = filepath.split(os.sep)
            
            try:
                # 找到 "results" 在路径中的位置，它的紧接着下一级就是 dataset_name
                # 为了兼容 args.results_dir 被用户修改的情况，提取基准目录名
                base_dir_name = os.path.basename(os.path.normpath(args.results_dir)) 
                results_idx = parts.index(base_dir_name)
                dataset_name = parts[results_idx + 1]
            except (ValueError, IndexError):
                # 如果没找到，退化保底策略（倒数第4层）
                if len(parts) >= 4:
                    dataset_name = parts[-4]
                else:
                    print(f"  ⚠️ Skipping {filepath} (path too shallow to find dataset name)")
                    continue

            # 读取文本内容
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # --- 解析第一行头部信息 (Exp Name 和 Threshold) ---
            # 示例: === MILAN_DEFAULT_dim128_seq3 (Thresh: 0.47) ===
            header_match = re.search(r"===\s*(.*?)\s*\(Thresh:\s*([\d.]+)\)\s*===", content)
            if not header_match:
                print(f"  ⚠️ Cannot parse header in {filepath}, skipping...")
                continue
                
            exp_name = header_match.group(1)
            threshold = float(header_match.group(2))

            # 从 exp_name 中提取: Variant, Group, Hidden, Seq
            # 匹配格式: (Variant)_(Group)_dim(Hidden)_seq(Seq)
            name_match = re.match(r"^(.*?)_(.*)_dim(\d+)_seq(\d+)$", exp_name)
            if not name_match:
                print(f"  ⚠️ Cannot parse exp_name '{exp_name}' in {filepath}, skipping...")
                continue
                
            variant = name_match.group(1)
            group_str = name_match.group(2)
            hidden = int(name_match.group(3))
            seq_len = int(name_match.group(4))

            # --- 解析各项指标 ---
            # 建立一个字典存储指标，如果在txt中找不到则默认为 NaN
            metrics = {
                'ACC': 'NaN', 'APR': 'NaN', 'RE': 'NaN', 
                'F1 (Macro)': 'NaN', 'F1 (Weighted)': 'NaN', 
                'AUC': 'NaN', 'ASA': 'NaN', 'FAR': 'NaN'
            }

            for key in metrics.keys():
                # 动态构建正则匹配每个指标，例如匹配 "ACC: 0.9971"
                escaped_key = re.escape(key)
                metric_match = re.search(fr"{escaped_key}:\s*([\d.]+)", content)
                if metric_match:
                    metrics[key] = float(metric_match.group(1))

            # 4. 写入 CSV 行
            csv_line = (
                f"{dataset_name},{variant},{group_str},{hidden},{seq_len},{threshold:.4f},"
                f"{metrics['ACC']},{metrics['APR']},{metrics['RE']},"
                f"{metrics['F1 (Macro)']},{metrics['F1 (Weighted)']},{metrics['AUC']},"
                f"{metrics['ASA']},{metrics['FAR']}\n"
            )
            csv_f.write(csv_line)

    print(f"\n✅ All done! Results saved to '{args.output_csv}'")

if __name__ == "__main__":
    main()