import os
import re
import glob
import matplotlib.pyplot as plt

# --- 全局图表设置 (适配顶刊的排版规范) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42  # 确保字体嵌入到 PDF 中
plt.rcParams['ps.fonttype'] = 42

def plot_training_history(log_file_path, save_path, title_name="Training History"):
    """
    读取 training_history.log 并绘制 Loss 与 Val AUPRC/F1 随时间变化的曲线图
    导出符合高水平期刊规范的高清 PDF 矢量图
    """
    epochs = []
    losses = []
    cl_losses = []
    val_metrics = []
    metric_name = "Val Metric" # 默认指标名称，由正则动态更新

    # 1. 解析日志文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 优化正则：兼容匹配 "Val F1" 或 "Val AUPRC"
            match = re.search(r"Epoch\s+(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*CL:\s*([\d.]+)\s*\|\s*(Val F1|Val AUPRC):\s*([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                cl_losses.append(float(match.group(3)))
                metric_name = match.group(4)  # 自动识别是 F1 还是 AUPRC
                val_metrics.append(float(match.group(5)))

    if not epochs:
        print(f"  ⚠️ No valid log data found in {log_file_path}")
        return

    # 2. 准备绘图 
    # 将比例调整为宽版，更适合双栏论文的单栏插图或跨栏插图
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- 左 Y 轴：Loss 曲线 ---
    color1 = '#C00000' # 深红色，稳重
    color2 = '#ED7D31' # 橙色，用于对比学习
    
    ax1.set_xlabel('Epochs', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', color='black', fontsize=14, fontweight='bold') 
    
    # 画 Total Loss 和 CL Loss
    l1 = ax1.plot(epochs, losses, color=color1, label='Total Loss', linewidth=2.5, marker='o', markersize=5, alpha=0.9)
    # 在图例中使用 LaTeX 语法标注 CL Loss
    l2 = ax1.plot(epochs, cl_losses, color=color2, label='Contrastive Loss ($\\mathcal{L}_{CL}$)', linewidth=2.5, linestyle='--', marker='^', markersize=5, alpha=0.9)
    
    ax1.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    # 修改网格线样式，使其不喧宾夺主
    ax1.grid(True, linestyle='--', alpha=0.4, color='gray')

    # --- 右 Y 轴：Validation 曲线 ---
    ax2 = ax1.twinx()  
    color3 = '#2F5597' # 深蓝色
    ax2.set_ylabel(metric_name, color=color3, fontsize=14, fontweight='bold')
    
    # 画验证集指标
    l3 = ax2.plot(epochs, val_metrics, color=color3, label=metric_name, linewidth=2.5, marker='s', markersize=5, alpha=0.9)
    ax2.tick_params(axis='y', labelcolor=color3, labelsize=12)
    
    # 动态设置右侧 Y 轴范围，留出顶部 15% 空间防止图例遮挡曲线
    min_m, max_m = min(val_metrics), max(val_metrics)
    padding = (max_m - min_m) * 0.15 if max_m > min_m else 0.1
    ax2.set_ylim([max(0, min_m - padding), min(1.05, max_m + padding)])

    # --- 合并图例 ---
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    # 图例加上黑框，放在中央偏右，确保在双 Y 轴中清晰可见
    ax1.legend(lines, labels, loc='center right', fontsize=12, frameon=True, edgecolor='black', framealpha=0.95)

    # 顶刊通常通过 LaTeX 的 \caption 来写标题，图内标题可选
    # 如果不需要图内标题，可以注释掉下面这一行
    plt.title(f'{title_name}', fontsize=15, fontweight='bold', pad=15)
    
    fig.tight_layout()
    
    # 3. 保存图片 (强制替换扩展名为 .pdf)
    save_path_pdf = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(save_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved learning curve to: {save_path_pdf}")

def main():
    search_pattern = os.path.join("results", "**", "training_history.log")
    log_files = glob.glob(search_pattern, recursive=True)

    if not log_files:
        print("❌ No training_history.log files found in 'results/' directory.")
        return

    print(f"🎯 Found {len(log_files)} training logs. Generating plots...\n")

    for idx, log_file in enumerate(log_files, 1):
        print(f"[{idx}/{len(log_files)}] Processing: {log_file}")
        
        model_dir = os.path.dirname(log_file)
        parts = model_dir.split(os.sep)
        
        if len(parts) >= 3:
            dataset_name = parts[-3]
            exp_name = parts[-2]
            # 美化标题显示
            title = f"{dataset_name.upper()} | {exp_name.replace('_', ' ').title()}"
        else:
            title = "Training Dynamics"

        save_path = os.path.join(model_dir, "learning_curve.png") 
        plot_training_history(log_file, save_path, title_name=title)

if __name__ == "__main__":
    main()