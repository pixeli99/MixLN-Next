import pandas as pd
import matplotlib.pyplot as plt

# 滑动平均函数，保留第一个点为原始值
def moving_average_with_initial(data, window_size):
    """
    计算滑动平均，保留第一个点为原始值。
    """
    smoothed = data.rolling(window=window_size, min_periods=1, center=False).mean()
    return smoothed

# 读取数据
file_path = './wandb_export_2025-01-30T13_16_59.957+08_00.csv'
data = pd.read_csv(file_path)

# 筛选数据，包含前 1000 步
filtered_data = data[data['Step'] < 50000]

# 移动窗口大小
window_size = 16

# 应用滑动平均进行平滑
smoothed_data = filtered_data.copy()
smoothed_data['post'] = moving_average_with_initial(filtered_data['post'], window_size)
smoothed_data['pre'] = moving_average_with_initial(filtered_data['pre'], window_size)
smoothed_data['cod'] = moving_average_with_initial(filtered_data['cod'], window_size)

# 绘制平滑后的损失曲线
plt.figure(figsize=(14, 8))
# plt.plot(smoothed_data['Step'], smoothed_data['post'], label='$Post-LN$', color='red', linewidth=2)
plt.plot(smoothed_data['Step'], smoothed_data['pre'], label='Pre-LN', color='blue', linewidth=2)
plt.plot(smoothed_data['Step'], smoothed_data['cod'], label='LayerNorm Scaling', color='red', linewidth=2)

# 图表设置
plt.xlabel('Step', fontsize=25, fontweight='bold')
plt.ylabel('Loss', fontsize=25, fontweight='bold')
plt.title('Comparison of Training Loss Curves on C4 Dataset for LLaMa-1B', fontsize=30)
plt.legend(fontsize=22)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 调整布局并保存图表
plt.tight_layout()
plt.savefig('loss_cod.pdf', bbox_inches='tight')
plt.show()