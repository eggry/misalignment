import pandas as pd

# 读取CSV文件
df = pd.read_csv('data/evaluation/strongreject/strongreject_small_dataset.csv')

# 计算每份的行数
num_rows = df.shape[0] // 5

# 将数据分成5份并保存为新的CSV文件
for i in range(5):
    start_row = i * num_rows
    if i < 4:
        end_row = (i + 1) * num_rows
    else:  # 最后一份可能包含多余的行
        end_row = df.shape[0]
    
    df_subset = df.iloc[start_row:end_row]
    df_subset.to_csv(f'output_part_{i+1}.csv', index=False)

print("CSV文件已成功分成5份。")
