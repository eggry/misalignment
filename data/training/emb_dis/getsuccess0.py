import pandas as pd

# 读取CSV文件
input_file = 'Infer_Output_CSV_qwen2-7b_ORIGINAL_SA_safebench150__.csv'  # 请将此文件名替换为你的实际文件名
df = pd.read_csv(input_file)

# 提取success列为0的数据行
subset_df = df[df['success'] == 0]

# 保存提取出的数据行到新文件
output_file = 'qwen2refusedsafebench.csv'
subset_df.to_csv(output_file, index=False)

print(f"成功提取到 {len(subset_df)} 行数据，并保存到 {output_file}")
