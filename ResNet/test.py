import csv

# 读取原始CSV文件并修改第一列
input_file = 'result.csv'  # 输入的CSV文件
output_file = 'output.csv'  # 输出修改后的CSV文件

# 打开原始CSV文件读取数据
with open(input_file, 'r', newline='') as infile:
    reader = csv.reader(infile)
    rows = list(reader)  # 读取所有行

# 遍历每一行，修改第一列的值
for row in rows:
    if row:  # 确保行不为空
        try:
            row[0] = str(int(row[0]) + 1)  # 将第一列的值加1
        except ValueError:
            pass  # 如果第一列的值不是数字，跳过该行

# 将修改后的数据写回新的CSV文件
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(rows)  # 写入所有行

print("修改完成！")
