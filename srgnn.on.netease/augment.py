def augment_sessions(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            session_id, item_seq = line.strip().split()
            items = item_seq.split(',')

            # 只保留长度 >= 2 的子序列
            for i in range(2, len(items) + 1):
                new_session_id = f"{session_id}_{i-1}"
                new_item_seq = ','.join(items[:i])
                outfile.write(f"{new_session_id} {new_item_seq}\n")

    print(f"✅ 数据增强完成，写入：{output_path}")


augment_sessions('./netease.txt', './netease.aug.txt')