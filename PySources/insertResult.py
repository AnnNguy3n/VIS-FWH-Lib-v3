import sys
import sqlite3
import numpy as np
import os
import time

def process_and_insert(db_path: str, num_cycle: int, fml_shape: int, num_field: int, num_column: int):
    folder = os.path.dirname(db_path)
    bin_path = f"{folder}/result.bin"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("BEGIN")
    try:
        with open(bin_path, "rb") as f:
            # 1. Đọc count_fml
            count_fml = np.frombuffer(f.read(4 * num_cycle), dtype=np.int32)

            for i in range(num_cycle):
                n = count_fml[i]
                if n == 0:
                    continue  # không có gì để lưu

                # 3.1 Đọc fmls_idx
                fmls_idx = np.frombuffer(f.read(8 * n), dtype=np.int64)

                # 3.2 Đọc fmls_to_save và xử lý
                raw_fmls = np.frombuffer(f.read(fml_shape * n), dtype=np.uint8).reshape((n, fml_shape))
                fmls_to_save = np.empty((n, fml_shape // 2), dtype=np.int32)
                for col in range(0, fml_shape, 2):
                    combined = raw_fmls[:, col] * num_column + raw_fmls[:, col + 1]
                    fmls_to_save[:, col // 2] = combined

                # 3.3 Đọc fields_to_save
                fields_to_save = np.frombuffer(f.read(8 * num_field * n), dtype=np.float64).reshape((n, num_field))

                # 3.4 Insert vào database
                table_name = f"T{i}_{fml_shape//2}"

                # Chuẩn bị câu SQL insert
                placeholders = ",".join(["?"] * (1 + fml_shape // 2 + num_field))
                sql = f"INSERT INTO {table_name} VALUES ({placeholders})"

                # Ghép dữ liệu từng dòng
                data_to_insert = []
                for j in range(n):
                    row = [int(fmls_idx[j])] + list(map(int, fmls_to_save[j])) + list(map(float, fields_to_save[j]))
                    data_to_insert.append(row)

                cursor.executemany(sql, data_to_insert)

        queries_path = f"{folder}/queries.bin"
        with open(queries_path, "rb") as qf:
            queries_content = qf.read().decode('utf-8')
            cursor.executescript(queries_content)

        conn.commit()
    except Exception as ex:
        conn.rollback()
        raise ex
    finally:
        conn.close()


if __name__ == "__main__":
    db_path = sys.argv[1]
    num_cycle = int(sys.argv[2])
    fml_shape = int(sys.argv[3])
    num_field = int(sys.argv[4])
    num_column = int(sys.argv[5])

    a = time.time()
    process_and_insert(db_path, num_cycle, fml_shape, num_field, num_column)
    print("Insert Time:", time.time() - a)
