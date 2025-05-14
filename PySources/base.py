import os
import sys
import pandas as pd
import numpy as np
import numba as nb


def check_required_cols(data: pd.DataFrame, required_cols: set):
    missing_cols = required_cols - set(data.columns)
    assert not missing_cols


def check_column_dtypes(data: pd.DataFrame, expected_dtypes: dict):
    for col, expected_dtype in expected_dtypes.items():
        if isinstance(expected_dtype, list):
            assert data[col].dtype in expected_dtype
        else:
            assert data[col].dtype == expected_dtype


class Base:
    def __init__(self, data: pd.DataFrame, interest: float, valuearg_threshold: float):
        data = data.reset_index(drop=True).fillna(0.0)

        # Check các cột bắt buộc
        dropped_cols = {"TIME", "PROFIT", "SYMBOL", "VALUEARG"}
        check_required_cols(data, dropped_cols)

        # Check dtypes
        check_column_dtypes(data, {"TIME": "int64",
                                   "PROFIT": "float64",
                                   "VALUEARG": ["int64", "float64"]})

        # Cột TIME phải giảm dần, PROFIT & VALUE không âm
        assert data["TIME"].is_monotonic_decreasing
        assert (data["PROFIT"] >= 0.0).all()
        assert (data["VALUEARG"] >= 0.0).all()

        # Check các chu kỳ trong INDEX, lập INDEX
        unique_times = data["TIME"].unique()
        assert np.array_equal(
            np.arange(data["TIME"].max(), data["TIME"].min()-1, -1, dtype=int),
            unique_times
        )
        self.INDEX = np.searchsorted(-data["TIME"], -unique_times, side="left")
        self.INDEX = np.append(self.INDEX, data.shape[0])

        # Check SYMBOL có unique ở mỗi chu kỳ hay không
        assert np.array_equal(
            data.groupby("TIME", sort=False)["SYMBOL"].nunique().values,
            np.diff(self.INDEX)
        )

        # Loại bỏ các cột có kiểu dữ liệu không phải là int64 hoặc float64
        dropped_cols.update(data.select_dtypes(exclude=["int64", "float64"]).columns)
        self.dropped_cols = dropped_cols
        print("Các cột không được coi là biến chạy:", dropped_cols)

        # Mã hoá SYMBOL thành số nguyên
        unique_symbols, inverse = np.unique(data["SYMBOL"], return_inverse=True)
        self.symbol_name = dict(enumerate(unique_symbols))

        data["SYMBOL_encoded"] = inverse
        data.sort_values(["TIME", "SYMBOL_encoded"], inplace=True, ascending=[False, True], ignore_index=True)
        self.SYMBOL = data["SYMBOL_encoded"].to_numpy(int)
        data.drop(columns=["SYMBOL_encoded"], inplace=True)

        # Các thuộc tính
        self.data = data
        self.INTEREST = interest
        self.PROFIT = np.array(np.maximum(data["PROFIT"], 5e-324))
        self.VALUEARG = data["VALUEARG"].to_numpy(float)
        self.BOOL_ARG = self.VALUEARG >= valuearg_threshold

        operand_data = data.drop(columns=dropped_cols)
        self.OPERAND = operand_data.to_numpy(float).transpose()
        self.operand_name = dict(enumerate(operand_data.columns))

    def save_array(self, folder, name, dtype):
        array = getattr(self, name)
        with open(f"{folder}/{name}.bin", "wb") as f:
            f.write(np.array(array.shape, np.int32).tobytes())
            f.write(np.array(array, dtype).tobytes())

    def extract_data(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.save_array(folder, "BOOL_ARG", np.int32)
        self.save_array(folder, "INDEX", np.int32)
        self.save_array(folder, "SYMBOL", np.int32)
        self.save_array(folder, "OPERAND", np.float64)
        self.save_array(folder, "PROFIT", np.float64)


if __name__ == "__main__":
    data_path = sys.argv[1]
    interest = float(sys.argv[2])
    valuearg_threshold = float(sys.argv[3])
    folder_save = sys.argv[4]

    data = pd.read_excel(data_path)
    vis = Base(data, interest, valuearg_threshold)
    vis.extract_data(folder_save)
