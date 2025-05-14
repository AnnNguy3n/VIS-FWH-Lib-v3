import pandas as pd


filter_fields = {
    "M_investMethod": {
        "command": "MIM_filter_"
    }
}


generate_method = {
    "HomogeneousPolynomial": {
        "command": "HP_method_"
    },
}


def compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame):
    assert df1.shape == df2.shape
    for i in range(df1.shape[1]):
        assert df1.columns[i] == df2.columns[i]
        col = df1.columns[i]
        print("Compare:", col)
        assert df1[col].dtype == df2[col].dtype
        if df1[col].dtype == "object":
            assert (df1[col] == df2[col]).all()
        elif df1[col].dtype in ["int64", "float64"]:
            assert (df1[col] - df2[col]).abs().max() <= 0.0
        else:
            assert False
