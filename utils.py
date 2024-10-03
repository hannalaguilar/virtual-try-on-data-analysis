import pandas as pd


def group_small_segments(df: pd.DataFrame, column: str, threshold: float =0.02) -> pd.DataFrame:
    df = df[df['proportion'] >= threshold]
    others_proportion = 1 - df.iloc[-1, -1]
    if others_proportion >= 0.002:
        others_row = pd.DataFrame([{column: 'Others', 'proportion': others_proportion}])
        df = pd.concat([df, others_row], ignore_index=True)
    return df[[column, 'proportion']]
