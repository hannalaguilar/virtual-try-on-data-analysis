from typing import Optional
import json
import pandas as pd
from bokeh.io import export_png

import plots
from utils import group_small_segments


def create_dataframe(filename: str) -> pd.DataFrame:
    with open(filename) as f:
       data = json.load(f)['data']
    rows = []
    for entry in data:
        file_name = entry["file_name"]
        category_name = entry["category_name"]
        row = {
            "file_name": file_name,
            "category_name": category_name
        }
        # Add tag_info columns
        for tag in entry["tag_info"]:
            tag_name = tag["tag_name"]
            tag_category = tag["tag_category"]
            row[tag_name] = tag_category
        # Add processed row
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def process_df(columns_to_analyze: pd.DataFrame, column: str) -> pd.DataFrame:

    print(f"Value counts for {column}:")
    print(columns_to_analyze[column].value_counts(normalize=True, dropna=False))
    print("\n")

    df = columns_to_analyze[column].value_counts(normalize=True, dropna=False).reset_index()
    df[column] = df[column].str.title()
    df['cumulative'] = df['proportion'].cumsum()

    if df.shape[0] <= 7:
        df = group_small_segments(df, column)
        return df
    else:
        df_to_keep = df[df['cumulative'] <= 0.95]
        if df_to_keep.shape[0] <= 2:
            df_to_keep = df.iloc[:2, :]
        if df_to_keep.shape[0] > 8:
            df_to_keep = df_to_keep.head(8)
        df_to_keep = group_small_segments(df_to_keep, column)
        return df_to_keep


def main(filename: str, split: str = Optional['train'], save: bool = False) :
    df = create_dataframe(filename)
    columns_to_analyze = df.drop(columns=['file_name'])
    for column in columns_to_analyze.columns:
        column_df = process_df(columns_to_analyze, column)
        p = plots.plot_cat_distribution(column_df)
        if save:
            export_png(p, filename=f'figures/{split}/{column}.png')


if __name__ == '__main__':
    # main('vitonhd_train_tagged.json', split='train', save=True)
    main('vitonhd_test_tagged.json', split='test', save=True)
