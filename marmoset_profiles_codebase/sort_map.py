import pandas as pd


def get_sort_map(areas_order):
    df_mapping = pd.DataFrame({"area": areas_order[1]})
    sort_map = df_mapping.reset_index().set_index("area")
    return sort_map
