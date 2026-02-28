import pandas as pd
from cleaning_data import process_data


def get_top_rated_items(df, top_n=10):
    """
    Returns top N products based on average rating.
    """

    average_ratings = (
        df
        .groupby(['Name', 'Review Count', 'Brand', 'ImageURL'])['Rating']
        .mean()
        .reset_index()
    )

    top_rated_items = average_ratings.sort_values(
        by='Rating',
        ascending=False
    )

    return top_rated_items.head(top_n)


if __name__ == "__main__":
    raw_data = pd.read_csv("clean_data.csv")
    data = process_data(raw_data)

    print(get_top_rated_items(data))