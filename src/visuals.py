import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def countplot_percentage(df: pd.DataFrame, title:str="", show_percentage:bool=True) -> None:
    """
    shows countplot and percentage of categories
    """
    total = len(df)
    plt.figure(figsize=(10, 10))
    ax = sns.countplot(data=df, x="Type")
    plt.title(title)

    if show_percentage:
        for p in ax.patches:
            height = p.get_height()
            percentage = '{:.1f}%'.format(100 * height / total)
            ax.text(p.get_x() + p.get_width() / 2., height + 0.5, percentage, ha="center")

    plt.grid()
    plt.show()
