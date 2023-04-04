import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Read the data
def read_data(file_path):
    df = pd.read_excel(file_path)
    return df


if __name__ == "__main__":
    fig = plt.figure(figsize=(15, 15))
    file_path = 'all.xlsx'
    df = read_data(file_path)
    sns.violinplot(x='Area', y='Price', hue='flag', data=df, split=True)
    plt.xticks(fontsize=10)
    plt.legend(['catamarans', 'monohulls'])
    plt.show()
