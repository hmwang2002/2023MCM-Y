import pandas as pd
import numpy as np


# use 3sigma to remove outliers
def remove_outliers(data):
    n = 3
    data_price = data['Listing Price (USD)']
    data_make = data['Make']
    data_variant = data['Variant']
    data_copy = pd.Series(dtype=np.int64)
    make = data_make[0]
    variant = data_variant[0]
    for i in range(0, len(data_price)):
        if data_make[i] == make and data_variant[i] == variant:
            data_copy = np.append(data_copy, data_price[i])
        else:
            mean = np.mean(data_copy)
            std = np.std(data_copy)
            for j in range(0, len(data_copy)):
                if data_copy[j] > mean + n * std or data_copy[j] < mean - n * std:
                    print(data_copy[j])
            data_copy = pd.Series(dtype=np.int64)
            make = data_make[i]
            variant = data_variant[i]


if __name__ == '__main__':
    # first sheet
    data0 = pd.read_excel('2023_MCM_Problem_Y_Boats.xlsx', sheet_name='Monohulled Sailboats ')
    remove_outliers(data0)
    # second sheet
    data1 = pd.read_excel('2023_MCM_Problem_Y_Boats.xlsx', sheet_name='Catamarans')
    remove_outliers(data1)
