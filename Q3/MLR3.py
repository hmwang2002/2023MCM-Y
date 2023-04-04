# multiple linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# monohulls

file_path = ['US East.xlsx', 'HongKong.xlsx']
Legend = ['Model Prediction', 'Hong Kong']
COLORS = ['r', 'b']


if __name__ == "__main__":
    df = pd.read_excel('US East.xlsx')
    df_price = df['Price'].tolist()
    x = df[['Y', 'L']]
    y = df['Price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    model = LinearRegression()
    model.fit(x_train, y_train)
    a = model.intercept_  # intercept
    model.intercept_ += 124962.59391861025
    b = model.coef_  # coefficient
    print('intercept:', a)
    print('coefficient:', b)
    x_test = []
    y_test = []
    df = pd.read_excel('HongKong.xlsx')
    df_price = df['Price'].tolist()
    df_Y = df['Y'].tolist()
    df_L = df['L'].tolist()
    for i in range(0, len(df_Y)):
        x_test.append([df_Y[i], df_L[i]])
        y_test.append(df_price[i])
    Y_pred = model.predict(x_test)
    plt.plot(range(len(Y_pred)), Y_pred, COLORS[0])
    plt.plot(range(len(y_test)), y_test, COLORS[1])
    plt.legend(Legend, loc='upper right')
    plt.xlabel('Variant', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.title('Price prediction', fontsize=22)
    plt.grid()
    plt.show()
    diff = []
    for i in range(0, len(Y_pred)):
        diff.append(y_test[i] - Y_pred[i])
    print(diff)
    print(sum(diff) / len(diff))
