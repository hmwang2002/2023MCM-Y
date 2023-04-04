# multiple linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm

file_path = ['US East.xlsx', 'US West.xlsx', 'Eu.xlsx', 'Ca.xlsx', 'Med.xlsx']
AREAS = ['U.S. East Coast', 'U.S. West Coast', 'European Atlantic coast', 'Caribbean',
         'the Mediterranean and Baltic Sea']
COLORS = ['r', 'g', 'm', 'c', 'y', 'b']
if __name__ == "__main__":
    plt.figure(figsize=(9, 6))
    for i in range(0, 5):
        df = pd.read_excel(file_path[i])
        df_price = df['Price'].tolist()
        x = df[['Y', 'L']]
        y = df['Price']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
        model = LinearRegression()
        model.fit(x_train, y_train)
        smodel = sm.OLS(y_train, x_train)
        print(smodel.fit().summary())
        a = model.intercept_  # intercept
        b = model.coef_  # coefficient
        print('intercept:', a)
        print('coefficient:', b)
        score = model.score(x_test, y_test)
        print(AREAS[i] + ' score:', score)
        x_test = [[0, 0.15], [0.0714285714285714, 0.15], [0.0714285714285714, 0.35], [0.0714285714285714, 0.4],
                  [0.142857142857142, 0.35], [0.214285714285714, 0.2], [0.214285714285714, 0.35],
                  [0.214285714285714, 0.55], [0.214285714285714, 0.65], [0.285714285714285, 0.2],
                  [0.285714285714285, 0.35]
            , [0.285714285714285, 0.65], [0.357142857142857, 0.2], [0.357142857142857, 0.35],
                  [0.428571428571428, 0.2], [0.428571428571428, 0.35], [0.428571428571428, 0.55]
            , [0.5, 0.2], [0.5, 0.45], [0.571428571428571, 0.2], [0.571428571428571, 0.45],
                  [0.642857142857142, 0.05], [0.642857142857142, 0.1], [0.714285714285714, 0.2],
                  [0.642857142857142, 0.45], [0.714285714285714, 0.2], [0.714285714285714, 0.95],
                  [0.785714285714285, 0.6], [0.857142857142857, 0.1], [0.857142857142857, 0.2],
                  [0.928571428571428, 0.2], [0.928571428571428, 0.45], [1, 0.55]]
        Y_pred = model.predict(x_test)
        plt.plot(range(len(Y_pred)), Y_pred, COLORS[i])
    plt.legend(AREAS, loc='upper right')
    plt.title('Price prediction', fontsize=22)
    plt.show()
