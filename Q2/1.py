import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import pandas as pd

if __name__ == "__main__":
    fig = plt.figure()
    ax = Axes3D(fig)
    df = pd.read_excel('US East.xlsx')
    df_price = df['Price'].tolist()
    df_Y = df['Y'].tolist()
    df_L = df['L'].tolist()
    ax.scatter(df_Y, df_L, df_price, c='r')

    x = np.arange(0, 1, 0.001)
    y = np.arange(0, 1, 0.001)
    X, Y = np.meshgrid(x, y)
    Z = 156614.81816578 * X + 281287.22740948 * Y + 23272.19982807935
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.savefig('1.png', dpi=300)
    plt.show()
    plt.clf()

