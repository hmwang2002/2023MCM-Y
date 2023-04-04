# multiple linear regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_excel('exp.xlsx')
    df_price = df['Price'].tolist()
    x = df[['Y', 'L']]
    y = df['Price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    model = LinearRegression()
    model.fit(x_train, y_train)
    a = model.intercept_  # intercept
    b = model.coef_  # coefficient
    print('intercept:', a)
    print('coefficient:', b)
    score = model.score(x_test, y_test)
    print('score:', score)
    Y_pred = model.predict(x_test)
    plt.figure(figsize=(9, 6))
    plt.plot(range(len(y_test)), y_test, 'r', label='real')
    plt.plot(range(len(Y_pred)), Y_pred, 'b', label='predict')
    plt.legend(loc='upper right')
    plt.show()
    input_ = [[0, 0.15], [0.0714285714285714, 0.15], [0.0714285714285714, 0.35], [0.0714285714285714, 0.4],
              [0.142857142857142, 0.35], [0.214285714285714, 0.2], [0.214285714285714, 0.35],
             [0.214285714285714, 0.55], [0.214285714285714, 0.65], [0.285714285714285, 0.2], [0.285714285714285, 0.35]
             , [0.285714285714285, 0.65], [0.357142857142857, 0.2], [0.357142857142857, 0.35],
            [0.428571428571428, 0.2], [0.428571428571428, 0.35], [0.428571428571428, 0.55]
            , [0.5, 0.2], [0.5, 0.45], [0.571428571428571, 0.2], [0.571428571428571, 0.45],
            [0.642857142857142, 0.05], [0.642857142857142, 0.1], [0.714285714285714, 0.2],
            [0.642857142857142, 0.45], [0.714285714285714, 0.2], [0.714285714285714, 0.95],
            [0.785714285714285, 0.6], [0.857142857142857, 0.1], [0.857142857142857, 0.2],
            [0.928571428571428, 0.2], [0.928571428571428, 0.45], [1, 0.55]]
    target = []
    ans = []
    predict = model.predict(input_)
    for i in range(0, len(input_)):
        target_ = []
        for j in range(0, len(df_price)):
            if input_[i][0] == df['Y'][j] and input_[i][1] == df['L'][j]:
                target_.append(df_price[j])
        target.append(target_)
    for t in target:
        mean_ = sum(t) / len(t)
        ans.append(1 - abs(predict[target.index(t)] - mean_) / mean_)
    avg_accuracy = sum(ans) / len(ans)
    print('average accuracy:', avg_accuracy)
    x_ = []
    for i in range(1, len(ans) + 1):
        x_.append(i)
    data = pd.DataFrame()
    data['variant'] = x_
    data['accuracy'] = ans
    pic = sns.pairplot(data, x_vars='variant', y_vars='accuracy', height=5, aspect=1.5, kind='reg')
    plt.show()
