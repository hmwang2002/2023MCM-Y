import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np


if __name__ == '__main__':
    dataPath = 'exp.xlsx'
    data = pd.read_excel(dataPath)

    sns.set(style='whitegrid', palette='tab10')

    # Observe the data density distribution
    fig = plt.figure()
    ax = sns.displot(data['Price'], )
    ax.set(xlabel='Price', title='Distribution of Price', )
    plt.show()

    # Handle outliers
    mean = np.mean(data['Price'])
    std = np.std(data['Price'])
    threshold = 3 * std
    data_WithoutOutliers = data[np.abs(data['Price'] - mean) <= threshold]

    # Look at the density distribution of the data again
    fig = plt.figure()
    ax = sns.displot(data=data_WithoutOutliers['Price'])
    ax.set(xlabel='Price', title='Distribution of Price without outliers')
    plt.show()

    fig = plt.figure()
    ax1 = sns.displot(data_WithoutOutliers['Y'])
    ax2 = sns.displot(data_WithoutOutliers['L'])
    plt.show()

    # read data
    data = pd.get_dummies(data, columns=['Make'])
    features = data.drop(['Price', 'Longitude', 'Latitude'], axis=1)
    feature_list = list(features.columns)
    features = np.array(features)
    labels = np.array(data['Price'])

    # Data padding with random forest

    # View data distribution
    sns.pairplot(data_WithoutOutliers, x_vars=['Y', 'L', 'flag'],
                 y_vars=['Price'])
    plt.show()

    # correlation matrix
    corrDF = data_WithoutOutliers.corr()
    corrDF['Price'].sort_values(ascending=False)
    # Set the pandas display options to display the DataFrame in its entirety
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(corrDF)

    # Split the dataset into training set and test set
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state=42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # 创建随机森林模型
    rmodel = RandomForestRegressor(n_estimators=1000, random_state=42)

    # Create a random forest model
    rmodel.fit(train_features, train_labels)
    # prediction test set
    predictions = rmodel.predict(test_features)
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2), '$')

    # Calculate the mean absolute percent error (MAPE)
    mape = 100 * (errors / test_labels)
    # Compute and display precision
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    score = rmodel.score(test_features, test_labels)
    print('Score:', score)

    importances = list(rmodel.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x:x[1], reverse=True)

    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    # Repurpose the two most important features for modeling
    rmodel_most_important = RandomForestRegressor(n_estimators=1000, random_state=42)
    important_indices = [feature_list.index('Y'), feature_list.index('L')]
    train_important = train_features[:, important_indices]
    test_important = test_features[:, important_indices]

    rmodel_most_important.fit(train_important, train_labels)
    predictions = rmodel_most_important.predict(test_important)
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2), '$')
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    score = rmodel.score(test_features, test_labels)
    print('Score:', score)

    # visualization
    plt.figure(figsize=(9, 6))
    plt.plot(range(len(test_labels)), test_labels, label='real')
    plt.plot(range(len(predictions)), predictions, label='predict')
    plt.legend(loc='upper right')
    plt.show()

    df = pd.read_excel('exp.xlsx')
    df_price = df['Price'].tolist()
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
    predict = rmodel_most_important.predict(input_)
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
