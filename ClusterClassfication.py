import pandas as pd

# add 'the Mediterranean and Baltic Sea' to one csv file
if __name__ == "__main__":
    data = pd.read_excel('combination.xlsx')
    data_make = data['Make'].tolist()
    data_Year1 = data['Year1'].tolist()
    data_Length1 = data['Length1'].tolist()
    data_price = data['Listing Price (USD)'].tolist()
    data_flag = data['flag'].tolist()
    ref = pd.read_csv('K_Means.csv')
    ref_cluster = ref['Clusters'].tolist()
    ref_longitude = ref['Longitude'].tolist()
    ref_latitude = ref['Latitude'].tolist()
    target = []
    for i in range(0, len(data)):
        if ref_cluster[i] == 4:
            target.append([data_make[i], data_Year1[i], data_Length1[i], data_price[i], ref_longitude[i],
                           ref_latitude[i], data_flag[i]])
    target = pd.DataFrame(target, columns=['Make', 'Year1', 'Length1', 'Listing Price (USD)', 'Longitude', 'Latitude',
                                           'flag'])
    target.to_csv('Caribbean.csv', index=False)

