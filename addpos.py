import pandas as pd

pos_dict = {
    'Alabama': [30.362994, -87.532664],
    'Antigua and Barbuda' : [17.2234721, -61.9554608],
    'Aruba': [12.52111, -69.968338],
    'Bahamas': [25.03428, -77.39628],
    'Barbados': [13.193887, -59.543198],
    'Belize': [17.189877, -88.49765],
    'Belgium': [50.503887, 4.469936],
    'British Virgin Islands': [18.420695, -64.639968],
    'Bulgaria': [42.733883, 25.48583],
    'California': [36.778261, -119.4179324],
    'Cayman Islands': [19.513469, -80.566956],
    'Connecticut': [41.603221, -73.087749],
    'Cork': [51.8968917, -8.4863157],
    'Croatia': [45.1, 15.2],
    'Cyprus': [35.126413, 33.429859],
    'Denmark': [56.26392, 9.501785],
    'Dominican Republic': [18.735693, -70.162651],
    'Estonia': [58.595272, 25.013607],
    'Florida': [27.6648274, -81.5157535],
    'France': [46.227638, 2.213749],
    'Georgia': [32.1656221, -82.9000751],
    'Germany': [51.165691, 10.451526],
    'Gibraltar': [36.140751, -5.353585],
    'Greece': [39.074208, 21.824312],
    'Grenada': [12.262776, -61.604171],
    'Guadeloupe': [16.995971, -62.067641],
    'Guatemala': [15.783471, -90.230759],
    'Hawaii': [19.8967662, -155.5827818],
    'Honduras': [15.199999, -86.241905],
    'Hungary': [47.162494, 19.5033041],
    'Illinois': [40.6331259, -89.3985283],
    'Ireland': [53.41291, -8.24389],
    'Italy': [41.87194, 12.56738],
    'Jersey': [49.214439, -2.13125],
    'Lagos': [6.5243793, 3.3792057],
    'Louisiana': [30.3918305, -92.3291024],
    'Maine': [45.253783, -69.4454699],
    'Malta': [35.937496, 14.375416],
    'Martinique': [14.641528, -61.024174],
    'Maryland': [39.0457538, -76.6412734],
    'Massachusetts': [42.4072107, -71.3824374],
    'Mexico': [23.634501, -102.552784],
    'Michigan': [44.3148443, -85.6023643],
    'Mississippi': [32.3546679, -89.3985283],
    'Monaco': [43.750298, 7.412841],
    'Netherlands': [52.132633, 5.291266],
    'Netherlands Antilles': [12.226079, -69.060087],
    'New Jersey': [40.0583238, -74.4056612],
    'New York': [43.2994285, -74.2179324],
    'North Carolina': [35.7595731, -79.0192997],
    'Norway': [60.472024, 8.468946],
    'Ohio': [40.4172871, -82.907123],
    'Oregon': [43.8041334, -120.5542012],
    'Panama': [8.537981, -80.782127],
    'Portugal': [39.399872, -8.224454],
    'Puerto Rico': [18.220833, -66.590149],
    'Rhode Island': [41.5800945, -71.477429],
    'Romania': [45.943161, 24.96676],
    'Saint Kitts and Nevis': [17.357822, -62.782998],
    'Saint Lucia': [13.909444, -60.978893],
    'Saint Vincent and the Grenadines': [12.984305, -61.287228],
    'Saint-Martin': [18.0708, -63.0501],
    'Sint Maarten (Dutch part)': [18.0425, -63.0548],
    'Slovenia': [46.151241, 14.995463],
    'South Carolina': [33.836081, -81.163725],
    'Spain': [40.463667, -3.74922],
    'Sweden': [60.128161, 18.643501],
    'Switzerland': [46.818188, 8.227512],
    'Texas': [31.9685998, -99.9018131],
    'Trinidad and Tobago': [10.691803, -61.222503],
    'Turkey': [38.963745, 35.243322],
    'U.S. Virgin Islands': [18.335765, -64.896335],
    'United Kingdom': [55.378051, -3.435973],
    'Virginia': [37.9268688, -78.0249023],
    'Washington': [47.7510741, -120.7401385],
    'West Indies': [18.335765, -64.896335],
    'Wisconsin': [44.268543, -89.616508],
}


if __name__ == "__main__":
    data = pd.read_excel('combination.xlsx')
    target = []
    data_make = data['Make'].tolist()
    data_Year1 = data['Year1'].tolist()
    data_Length1 = data['Length1'].tolist()
    data_price = data['Listing Price (USD)'].tolist()
    data_region = data['Geographic Region'].tolist()
    data_detailed_region = data['Country/Region/State '].tolist()
    for i in range(0, len(data_make)):
        target.append([data_make[i], data_Year1[i], data_Length1[i], data_price[i], data_region[i],
                       data_detailed_region[i],
                      pos_dict[data_detailed_region[i].lstrip()][0], pos_dict[data_detailed_region[i].lstrip()][1]])
    df = pd.DataFrame(target, columns=['Make', 'Year1', 'Length1', 'Listing Price (USD)', 'Geographic Region',
                                       'Country/Region/State', 'Latitude', 'Longitude'])
    df.to_excel('pos.xlsx', index=False)
