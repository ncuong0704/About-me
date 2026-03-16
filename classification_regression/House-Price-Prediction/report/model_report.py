import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyRegressor

df = pd.read_csv("../BostonHousing.csv")

df.dropna(inplace=True)

x = df.drop("medv",axis=1)
y = df['medv']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=48)

ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test  = ss.transform(x_test)

# Thống kê xác suất của tất cả các thuật toán
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)

#                                Adjusted R-Squared  R-Squared  RMSE  Time Taken
# Model
# GradientBoostingRegressor                    0.92       0.93  2.66        0.15
# RandomForestRegressor                        0.88       0.90  3.17        0.42
# HistGradientBoostingRegressor                0.88       0.89  3.22        0.23
# XGBRegressor                                 0.88       0.89  3.22        1.14
# BaggingRegressor                             0.87       0.89  3.27        0.05
# LGBMRegressor                                0.87       0.88  3.36        0.09
# ExtraTreesRegressor                          0.85       0.87  3.59        0.31
# DecisionTreeRegressor                        0.76       0.79  4.55        0.01