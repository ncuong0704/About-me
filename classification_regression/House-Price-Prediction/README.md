## House-Price-Prediction

This project aims to develop a machine learning model that accurately predicts housing prices using the Boston Housing dataset. By analyzing various features of houses, such as crime rate, number of rooms, and accessibility to highways, the model provides valuable insights for potential buyers or sellers in estimating housing prices. The project utilizes the powerful CatBoostRegressor algorithm for optimal performance and incorporates techniques like data preprocessing, exploratory data analysis, and model training. The trained model can be used as a tool to make informed decisions in the real estate market.

## Dataset

The Boston Housing dataset is imported from the sklearn.datasets module in Python. It consists of a total of 506 instances, each representing a house in the Boston area. The dataset contains 13 numerical features that describe various aspects of the houses, such as crime rate, average number of rooms, and proximity to employment centers. The target variable is the median value of owner-occupied homes in thousands of dollars.

### Features

CRIM: Per capita crime rate by town
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX: Nitric oxide concentration (parts per 10 million)
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built prior to 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town
B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT: Percentage of lower status of the population

CRIM: Tỷ lệ tội phạm của khu vực

ZN: Tỷ lệ đất dành cho nhà diện tích lớn

INDUS: Mức độ khu công nghiệp trong khu vực

CHAS: Nhà có nằm gần sông Charles hay không (1: có, 0: không)

NOX: Mức độ ô nhiễm không khí

RM: Số phòng trung bình của mỗi căn nhà

AGE: Tỷ lệ nhà xây trước năm 1940

DIS: Khoảng cách đến các trung tâm việc làm

RAD: Mức độ tiếp cận đường cao tốc

TAX: Thuế bất động sản của khu vực

PTRATIO: Tỷ lệ học sinh / giáo viên

B: Chỉ số liên quan đến tỷ lệ người da đen trong khu vực

LSTAT: Tỷ lệ dân số thu nhập thấp