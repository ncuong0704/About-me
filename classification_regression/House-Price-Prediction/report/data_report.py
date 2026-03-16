from ydata_profiling import ProfileReport
import pandas as pd

# Lấy data
data = pd.read_csv('../BostonHousing.csv')
# Tạo báo cáo phân tích chi tiết
profile = ProfileReport(df=data, title='report boston housing', explorative=True)
profile.to_file('report_boston_housing.html')