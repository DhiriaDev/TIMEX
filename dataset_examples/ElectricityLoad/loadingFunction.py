'''
-----DATASET INFORMATION-------------
- Data set has no missing values.
- Values are in kW of each 15 min. To convert values in kWh values must be divided by 4.
- Each column represent one client(370 clients). Some clients were created after 2011. 
    In these cases consumption were considered zero.
- Number of entries : 140256
- All time labels report to Portuguese hour. 
- All days present 96 measures (24*4 since samples are every 15 minutes). 
- Every year in March time change day (which has only 23 hours) the values between 1:00 am and 2:00 am are zero for all points.
- Every year in October time change day (which has 25 hours) the values between 1:00 am and 2:00 am aggregate the consumption of two hours.

-----ATTRIBUTE INFORMATION--------
- Data set were saved as txt using csv format, using semi colon (;).
- First column present date and time as a string with the following format 'yyyy-mm-dd hh:mm:ss'
- Other columns present float values with consumption in kW
'''
import pandas as pd
from datetime import datetime

separator = ';'
index_column = 0
data_parser = lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

dataset_path = "https://drive.google.com/file/d/1HvNI3bC5oCcp0Xpt783g9igTOdgOKQ59/view?usp=sharing"
dataset_path = 'https://drive.google.com/uc?id=' + \
                dataset_path.split('/')[-2]


dataset = pd.read_csv(filepath_or_buffer='dataset_examples/ElectricityLoad/ElectricityLoadDiagrams2011_2014.txt', 
                        sep=separator, decimal=',',
                        index_col=index_column, parse_dates=[index_column], date_parser=data_parser,
                        low_memory=False)