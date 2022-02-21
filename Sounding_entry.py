from numpy.ma.core import array
import pandas as pd
import numpy as np
import datetime


Sounding_filepath = "Data/CAM00071867-data.txt"

with open(Sounding_filepath) as f:
   sounding_lines = f.readlines()
f.close

sounding_array = []
date_index = []
for i, line in enumerate(sounding_lines):
    if line[0] == "#":
        line = line.split()
        date = datetime.datetime(int(line[1]), int(line[2]), int(line[3]))
        date_index.append(date)

    else:
        line = line.replace("A", " ")
        line = line.replace("B", " ")
        # line = line.replace("\n", " ")
        line = line.split()
        if float(line[2]) == 50000:
            sounding_array.append(float(line[3]))
    
# sounding_data = pd.read_csv(Sounding_filepath)
# print(sounding_data[:26])
date_index = pd.DatetimeIndex(date_index)
#return pd.Series(sounding_array, index= date_index)
#sounding_data = np.array(np.mat(read_data))
