import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft



model_data = pd.read_pickle("train_data.pkl")
model_data["Day_of_Year"] = model_data.index.day_of_year 
# running_avg = 0.0
# for i in range(1,366):
#     Day = model_data[(model_data["Day_of_Year"] == i)]
#     running_avg += np.std(Day["Target_Value"])
    
# #print(model_data["TMAX"].corr(model_data["MONTH"]))
# print(running_avg/365)

c = fft.rfft(model_data["Target_Value"].values)

plt.plot(c)
plt.xlim(left = -100, right = 500)
plt.show()
