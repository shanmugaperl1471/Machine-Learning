import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_excel(r"D:\PythonProject\ML\EX1.xlsx")
plt.xlabel('Year')
plt.ylabel('Canada Per Capital Income')
plt.scatter(df.year, df.pci, color='red', marker='+')
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df.pci)
x = int(input('Enter The Year To Be Predicted : '))
print('Prediction1 : ', reg.predict([[x]])[0])  # By inbuilt Method
m = reg.coef_
b = reg.intercept_
y = m*x+b
print('Prediction2 : ', y[0])  # By Manual
plt.plot(df.year, reg.predict(df[['year']]), color='blue')
plt.show()
