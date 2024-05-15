# EX-03 Implementation of Linear Regression using Gradient Descent
<table>
<tr>
<td width=85% align=left>
    
### Aim:
To write a program to predict the profit of a city using the linear regression model with gradient descent.
</td> 
<td valign=top>

**DATE:**
</td>
</tr> 
</table>

### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
### Algorithm:
1. Import pandas, numpy and mathplotlib.pyplot.
2. Trace the best fit line and calculate the cost function.
3. Calculate the gradient descent and plot the graph for it.
4. Predict the profit for two population sizes.
```
Developed By: Sathish R  
Register No: 212222230138
```
### Program:
```Python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linearREG(X1,Y,learnRate=0.01,Iteration=1000):        # Add a column of ones to X for the intercept term
    X=np.c_[np.ones(len(X1)),X1]                          # Initialize theta with zeros
    theta=np.zeros(X.shape[1]).reshape(-1,1)              # Perform gradient descent
    for _ in range(Iteration):
        predictions=(X).dot(theta).reshape(-1,1)          # Calculate predictions
        errors=(predictions-Y).reshape(-1,1)              # Calculate errors
        theta-=learnRate*(1/len(X1))*X.T.dot(errors)      # Update theta using gradient descent
    return theta
data=pd.read_csv('CSVs/50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:,:-2].values) # Assuming the last column is your target variable 'Y' and the preceding column
print(X)
X1=X.astype(float)
scaler=StandardScaler()
Y=(data.iloc[1:,-1].values).reshape(-1,1)
print(Y)
X1scaled=scaler.fit_transform(X1)
Y1scaled=scaler.fit_transform(Y)
print(X1scaled,Y1Scaled)
theta=linearREG(X1scaled,Y1scaled)                             # Learn model parameters
newData=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)   # Predict target value for a new data point
newScaled=scaler.fit_transform(newData)
prediction=np.dot(np.append(1, newScaled), theta) 
prediction=prediction.reshape(-1,1) 
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```
### Output:
**data.head()** <br>
<img height=10% width=99% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/b3ef5ab5-c8d3-42d3-86d5-29eea435dac9"><br><br>
**X values**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Y values** <br>
<img height=10% width=48% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/4fa96a4f-0a85-4307-b011-2ab04b73b9a9">&emsp;<img height=10% width=28% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/72810905-e103-4c76-ae8c-8a62f25cce8b"><br>
<br>
**X scaled**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Y scaled** <br>
<img height=10% width=48% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/9b3626af-2148-45d1-a822-a4019da4a3f5">&emsp;<img height=10% width=28% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/a26092ce-9f5e-47b7-97e2-636a1ffe9dc7"><br><br>
**Predicted Value**<br>
<img height=5% width=49% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/5f807fd5-7777-40aa-9bb4-ac2508e9026e">


### Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
