#scholar No.181112296

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sns.set(rc={'figure.figsize':(11.7,8.27)})
data=pd.read_csv('iris.csv')

iris=data.copy()

iris.info()
iris.describe()

plt.xlabel('Features')
plt.ylabel('Species')
plt.scatter(iris['SepalLengthCm'],iris['Species'],c='blue',label='Sepal_Length')
plt.scatter(iris['SepalWidthCm'],iris['Species'],c='green',label='sepal_width')
plt.scatter(iris['PetalLengthCm'],iris['Species'],c='red',label='petal_length')
plt.scatter(iris['PetalWidthCm'],iris['Species'],c='black',label='petal_width')
plt.legend(loc=4,prop={'size':8})
plt.show()

mapping = {
    'Iris-setosa' : 1,
    'Iris-versicolor' : 2,
    'Iris-virginica' : 3
}

X=iris.drop(['Species'],axis=1,inplace=False)
Y=iris['Species'].replace(mapping)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

base_pred=np.mean(y_test)
print(base_pred)

base_pred=np.repeat(base_pred,len(y_test))

base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error)

lgr=LinearRegression(fit_intercept=True)

model_lin1=lgr.fit(x_train,y_train)

iris_predictions_lin1=lgr.predict(x_test)

lin_mse1=mean_squared_error(y_test,iris_predictions_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

r2_lin_test1=model_lin1.score(x_test,y_test)
r2_lin_train1=model_lin1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)

residuals=y_test-iris_predictions_lin1
sns.regplot(x=iris_predictions_lin1,y=residuals,scatter=True,fit_reg=False)
residuals.describe()