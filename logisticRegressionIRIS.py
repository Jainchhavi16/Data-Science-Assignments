#Scholar Number:181112296

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

X=iris.drop(['Species'],axis=1,inplace=False)
Y=iris['Species']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

model=LogisticRegression()
model.fit(x_train,y_train)

predictions=model.predict(x_test)

print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))