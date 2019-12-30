import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


salary_data=pd.read_csv('house.csv')
x=salary_data.iloc[:,5:6].values
y=salary_data.iloc[:,2:3].values



class regression:
    def train_test_split(self,x_data,y_data,p):
        shuffle_index=np.random.permutation(len(x))
        x_data,y_data=x_data[shuffle_index],y_data[shuffle_index]
        self.l=round(len(x)*p)
        self.x_train=x_data[self.l:]
        self.x_test=x_data[0:self.l]
        self.y_train=y_data[self.l:]
        self.y_test=y_data[0:self.l]
    def simple_linear_regression(self):
        self.x_new=np.c_[np.ones((len(self.x_train),1)),self.x_train]
        c_and_m=np.linalg.inv(self.x_new.T.dot(self.x_new)).dot(self.x_new.T).dot(self.y_train)
        self.x_test_new=np.c_[np.ones((len(self.x_test),1)),self.x_test]
        self.y_pred=[i*c_and_m[1]+c_and_m[0] for i in self.x_test]
    def visual(self):
        plt.scatter(self.x_train,self.y_train,s=40,color='green')
        plt.plot(self.x_test,self.y_pred,'r-',marker='o',markerfacecolor='blue',markersize=5)
        plt.xlabel('sq_living')
        plt.ylabel('price')
        plt.title('simple linear regression')
        plt.show()
        
model=regression()
model.train_test_split(x,y,0.3)
model.simple_linear_regression()
model.visual()

