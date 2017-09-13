from random import choice
from numpy import array, dot, random
import numpy as np
import math
import matplotlib.pylab as plt

training_data_or = [(array([0, 0, 1]), 0),      #ques1
                (array([0, 1, 1]), 1),
                (array([1, 0, 1]), 1),
                (array([1, 1, 1]), 1), ]

testing_data_xor=[(array([0, 0, 1]), 0),        #ques2
                (array([0, 1, 1]), 1),
                (array([1, 0, 1]), 1),
                (array([1, 1, 1]), 0),]
label_train=[0,1,1,1]
label_test=[0,1,1,0]

def activation_func(x):
    if x<0:
        return 0
    else:
        return 1

def choice(data):
    l=len(data)
    r=random.randint(0,l)
    return data[r][0],data[r][1]

def perceptron(n,eta,data):                     #ques3
    weight=np.random.random((3))
    for i in range(n):
        x, expected = choice(data)
        res=np.dot(x,weight)
        prediction=activation_func(res)
        error=prediction-expected
        #print(weight)
        if error:
            weight-=eta*error*x
    #print("END")
    return weight

def transform(data):                            #ques4
    a=[0,0]
    b=[1,1]
    x=data
    lista=array([x[0]-a[0],x[1]-a[1]])
    listb=array([x[0]-b[0],x[1]-b[1]])
    y1=math.exp(-1/2*(np.dot(lista,lista)))
    y2=math.exp(-1/2*(np.dot(listb,listb)))
    ret = array([y1,y2,1])
    return ret

def plot(data,weight,label):                                 #ques6
    xs=[x[0][0]for x in data]
    ys=[x[0][1]for x in data]
    col = ['b','g']
    for x,y,l in zip(xs,ys,label):
        if l==1:
            plt.scatter(x,y,c=col[0])
        else:
            plt.scatter(x,y,c=col[1])
    w1, w2 ,b = weight
    x = float(-b / w1)
    y = float(-b / w2)
    d = y
    c = -y / x
    x_ = array([0, x])
    y_= c * x_ + d
    plt.plot(x_,y_)
    plt.show()

w=perceptron(500,0.1,training_data_or)              #ques5
w_test=perceptron(500,1,testing_data_xor)
for i in range(len(training_data_or)):
    print("Train:",training_data_or[i][0],training_data_or[i][1],activation_func(np.dot(w,training_data_or[i][0])))
for i in range (len(testing_data_xor)):
    print("Test:",testing_data_xor[i][0],testing_data_xor[i][1], activation_func(np.dot(w_test,testing_data_xor[i][0])))

transformed_training_data=[]
transformed_test_data=[]
for i in range(len(training_data_or)):
    transformed_training_data.append((transform(training_data_or[i][0]),label_train[i]))
for i in range(len(testing_data_xor)):
    transformed_test_data.append((transform(testing_data_xor[i][0]),label_test[i]))         #ques9

w_transformed=perceptron(500,0.15,transformed_training_data)     #ques10
w_transformed_test=perceptron(500,.1,transformed_test_data)

for i in range(len(transformed_training_data)):
    print("Train_Transformed:", transformed_training_data[i][0],transformed_training_data[i][1],activation_func(np.dot(w_transformed,transformed_training_data[i][0])))
for i in range (len(transformed_test_data)):
    print("Test_Transformed:",transformed_test_data[i][0],transformed_test_data[i][1], activation_func(np.dot(w_transformed_test,transformed_test_data[i][0])))

plot(training_data_or,w,label_train)                #ques7
plot(testing_data_xor,w_test,label_test)                  #ques8
plot(transformed_training_data,w_transformed,label_train)       #ques11
plot(transformed_test_data,w_transformed_test,label_test)
