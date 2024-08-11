
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
#load datashet
print("Load MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train=np.reshape(x_train,(60000,784))/255.0
x_test=np.reshape(x_test,(10000,784))/255.0
y_train=np.matrix(np.eye(10)[y_train])
y_test=np.matrix(np.eye(10)[y_test])
print("----------------------------------")
print(x_train.shape)
print(y_train.shape)

def relu(x):
    return(np.maximum(0,x))

def drelu(x):
    x[x<=0]=0
    x[x>0]=1
    return x

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def softmax(x):
    return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))

def Forwardpass(X,Wh1,bh1,Wh2,bh2,Wo,bo):
    zh1=X@Wh1.T + bh1
    a1=relu(zh1)
    zh2=a1@Wh2.T + bh2
    a2=sigmoid(zh2)
    z=a2@Wo.T + bo
    o=softmax(z)
    return o
def AccTest(label,prediction):    # calculate the matching score
    OutMaxArg=np.argmax(prediction,axis=1)
    LabelMaxArg=np.argmax(label,axis=1)
    Accuracy=np.mean(OutMaxArg==LabelMaxArg)
    return Accuracy

learningRate=0.5
Epoch=3
NumTrainSamples=60000
NumTestSamples=10000

NumInputs=784
NumHiddenUnits=512
NumClasses=10
#inital weights
#hidden layer 1
Wh1=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits,NumInputs)))
bh1= np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh1= np.zeros((NumHiddenUnits,NumInputs))
dbh1= np.zeros((1,NumHiddenUnits))
#hidden layer 2
Wh2=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits,NumHiddenUnits)))
bh2= np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh2= np.zeros((NumHiddenUnits,NumHiddenUnits))
dbh2= np.zeros((1,NumHiddenUnits))
#Output layer
Wo=np.random.uniform(-0.5,0.5,(NumClasses,NumHiddenUnits))
bo= np.random.uniform(0,0.5,(1,NumClasses))
dWo= np.zeros((NumClasses,NumHiddenUnits))
dbo= np.zeros((1,NumClasses))

from IPython.display import clear_output
loss=[]
Acc=[]
batch_size=200
stochastic_samples = np.arange(NumTrainSamples)

for ep in range (Epoch):
  np.random.shuffle(stochastic_samples)
  for ite in range (0,NumTrainSamples,batch_size):  
    #feed fordware propagation
    batch_samples = stochastic_samples[ite:ite+batch_size]
    x=x_train[batch_samples,:]
    y=y_train[batch_samples,:]
    # hidden layer 1 computation
    zh1=x@Wh1.T + bh1
    a1=relu(zh1)
    # hidden layer 2 computation
    zh2=a1@Wh2.T + bh2
    a2=sigmoid(zh2)
    # output layer computation
    z=a2@Wo.T + bo
    o=softmax(z)
    
    # calculate loss
    loss.append(-np.sum(np.multiply(y,np.log10(o))))
    
    # error for the ouput layer
    d=o-y
    dWo=np.matmul(np.transpose(d),a2) 
    dbo=np.mean(d)    # consider a is 1 for bias
    
    # backpropagation for the 2nd hidden layer
    dhs2=np.multiply(d@Wo,np.multiply(a2,(1-a2)))
    dWh2=np.matmul(np.transpose(dhs2),a1)
    dbh2=np.mean(dhs2)  # consider a is 1 for bias
    
    # backpropagation for the 1st hidden layer
    dhs1=np.multiply(dhs2@Wh2,(a1>0))
    dWh1=np.matmul(np.transpose(dhs1),x)
    dbh=np.mean(dhs1)  # consider a is 1 for bias  
    
    #update weight
    Wo=Wo-learningRate*dWo/batch_size
    bo=bo-learningRate*dbo
    Wh2=Wh2-learningRate*dWh2/batch_size
    bh2=bh2-learningRate*dbh2
    Wh1=Wh1-learningRate*dWh1/batch_size
    bh1=bh1-learningRate*dbh1
    #Test accuracy with random innitial weights
  prediction = Forwardpass(x_test,Wh1,bh1,Wh2,bh2,Wo,bo)
  Acc.append(AccTest(y_test,prediction))
  clear_output(wait=True)
  print('Epoch:', ep )
  print('Accuracy:',AccTest(y_test,prediction) )
    # plt.plot([i for i, _ in enumerate(Acc)],Acc,'o')
    # plt.show()

prediction = Forwardpass(x_test,Wh1,bh1,Wh2,bh2,Wo,bo)
Rate = AccTest(y_test,prediction)
print(Rate)
