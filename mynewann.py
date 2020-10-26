import numpy as np
import pandas as pd

loss_cache=[]
lr=0
batch_size=0
layer_error=0
layer_error_cache=[]
weights=[]
bias=[]
shape=[]
z_cache=[]
a_cache=[]
batch_inputs=[]
epochs=0
vdw,vdb,sdw,sdb=0,0,0,0
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))



def relu(x):
    x=x.astype('float64')
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            if x[i,j]<=0:
                x[i,j]=0.01*x[i,j].astype('float64')
    return x
                
def der_relu(x):
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            if x[i,j]<0:
                x[i,j]=0.01
            else:
                x[i,j]=1
    return x.astype('float64')

def feed_relu(inputs,weights,bias):
    z=np.dot(inputs,weights)+bias
    a=relu(z)
    return([z,a])

def feed_num(inputs,weights,bias):
    z=np.dot(inputs,weights)+bias
    a=np.dot(inputs,weights)+bias
    return([z,a])

#calculating loss for each sample
def output_loss_calc(y,output):
    loss=np.subtract(y,output)**2
    der_loss=np.subtract(output,y)
    single_loss=np.multiply(der_loss,1)
    loss_cache.append(loss) #appending loss for each sample(used later to plot)
    return single_loss

#calculating loss for the mini-batch
def output_loss(y,output):
    loss=0
    final_loss=0
    for i in range(0,batch_size):
        loss+=output_loss_calc(y[i],output[i])
    final_loss=loss/float(batch_size)
    return final_loss

#finding errors for each layer using backpropagation    
def backprop(next_layer_error,next_layer_weights,layer_z):
    delta=np.multiply(np.dot(next_layer_weights,next_layer_error),layer_z)
    
    return delta


def grad_descent_calc(layer_error,prev_layer_a,weights,bias):
    global vdw,vdb,sdw,sdb
    del_b=np.array(layer_error)
    del_w=np.dot(prev_layer_a.reshape(-1,1,),np.array(layer_error).reshape(1,-1,))
   #using momentum
   
    vdw=0.9*vdw+0.1*del_w
    vdb=0.9*vdb+0.1*del_b
    
    #using RMSprop
    sdw=0.999*sdw+0.001*np.multiply(del_w,del_w)
    sdb=0.999*sdb+0.001*np.multiply(del_b,del_b)
    
    #finding change in weight and bias
    
    c=(vdw/np.add(np.sqrt(sdw),10**(-8)))*(lr/batch_size)
    d=(vdb/np.add(np.sqrt(sdb),10**(-8)))*(lr/batch_size)
    
    return([c,d])
    
        
class Dense:
    def __init__(self,learn_rate,batch_length,nn_shape,ep):
        global lr,batch_size,shape,epochs
        lr=learn_rate
        batch_size=batch_length
        shape=nn_shape 
        #IMPORTANT: Everywhere below len(shape) refers to the number of layers
        #IMPORTANT: First hidden layer has index 0, 2nd hidden layer has index 1 and so on.
        #initialising weights and bias
        for i in range(1,len(shape)):
            weights.append(0.01*np.random.randn(shape[i-1],shape[i]))
            bias.append(np.zeros((1,shape[i])))
        epochs=ep
        
    def train(self,data,y):
        global z_cache,a_cache,batch_inputs,batch_y,error_cache,epochs,vdw,vdb,sdw,sdb
        
        batch_inputs=[]
        batch_y=[]
        
        for i in range(0,int(data.shape[0]/batch_size)):
            batch_inputs.append(np.zeros(0).tolist())
            batch_y.append(np.zeros(0).tolist())
        #arranging input data in batches of size batch_size    
        index=0
        for i in range(0,data.shape[0],batch_size):
            for j in range(i,i+batch_size):
                batch_inputs[index].append(dat[j,0])
                batch_y[index].append(y[j,0])
            index+=1
        
        
        for e in range(0,epochs):
            #initialising z_cache,a_cache and error_cache(storing error of each layer)
            for i in range(0,int(data.shape[0]/batch_size)):
                error_cache=np.zeros(len(shape)-1).tolist() 
                z_cache=np.zeros((len(shape)-1,0)).tolist()
                a_cache=np.zeros((len(shape)-1,0)).tolist()
                for j in range(0,len(shape)-1):
                    z_cache[j]=np.array(z_cache[j])
                    a_cache[j]=np.array(a_cache[j])
                
                
                for j in range(0,batch_size):
                    feed_sample=batch_inputs[i][j]
                    
                    #feed-forward
                    for l in range(0,len(shape)-1):
                        if(l<len(shape)-2): #till last hidden layer
                            #the following if-else'es are just for some numpy complications
                            if(j==0):
                                z_cache[l]=np.hstack((z_cache[l],feed_relu(feed_sample, weights[l], bias[l])[0].flatten()))
                            else:
                                z_cache[l]=np.vstack((z_cache[l],feed_relu(feed_sample, weights[l], bias[l])[0]))
                              
                            
                            feed_sample=feed_relu(feed_sample, weights[l], bias[l])[1]
                            
                            
                            if(j==0):
                                a_cache[l]=np.hstack((a_cache[l],feed_sample.flatten()))
                            else:
                                a_cache[l]=np.vstack((a_cache[l],feed_sample))
                        else: #for output layer
                            
                            if(j==0):
                                z_cache[l]=np.hstack((z_cache[l],feed_num(feed_sample, weights[l], bias[l])[0].flatten()))
                            else:
                                z_cache[l]=np.vstack((z_cache[l],feed_num(feed_sample, weights[l], bias[l])[0]))
                                
                            feed_sample=feed_num(feed_sample, weights[l], bias[l])[1]
                            
                            
                            if(j==0):
                                a_cache[l]=np.hstack((a_cache[l],feed_sample.flatten()))
                            else:
                                a_cache[l]=np.vstack((a_cache[l],feed_sample))
                                
                a_cache.insert(0,np.array(batch_inputs[i]))
                z_cache.insert(0,np.array(batch_inputs[i])) #inserting inputs in the z and a cache
                
                #finding loss of the output layer
                error_cache[len(shape)-2]=output_loss(batch_y[i],a_cache[len(shape)-1]) 
            
                #backpropagating the loss till first hidden layer for each sample in the mini batch\
                #and adding the losses 
                for j in range(0,batch_size):
                    for l in range(len(shape)-3,-1,-1):
                        error_cache[l]=np.add(error_cache[l],backprop(error_cache[l+1],weights[l+1],z_cache[l+1][j]))
                
                #dividing the losses by the batch_size
                for j in range(0,len(error_cache)):
                    error_cache[j]/=batch_size
                    
                #using gradient descent for each layer    
                for l in range(len(shape)-2,-1,-1):
                    c,d=0,0
                    
                    for j in range(0,batch_size):
                        del_wb=grad_descent_calc(error_cache[l],a_cache[l][j],weights[l],bias[l])
                        c=np.add(c,del_wb[0])
                        d=np.add(d,del_wb[1]) 
                     
                    #print(c)
                    #print(d)
                    weights[l]=np.subtract(weights[l],c)
                    bias[l]=np.subtract(bias[l],d.reshape(1,-1,))
                    vdw,vdb,sdw,sdb=0,0,0,0
                
                
            
ann=Dense(0.01,10,[1,32,32,32,1],50)
ann.train(dat,y)            
            

#Generating a sin curve                
import math

data=pd.DataFrame(np.random.randn(1000,2))
for i in range(0,1000):
    data.iloc[i,0]=i
    data.iloc[i,1]=math.sin(math.radians(18*i/100))
    
data=np.array(data)
np.random.shuffle(data)           
            
y=[] 
dat=[]         

for i in range(0,1000):
    y.append(data[i,1])
    dat.append(data[i,0])
y=np.array(y).reshape(-1,1,)
dat=np.array(dat).reshape(-1,1,)
#Plot loss_cache variable to find how the network trained
 




#Experimental stuff
'''       
k=[]
yk=[]       
for i in range(0,1000):           
    pred1=feed_relu([[i]],weights[0],bias[0])[1]
    pred2=feed_relu(pred1,weights[1],bias[1])[1]
    pred3=feed_num(pred2,weights[2],bias[2])[1]
    k.append(pred3[0,0])
    yk.append(math.atan(i))
              
lc=[]            
for i in range(0,20000):
    lc.append(loss_cache[i])               
                

data=pd.read_csv('train.csv')
data=np.array(data)
        
        
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data[:,1:2]=sc.fit_transform(data[:,1:2])          
data[:,0:1]=sc.fit_transform(data[:,0:1])    
        
             
                
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 100)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 100)
dat=x.reshape(-1,1,)
y=y.reshape(-1,1,)
  
y[:,0:1]=sc.fit_transform(y[:,0:1])
dat[:,0:1]=sc.fit_transform(dat[:,0:1])                  
 '''               
                
                
        















                 
                    
            
            
        
        
        
        
        
            
        
        
        
            
                
        
        
        
        
        
        
        
        
        
    