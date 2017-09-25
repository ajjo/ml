import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

x = np.zeros(shape=(101,16))
y = np.zeros(shape=(101,7))

with open('zoo.csv') as csvfile:
     reader = csv.DictReader(csvfile)

     for i,row in enumerate(reader):
            x[i] = [row['hair'], row['feathers'], row['eggs'], row['milk'], row['airborne'], 
                    row['aquatic'], row['predator'], row['toothed'], row['backbone'],
                    row['breathes'], row['venomous'], row['fins'], row['legs'], row['tail'],
                    row['domestic'], row['catsize']]           

            class_type = int(row['class_type'])
            
            for j in range(7):
                if (j+1 == class_type):
                    y[i,j] = 1
                else:
                    y[i,j] = 0

layer1_output_dim = 150
layer2_output_dim = 150
num_outputs = 7
epoch_count = 150
activation_function='sigmoid'

# model.add(Dropout(0.1))

adam = Adam()
model = Sequential()
model.add(Dense(layer1_output_dim,input_shape=(16,),init='uniform',activation=activation_function))
model.add(Dense(layer2_output_dim,init='uniform',activation=activation_function))
model.add(Dense(num_outputs,init='uniform',activation=activation_function))
model.compile(optimizer=adam,loss='mse',metrics=['accuracy'])
history = model.fit(x,y,nb_epoch=epoch_count)#,validation_split=0.8)

y_pred = np.array([ [1,0,0,1,1,0,0,1,1,1,0,0,2,1,0,0], #1
                    [1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,0], #1
                    [0,1,1,0,1,0,1,0,1,1,0,0,2,1,0,1], #2
                    [1,0,0,1,0,0,0,1,1,1,0,0,2,1,0,1], #1
                    [1,0,1,0,1,0,0,0,0,1,1,0,6,0,0,0], #6
                    [1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1], #1
                    [0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0], #7
                    [0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,0]])#2

n = model.predict(y_pred)
n = np.around(n)
print(n)               

            

