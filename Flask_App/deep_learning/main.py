#%%
#importing all necessary libraries
from tensorflow.keras.models import Sequential    #For initialising the model
from tensorflow.keras.layers import Conv2D        #For adding convolutional layer
from tensorflow.keras.layers import MaxPooling2D  #For adding max pooling layer
from tensorflow.keras.layers import Flatten       #For flattening max pooled layer values into a single vector
from tensorflow.keras.layers import Dense         #For adding layers to NN

import glob             #for accessing all the images
import numpy as np      #for handling the images as numpy arrays 
from PIL import Image   #for resizing the images

from sklearn import preprocessing, model_selection as ms  #for splitting data into Training, Cross - Validating, and Testing parts
from tensorflow.keras.preprocessing.image import ImageDataGenerator  #for image augmentation
import h5py                                               #for saving the model
from tensorflow.keras.models import load_model                       #for loading the model

import matplotlib.pyplot as plt  #for plotting training and cross validation accuracies vs epochs

#%%
def cnn_model(h_layers, features, neurons):        #returns the model with desired parameters
    model = Sequential() #initialise the model

    model.add( Conv2D( features, (3, 3),input_shape = (64, 64, 3), activation = 'relu' )) #Dims of feature map = 3*3, accepting 64*64 pixels grayscale images
    model.add( MaxPooling2D( pool_size = (2, 2) )) #add max pooling layer, with dims of each pool = 2*2
    model.add( Flatten() ) #add flattening layer 
    
    for i in range( h_layers ):  #add all hidden layers
        model.add( Dense( units = neurons, activation = 'relu' ))
        
    model.add( Dense( units = 1, activation = 'sigmoid' ))  #add an output layer

    model.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )  #define optimizer and loss functions as well as required metrics

    return model

#%%
h_layers = 1   #no. of hidden layers
features = 32  #no. of feature maps 
neurons = 128  #no. of neurons in each hidden layer

X_data = [] #list for holding all images as numpy arrays
y_data = [] #list for holding the labels corresponding to the respective images

#%%
PosImages = glob.glob(r'Parasitized\*.*') #path to all positive(infected) images
NegImages = glob.glob(r'Uninfected\*.*')  #path to all negative(uinfected) images

for file in PosImages:                    #resizing all positive images and converting them to numpy arrays
    img = Image.open(file)
    img_resized = img.resize((64, 64))
    img_array = np.asarray(img_resized)
    
    X_data.append(img_array)
    y_data.append(1)

for file in NegImages:                    #resizing all negative images and converting them to numpy arrays
    img = Image.open(file)
    img_resized = img.resize((64, 64))
    img_array = np.asarray(img_resized)
    
    X_data.append(img_array)
    y_data.append(0)
    
#%%
#converting the lists to numpy arrays(for flow method)
X_data = np.array(X_data)  
y_data = np.array(y_data)

m = X_data.shape[0]  #no. of samples

rand_idx = np.arange(m)     #generating indices
np.random.shuffle(rand_idx) #randomising indices

#randomly shuffling all the positive and negative samples
X_data = X_data[rand_idx]   
y_data = y_data[rand_idx]

#saving all the sample data
np.save(r'Data\X_data.npy', X_data)
np.save(r'Data\y_data.npy', y_data)

#loading the sample data
X_data = np.load(r'Data\X_data.npy')
y_data = np.load(r'Data\y_data.npy')

#%%

#creating training, cross-validation, and testing sets
X_train, X_new, y_train, y_new = ms.train_test_split(X_data, y_data, test_size = 0.7, random_state = 0)
X_crossval, X_test, y_crossval, y_test = ms.train_test_split(X_new, y_new, test_size = 0.3, random_state = 0)

val_size = X_crossval.shape[0] #cross-validation set size

#%%
#Augmenting the training images
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 20,
                                   horizontal_flip = True,
                                   vertical_flip = True,)

test_datagen = ImageDataGenerator(rescale=1./255)

#%%
#creating training, cross-validation and testing generators
train_generator = train_datagen.flow(X_train,
                                     y_train,
                                     batch_size = 32)

val_generator = test_datagen.flow(X_crossval,
                                  y_crossval,
                                  batch_size = 16)

test_generator = test_datagen.flow(X_test,
                                   y_test,
                                   batch_size = 1)

#%%
#creating model
model = cnn_model(h_layers, features, neurons)

#training the model
history = model.fit_generator(train_generator,
                              steps_per_epoch = len(X_train) / 32,
                              epochs = 20,
                              validation_data = val_generator,
                              validation_steps = val_size / 16)

#saving the model
model.save('cnn_model.h5')

#loading the model
model = load_model('cnn_model.h5')

#%%
#obtaining accuracy on test set 
test_acc = model.evaluate_generator(test_generator, steps = len(test_generator))

print(model.metrics_names)
print('Test Accuracy Obtained: ')
print(test_acc[1] * 100, ' %')

#%%
#Plotting Training and Testing accuracies
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='best')
plt.show()