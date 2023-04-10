import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import  Input, Dense, Dropout,Activation, Flatten, Embedding, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint 
from sklearn.preprocessing import LabelEncoder
import tqdm
import numpy as np
# !pip install librosa
import librosa
import os
import zipfile
import matplotlib.pyplot as plt
import IPython
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ### Perform EDA ###


# path=r'C:\Users\TanishSharma\OneDrive - TheMathCompany Private Limited\Desktop\Audio Classifications\UrbanSound8K\audio\fold1/7061-6-0-0.wav'

path=r'C:\Users\your_name\downloads\Audio Classifications\UrbanSound8K\audio\fold1/7061-6-0-0.wav'
# plot wave of this given sound##
plt.figure(figsize=(14,5))
data, sample_rate=librosa.load(path)
librosa.display.waveshow(data,sr=sample_rate)
# Play wave
# wave.play()

IPython.display.Audio(path)
## this gun shot sound ##

## explore the meta data given
path_e=r'C:\Users\TanishSharma\OneDrive - TheMathCompany Private Limited\Desktop\Audio Classifications\UrbanSound8K\audio/'
meta_data=pd.read_csv(r'C:\Users\TanishSharma\OneDrive - TheMathCompany Private Limited\Desktop\Audio Classifications\UrbanSound8K\metadata/UrbanSound8K.csv')
## having sounds of different class like dog bark children playing with their class id.
# we can create whole path here in the metadata
meta_data['file_paths'] = meta_data.apply(lambda row: path_e+'fold'+str(row['fold']) +'/' +row['slice_file_name'], axis=1)
df = pd.DataFrame(zip(meta_data['class'].unique(),meta_data['classID'].unique()), columns=['Class_ID', 'Category'])
with open('Categories.pickle', 'wb') as f:
    # use the pickle.dump method to write the data to the file
    pickle.dump(df, f)
## let's explore the dat distribution of class id
meta_data['classID'].value_counts()
## There are no signs of skewed or biased data in the distributions, 
# #which appear to be reasonable.

# Feature extraction using mel fcc


# mfccs=librosa.feature.mfcc(y=data,sr=sample_rate,n_mfcc=40)
file_paths = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(path_e) for file in files]

# ########## or alternative way###########
# file_paths=[]
# for subdir, dirs, files in os.walk(path_e):
#     for file in files:
#         file_paths.append(os.path.join(subdir, file))
# load audio data and sample rates for each file
# data_and_sample_rates = [(librosa.load(i)) for i in file_paths if i.endswith('.wav')]

# # unpack the data and sample rates into separate lists
# data, sample_rates = zip(*data_and_sample_rates)
# All_mfccs=[librosa.feature.mfcc(y=data,sr=sample_rate,n_mfcc=40) for i in file_paths]


def extract_feature(file):
    data, sample_rates=librosa.load(file)
    mfcc_features=librosa.feature.mfcc(y=data,sr=sample_rate,n_mfcc=40)
    mfcc_scaled_feature=np.mean(mfcc_features.T,axis=0)
    return mfcc_scaled_feature
extract_features=[]
for i in tqdm.tqdm(meta_data['file_paths']):
    extract_features.append(extract_feature(i))
# ########## Saving list of audio features into pickle
with open('extract_features.pickle', 'wb') as f:
    # use the pickle.dump method to write the data to the file
    pickle.dump(extract_features, f)
################################################3


Final_df=pd.DataFrame({'features':extract_features,'class':meta_data['class']})
X=np.array(Final_df['features'].tolist())
# X.shape (8732, 40), y.shape (8732,)
y=np.array(Final_df['class'].tolist())

lb=LabelEncoder()
y=to_categorical(lb.fit_transform(y))

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# y_test.shape (1747,10)
# X_train.shape (6985, 40)

# ###################### MODEL TRAINING#########################


def model_init(x, neuron):
    model = Sequential()
    model.add(Dense(neuron, input_shape= (x.shape[1], ) , activation=tf.keras.activations.relu))
    model.add(Dropout(0.5))
    # 2nd layer
    model.add(Dense(200,activation=tf.keras.activations.relu))
    model.add(Dropout(0.5))
    # 3rd layer
    model.add(Dense(200,activation=tf.keras.activations.relu))
    model.add(Dropout(0.5))
    #final layer
    model.add(Dense(y.shape[1],activation=tf.keras.activations.softmax))
    
    return model
ANN_Model = model_init(X_train, 600)

print(ANN_Model.summary())

ANN_Model.compile(loss='categorical_crossentropy',metrics=['accuracy','Recall','Precision'],optimizer='Adam')

val_ds = (X_test, y_test)
# checkpointer=ModelCheckpoint(filepath=r'C:\Users\TanishSharma\OneDrive - TheMathCompany Private Limited\Desktop\Audio Classifications/saved_models/audion_class.hdf5',verbose=1,save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-05, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5, verbose=1, mode='auto')

history = ANN_Model.fit(X_train, y_train, validation_data=val_ds, epochs=100, callbacks=[reduce_lr,early_stopping], verbose=1, batch_size=34)
print(history)
with open('ANN_Model.pickle', 'wb') as f:
    # use the pickle.dump method to write the data to the file
    pickle.dump(ANN_Model, f)

def evaluate_model(ANN_Model):
# evaluate the model on the test set
    test_loss, test_acc, test_recall, test_precision = ANN_Model.evaluate(X_test, y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
    print('Test precision:', test_precision)
    print('Test recall:', test_recall)
###################################### f1 score###############
    y_pred = ANN_Model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1) # convert probabilities to class labels
    y_true = np.argmax(y_test, axis=1)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print("F1 score:", f1)
# Calling evaluate_model
evaluate_model(ANN_Model)

def plot_loss(history):

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
# Calling plot_loss
plot_loss(history)



