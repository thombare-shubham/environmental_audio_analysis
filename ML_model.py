from keras.layers import Dense,Dropout,Flatten
from numpy.core.fromnumeric import size
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.utils import np_utils
import noisereduce as nr
import numpy as np
import librosa
import os

# Declarations
activities = ['CanOpening', 'CarHorn', 'Cat', 'ChirpingBirds', 'Clapping', 'ClockAlarm', 'Cow', 'CracklingFire', 'Crow', 'CryingBaby', 'Dog', 'Door_or_WoodCreaks', 'Engine', 'Fireworks', 'GlassBreaking', 'HandSaw', 'Helicopter', 'Hen', 'Laughing', 'Night', 'Pig', 'Rain', 'Rooster', 'SeaWaves', 'Siren', 'Snoring', 'Thunderstorm', 'Train', 'TwoWheeler', 'VaccumCleaner', 'WaterDrops', 'Wind']

subjects = ['s01', 's02', 's03', 's04', 's05']

train_subjects = ['s01', 's02','s03','s04']
test_subjects = ['s05']

chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-" 
charsLen = len(chars)

# Replicate LableEncoder
# LabelEncoder is used for normalizing label values
lb = LabelEncoder()

# EXTRACT AND SAVE STFT FEATURES
def save_STFT(file, name, activity, subject):
    # read audio data
    audio_data, sample_rate = librosa.load(file)
    noisy_part = audio_data[0:25000]  
    reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)
    
    #trimming
    trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)
    
    # extract features
    stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512)) 
    # save features
    np.save("Features/" + subject + "_" + name[:-4] + "_" + activity + ".npy", stft)

# Feature Extraction
def Feature_Extraction():
    for activity in activities:
        for subject in subjects:
            innerDir = subject + "/" + activity
            for file in os.listdir("Dataset_audio/"+innerDir+"/"):
                if(file.endswith(".wav")):
                    save_STFT("Dataset_audio/"+innerDir+"/" + file, file, activity, subject)
                    print("Extracting feature from "+subject+"-"+file+"-"+activity)

# MODEL BUILDING

# CONVERT STRING TO NUMBER
def strToNumber(numStr):
  num = 0
  for i, c in enumerate(reversed(numStr)):
    num += chars.index(c) * (charsLen ** i)
  return(num)

# DIFFERENTIATE DATA IN TRAIN,TEST MODULE
def get_data(path):

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
        
    for file in os.listdir(path ):
        if int(strToNumber(file.split("_")[1].split("_")[0]))!=1:
          a = (np.load(path + file)).T
          label = file.split('_')[-1].split(".")[0]
          if(label in activities):
              #if(a.shape[0]>100 and a.shape[0]<500):
                if file.split("_")[0] in train_subjects:
                  X_train.append(np.mean(a,axis=0))
                  Y_train.append(label)
                else:
                  X_test.append(np.mean(a,axis=0))
                  Y_test.append(label)
                  
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    return X_train,Y_train,X_test,Y_test

# BUILD AND SAVE MODEL
def Build_and_save_model():
    X_train,Y_train,X_test,Y_test = get_data("Features/")

    n_samples = len(Y_train)
    print("Sample Count","No of samples to train: "+str(n_samples))

    order = np.array(range(n_samples))
    np.random.shuffle(order)
    X_train = X_train[order]
    Y_train = Y_train[order]

    y_train = np_utils.to_categorical(lb.fit_transform(Y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(Y_test))
    num_labels = y_train.shape[1]

    # BUILD MODEL
    model = Sequential()

    model.add(Dense(256, input_shape=(257,), activation='relu'))
    # model.add(Dropout(0.5))#Function used for regularization. regularization is the process which regularizes or shrinks the coefficients towards zero which reduces overfitting.

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64,activation='relu'))
    model.add(Flatten())

    model.add(Dense(num_labels,activation='softmax'))

    model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')

    model.fit(X_train, y_train, epochs=200,validation_data=(X_test,y_test))

    # Calculate Accuracy
    accuracy = model.evaluate(X_train, y_train, verbose=1)

    # save model (optional)
    model_json = model.to_json()
    with open("audio_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("audio_model.h5")

    print("Model is ready! Accuracy of model is "+str(accuracy[1]*100)+"%")

# Function Calls
Feature_Extraction()
Build_and_save_model()