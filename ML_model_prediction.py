from tkinter import filedialog,messagebox
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import numpy as np
import librosa
import noisereduce as nr

# Definations
lb = LabelEncoder()

activities = ['CanOpening', 'CarHorn', 'Cat', 'ChirpingBirds', 'Clapping', 'ClockAlarm', 'Cow', 'CracklingFire', 'Crow', 'CryingBaby', 'Dog', 'Door_or_WoodCreaks', 'Engine', 'Fireworks', 'GlassBreaking', 'HandSaw', 'Helicopter', 'Hen', 'Laughing', 'Night', 'Pig', 'Rain', 'Rooster', 'SeaWaves', 'Siren', 'Snoring', 'Thunderstorm', 'Train', 'TwoWheeler', 'VaccumCleaner', 'WaterDrops', 'Wind']

# PREDICT SOUND
def predictSound(X):
    model_name = "audio_model"

    # Model reconstruction from a json file
    with open('audio_model.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights('audio_model.h5')

    # using fit_transform
    lb.fit_transform((activities))
    # Returns magnitude of frequency bin f at frame t
    stfts = np.abs(librosa.stft(X, n_fft=512, hop_length=256, win_length=512))
    stfts = np.mean(stfts, axis=1)
    result = model.predict(np.array([stfts]))  # Predict Output of model
    predictions = [np.argmax(y) for y in result]
    return lb.inverse_transform([predictions[0]])[0]

# FUNCTION TO RUN PREDICTION
def run_prediction():
# Import file using tkinter
    filename = filedialog.askopenfilename(
       initialdir="/", filetypes=(("Audio files", "*.wav"), ("all files", "*.*")))

    if filename.endswith('.wav'):

        raw_audio, sr = librosa.load(filename)
        noisy_part = raw_audio[0:25000]

        nr_audio = nr.reduce_noise(audio_clip=raw_audio, noise_clip=noisy_part, verbose=False)

        clip, index = librosa.effects.trim(nr_audio, top_db=20, frame_length=512, hop_length=64)
        result = predictSound(clip)
        res_prediction = str(result)
        if res_prediction in ("CanOpening" , "CarHorn" , "Clapping" , "ClockAlarm" , "Cow" , "Crow" , "CryingBaby" , "Dog" , "Engine" , "Fireworks" , "GlassBreaking" , "HandSaw" , "Helicopter" , "Laughing" , "Siren" , "Snoring" , "Thunderstorm" , "Train" , "TwoWheeler" , "VaccumCleaner"):
            messagebox.showwarning("UnSafe Environment","It's sound of a "+result+" and it's hazardous for children!")
        
        else:
            messagebox.showinfo("Safe Environment","It's sound of a"+result+" & Its safe for children!")
        
        del result

    else:
        messagebox.showinfo("Error","Wrong file selected/No file Selected")

run_prediction()