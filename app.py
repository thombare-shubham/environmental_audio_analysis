from flask import Flask, app,request,render_template
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import numpy as np
import librosa
import noisereduce as nr

app = Flask(__name__)

# REBUILD MODEL
model_name = "audio_model"

# Model reconstruction from a json file
with open('audio_model.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('audio_model.h5')

# DEFINATIONS
lb = LabelEncoder()

activities = ['CanOpening', 'CarHorn', 'Cat', 'ChirpingBirds', 'Clapping', 'ClockAlarm', 'Cow', 'CracklingFire', 'Crow', 'CryingBaby', 'Dog', 'Door_or_WoodCreaks', 'Engine', 'Fireworks', 'GlassBreaking', 'HandSaw', 'Helicopter', 'Hen', 'Laughing', 'Night', 'Pig', 'Rain', 'Rooster', 'SeaWaves', 'Siren', 'Snoring', 'Thunderstorm', 'Train', 'TwoWheeler', 'VaccumCleaner', 'WaterDrops', 'Wind']

# PREDICT SOUND
def predictSound(X):
    # using fit_transform
    lb.fit_transform((activities))
    # Returns magnitude of frequency bin f at frame t
    stfts = np.abs(librosa.stft(X, n_fft=512, hop_length=256, win_length=512))
    stfts = np.mean(stfts, axis=1)
    result = model.predict(np.array([stfts]))  # Predict Output of model
    predictions = [np.argmax(y) for y in result]
    return lb.inverse_transform([predictions[0]])[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    ''' For rendering results on HTML GUI'''
    pred_file = request.files['file']
    if(request.method == 'POST'):   
        raw_audio, sr = librosa.load(pred_file)
        noisy_part = raw_audio[0:25000]

        nr_audio = nr.reduce_noise(audio_clip=raw_audio, noise_clip=noisy_part, verbose=False)

        clip, index = librosa.effects.trim(nr_audio, top_db=20, frame_length=512, hop_length=64)
        result = predictSound(clip)
        res_prediction = str(result)
        if res_prediction in ("CanOpening" , "CarHorn" , "Clapping" , "ClockAlarm" , "Cow" , "Crow" ,"CryingBaby"   , "Dog" , "Engine" , "Fireworks" , "GlassBreaking" , "HandSaw" , "Helicopter" ,"Laughing" , "Siren" ,     "Snoring" , "Thunderstorm" , "Train" , "TwoWheeler" , "VaccumCleaner"):
            return render_template('index.html',prediction_text = "It's sound of a $ {} and it's disturbing for  children!".format(result))
    
        else:
            return render_template('index.html',prediction_text = "It's sound of a $ {} and it's safe for  children!".format(result))
    
    else:
        return render_template('index.html',prediction_text = 'Please upload audio file in .wav format')

if __name__ == "__main__":
    app.run(debug=True, threaded = True)