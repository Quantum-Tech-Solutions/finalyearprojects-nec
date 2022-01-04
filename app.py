from contextlib import nullcontext
from flask import Flask, render_template
from flask.globals import request
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2

from flask import Flask, render_template,request
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2

import os
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, send_from_directory,render_template
from keras.preprocessing import image
from werkzeug.utils import secure_filename

import os
from tensorflow import keras
from flask import Flask, request, render_template, flash, redirect
import pickle
import librosa
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import tensorflow as tf

import os
from PIL import Image
from numpy import asarray
import numpy as np
#import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from flask import Flask,request,render_template
import traceback
from werkzeug.utils import secure_filename
from tensorflow.keras import backend as K

global model


app = Flask(__name__)

UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

uday_saved_model = load_model('model/uday')
uday_saved_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

sowmya_model_path ="model/sowmya/model.h5"
sowmya_model=load_model(sowmya_model_path,compile=False)

ecg_model = load_model('model/ecg/model.h5')

diabetics_model = './model/diabetics/model.h5'
saved_model = load_model(diabetics_model,compile=False)

@app.route('/uday')
def hello():
    return render_template('uday/index.html')

@app.route('/uday/form')
def form():
    return render_template('uday/form.html')

@app.route('/uday/predict',methods=['POST'])
def predict():
    # print(request.files)
    
    if 'image_file' not in request.files:
        return 'there is no image attached in the form!'
    else:
        image_file = request.files['image_file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(path)
        img = cv2.imread(path)
        img = cv2.resize(img,(200,200))
        img = np.reshape(img,[1,200,200,3])
        d = uday_saved_model.predict(img)
        print(d)
        if(d[0][0]>0.8):
            return render_template('uday/braintumour.html')
        elif(d[0][0]>0.5 and d[0][0]<0.8):
            return 'You have symptoms of brain tumour'
        else:
            return render_template('uday/nobraintumour.html')

    return nullcontext

def sowmya_model_predict(img_path, model):

    test_image=image.load_img(img_path,color_mode="grayscale",target_size=(100,100))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    res=model.predict(test_image)[0][0]
    return res


@app.route('/sowmya', methods=['GET'])
def sowmya_index():
    return render_template('sowmya/index.html')


@app.route('/sowmya/predict', methods=["GET","POST"])
def sowmya_predict():
    if request.method == 'POST':
        file = request.files['image_file']
        basepath=os.path.dirname(__file__)
        filename=secure_filename(file.filename)
        filepath=os.path.join(basepath,'upload/',file.filename)
        file.save(filepath)
        livepreds = sowmya_model_predict(filepath,sowmya_model)
        if livepreds==1:
            return render_template('sowmya/covid_negative.html',filename=filename)
        else:
            return render_template('sowmya/covid_positive.html',filename=filename)
    return None

@app.route('/sowmya/predict/<filename>')
def send_image(filename):
    return send_from_directory("upload", filename)


def speech_get_audio_features(audio_path,sampling_rate):
    X, sample_rate = librosa.load(audio_path ,res_type='kaiser_fast',duration=2.5,sr=sampling_rate*2,offset=0.5)
    sample_rate = np.array(sample_rate)

    y_harmonic, y_percussive = librosa.effects.hpss(X)
    pitches, magnitudes = librosa.core.pitch.piptrack(y=X, sr=sample_rate)

    mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=13),axis=1)

    pitches = np.trim_zeros(np.mean(pitches,axis=1))[:20]

    magnitudes = np.trim_zeros(np.mean(magnitudes,axis=1))[:20]

    C = np.mean(librosa.feature.chroma_cqt(y=y_harmonic, sr=sampling_rate),axis=1)
    
    return [mfccs, pitches, magnitudes, C]
    
    

def get_features_dataframe(dataframe, sampling_rate):
    labels = pd.DataFrame(dataframe['label'])
    
    features  = pd.DataFrame(columns=['mfcc','pitches','magnitudes','C'])
    for index, audio_path in enumerate(dataframe['path']):
        features.loc[index] = speech_get_audio_features(audio_path, sampling_rate)
    
    mfcc = features.mfcc.apply(pd.Series)
    pit = features.pitches.apply(pd.Series)
    mag = features.magnitudes.apply(pd.Series)
    C = features.C.apply(pd.Series)
    
    combined_features = pd.concat([mfcc,pit,mag,C],axis=1,ignore_index=True)

    return combined_features, labels 


speech_model=tf.keras.models.load_model('model/speech')
opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
speech_model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
emotions={0:"anger",1:"disgust",2:"fear",3:"happy",4:"neutral",5:"sad",6:"surprise"}


@app.route("/speech",methods=['GET','POST'])
def speech_index():
  return render_template('speech/index.html')


@app.route("/speech/predict", methods=['GET','POST'])
def speech_predict():
  if request.method == 'POST':
        file = request.files['file']
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
          basepath=os.path.dirname(__file__)
          filepath=os.path.join(basepath,'upload/',secure_filename(file.filename))
          if filepath[-4:]=='.wav':
              file.save(filepath)
              demo_mfcc, demo_pitch, demo_mag, demo_chrom = speech_get_audio_features(filepath,22050)
              mfcc = pd.Series(demo_mfcc)
              pit = pd.Series(demo_pitch)
              mag = pd.Series(demo_mag)
              C = pd.Series(demo_chrom)
              demo_audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)
              demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
              demo_audio_features= np.expand_dims(demo_audio_features, axis=2)
              livepreds = speech_model.predict(demo_audio_features, batch_size=64, verbose=1)
              index = livepreds.argmax(axis=1).item()
              res=emotions[index].upper()
              return render_template('speech/index.html', prediction_text="The predicted emotion is : "+str(res))
          else:
              return render_template('speech/error.html')
  return redirect(request.url)


def ecg_process_image(filename):
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print(img.shape)
    img=img[1300:1600:,120:1900]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    horizontal = th2
    rows,cols = horizontal.shape
    print(rows,cols)
    horizontalsize = cols //30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal_inv = cv2.bitwise_not(horizontal)
    masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
    masked_img_inv = cv2.bitwise_not(masked_img)
    cv2.imwrite(app.config['UPLOAD_FOLDER']+"/"+filename,masked_img_inv )
    return
    
def ecg_convert_to_array(filename):
    image=Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    data = asarray(image)
    X = np.zeros((1,300,1780))
    X[0]=data
    X = (X - X.mean())/(X.std())
    return X
def ecg_change(x): 
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer

@app.route("/ecg")
def ecg_form():
    return render_template("ecg/form.html")

@app.route('/ecg/predict',methods=['POST','GET'])
def ecg_predict():
    try:
        print(request.files)
        file=request.files['img_file']
        filename=secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        ecg_process_image(filename)
        processed_img=ecg_convert_to_array(filename)
        #with graph.as_default():
        predictions = ecg_model.predict(processed_img)
        predictions=ecg_change(predictions)
        
        if predictions[0]==0:
            return render_template('ecg/form.html', prediction='Heart Health Analysis: '+'Abnormal!')
        elif predictions[0]==1:
            return render_template('ecg/form.html', prediction='Heart Health Analysis: '+'Normal!')
    except:
        traceback.print_exc()

@app.route('/diabetics', methods=['GET'])
def diabetics_index():
    return render_template('diabetics/index.html')

@app.route('/diabetics/predict',methods=['GET','POST'])
def diabetics_predict():
    if request.method == 'POST':
        image_file = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(path)
        img = cv2.imread(path)
        img = cv2.resize(img,(64,64))
        img = np.reshape(img,[1,64,64,3])
        d = saved_model.predict(img) 
        print(d)
        r=d[0][0]
        r=round(r)-2
        if(r==0):
            result = 'No DR'
        elif(r==1):
            result = 'Mild DR'
        elif(r==2):
            result='Moderate DR'
        elif(r==3):
            result='Severe DR'
        else:
            result='Proliferative DR'
        
        return result
    else:
        return 'there is no scanned image attached'

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=False,port=5000)