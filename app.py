import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

import speech_recognition as sr
import requests
from gtts import gTTS
from janome.tokenizer import Tokenizer
import pykakasi
import pygame
import time
import random

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        #image gray
        img_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            #cv2.rectangle(img=small_frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=1)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            if output=='happy':
                cv2.putText(small_frame, 'PASS!', label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return small_frame

def record_p():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("何か話しかけてください")
        ##録音開始
        audio = r.listen(source)
        ## 文字起こし
        output_p = r.recognize_google(audio, language='ja-JP')
        #st.write(output_p)
        with open("audio_pataka.wav", "wb") as f:
            f.write(audio.get_wav_data())
        st.write("録音終了")
    
    #録音再生
    with sr.AudioFile("audio_pataka.wav") as source:
        audio = r.record(source)
    text = r.recognize_google(audio, language='ja-JP')
    t=list(text)
    t1=[]
    for i in range(len(t)):
        if t[i]=='ぱ':
            t1.append(t[i])
    st.write("評価結果")
    value=str(round(len(t1)/10,2))
    return st.write(value)
    #return output_p

def record_h():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("上の文章を音読してください")
        ##録音開始
        audio = r.listen(source)
        ## 文字起こし
        output_h = r.recognize_google(audio, language='ja-JP')
        #st.write(output_h)
        ## 音声ファイル保存
        with open("audio_hatsuwa.wav", "wb") as f:
            f.write(audio.get_wav_data())
            print("録音終了")
    score=random.random()
    if  score < 0.2:
        st.write('1')
    elif 0.2 <= score and score < 0.4:
        st.write('2')
    elif 0.4 <= score and score < 0.6:
        st.write('3')
    elif 0.6 <= score and score < 0.8:
        st.write('4')
    elif 0.8 <= score and score < 1.0:
        st.write('5')
    return output_h

def main():
    # Face Analysis Application #
    st.title("認知症予防AIアプリ")
    activiteis = ["身体機能：表情分析","口腔機能：パタカ検査","口腔機能：発話明瞭度"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ 佐藤能臣・尾松紀依による開発    
            Email : yas-sato@clinks.jp""")
    
    if choice == "身体機能：表情分析":
        st.header("表情検出")
        st.write("笑顔を作りましょう。「PASS!」が表示されれば合格です。")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
                
    elif choice == "口腔機能：パタカ検査":
        st.title("パタカ検査")
        st.subheader("10秒間「パ」を言い続けてください")
        if st.button("録音"):
            record_p()
        
    elif choice == "口腔機能：発話明瞭度":
        st.title("発話明瞭度")
        st.subheader("次の文章を音読してください")
        st.markdown("""
        ある日、北風と太陽が力くらべをしました。
        旅人の外套を脱がせたほうが勝ちということに決めてまず風からはじめました。
        風は「ようし、ひとめくりにしてやろう」とはげしくふきたてました。
        風が吹けば吹くほど旅人は外套をぴったり身体にまきつけました。
        次は、太陽の番になりました。
        太陽は、雲の間から顔を出して温かな日差しを送りました。
        旅人はだんだんよい心持ちになりとうとう外套を脱ぎすてました。
        そこで風の負けになりました。 
        """)
        if st.button("録音"):
            record_h()
            
    else:
        pass

if __name__ == "__main__":
    main()
