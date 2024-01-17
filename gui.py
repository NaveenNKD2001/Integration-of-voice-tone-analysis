from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)
    return audio_path

def load_extract_audio_features(audio_path):
    audio_data, sampling_rate = librosa.load(audio_path)
    mfccs=librosa.feature.mfcc(y=audio_data,sr=sampling_rate,n_mfcc=10)
    chroma=librosa.feature.chroma_stft(y=audio_data,sr=sampling_rate)
    mel=librosa.feature.melspectrogram(y=audio_data,sr=sampling_rate)
        
    mfccs_flat=np.mean(mfccs,axis=1)
    chroma_flat=np.mean(chroma,axis=1)
    mel_flat=np.mean(mel,axis=1)
        
    audio_features=np.concatenate([mfccs_flat,chroma_flat,mel_flat])
        
    return audio_features

def EmotionDetectionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

EMOTION_LIST = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

integrated_model = EmotionDetectionModel("model_integrated_a.json", "Model_Integrated_weights.h5")

video_path='video.mp4'
audio_path='test.wav'
c = cv2.VideoCapture(video_path)

while True:
    r, frame = c.read()

    if not r:
        break

    frame = cv2.resize(frame, (1280, 720))
    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = faces.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi = gray_frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi[np.newaxis, :, :, np.newaxis]

        audio_path = extract_audio(video_path, audio_path)
        audio_features = load_extract_audio_features(audio_path)
        audio_features = audio_features.reshape((1,1,150))

        combined_pred = integrated_model.predict([audio_features, roi])

        pred = EMOTION_LIST[np.argmax(combined_pred)]

        cv2.putText(frame, pred, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

c.release()
cv2.destroyAllWindows()
