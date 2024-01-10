import os
import math
import langid
import numpy as np
import soundfile as sf
#import speech_recognition
import diarization
from pydub import AudioSegment
from spleeter.separator import Separator
from moviepy.editor import VideoFileClip
from pyAudioAnalysis import audioBasicIO
import torch

# Get the current process ID
pid = os.getpid()

def GetAudio(video_path, left_channel=0, right_channel=1):
    video = VideoFileClip(video_path)
    audio = video.audio
    tmp_audio_file = f"temp_audio{pid}.wav"
    audio.write_audiofile(tmp_audio_file)
    audio_data, sample_rate = sf.read(tmp_audio_file)
    channel_1 = audio_data[:, left_channel]
    channel_2 = audio_data[:, right_channel]
    stereo_audio_data = np.column_stack((channel_1, channel_2))
    stereo_audio_file = f"output_stereo{pid}.wav"
    sf.write(stereo_audio_file, stereo_audio_data, sample_rate)
    os.remove(tmp_audio_file)
    return os.getcwd()+"\\"+stereo_audio_file

#TODO performance optimization
def separate_audio(audio_path):
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(audio_path,os.getcwd())
    vocals = audio_path.split(".wav")[0]+"\\vocals.wav"
    accompaniment = audio_path.split(".wav")[0]+"\\accompaniment.wav"
    return vocals, accompaniment

def SplitBy10MbOverlapped(audio_path):
    audio = AudioSegment.from_mp3(audio_path)
    file_size = len(audio.raw_data)
    res = list()
    duration_ms = math.ceil(float(audio.duration_seconds) * 1000.0)
    if duration_ms<=0:
        return res, 0
    bytes_per_ms = math.floor(file_size/duration_ms)
    chunk_size_ms = math.floor(9*1024*1024 / bytes_per_ms)
    overlap_duration_ms = 10 * 1000
    start_time = 0
    end_time = 0
    while end_time<duration_ms:
        end_time = start_time + chunk_size_ms
        if end_time > duration_ms:
            end_time = duration_ms
        chunk = audio[start_time:end_time]
        res.append(f"chunk_{pid}_{start_time}.wav")
        chunk.export(f"chunk_{pid}_{start_time}.wav", format="wav")
        start_time = end_time-overlap_duration_ms
    return res, chunk_size_ms 
        
#def detect_language(file_path):
#    recognizer = speech_recognition.Recognizer()
#    # Load the audio file
#    with speech_recognition.AudioFile(file_path) as source:
#        audio = recognizer.record(source)
#    # Convert audio to text
#    text = recognizer.recognize_google(audio)    
#    # Detect language using langid.py
#    langid_result = langid.classify(text)
#    detected_language = langid_result[0]
#    return detected_language

#def identify_voice_actors(audio_file):
#    r = speech_recognition.Recognizer()
#    with speech_recognition.AudioFile(audio_file) as source:
#        audio = r.record(source)
#        text = r.recognize_google(audio)
#    language = langid.classify(text)[0]
#    audio_segments = []
#    for segment in audio_file:
#        audio_segments.append(AudioSegment.from_wav(segment))
#    unique_actors = []
#    timestamps = []
#    for segment in audio_segments:
#        actor = recognize_voice_actor(segment)
#        if actor not in unique_actors:
#            unique_actors.append(actor)
#            timestamps.append((segment.start_frame, segment.end_frame))
#    
#    return unique_actors, timestamps

def main(video_path, voice_only=False, voice_left=0, voice_right=1):    
    # Specify your input video file path
    video_path = "video.mp4"
    #might be specific audio channels specified either for language (or voice TODO)
    audio_path = GetAudio(video_path, voice_left, voice_right)
    vocals,accompaniment = separate_audio(audio_path)
    res = diarization.Diarize(vocals)
    diarization.Synthesize(res)

    #print(language)
    #for c in chunks:
    #    os.remove(c)
    os.remove(audio_path)
    
if __name__ == "__main__":
    main()