import sys
import torch
import pickle
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
from deep_translator import GoogleTranslator
import librosa
import soundfile as sf
import pickle
import math
import whisper
#from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, work_dir, audio_path, src_lang, dst_lang):
        
        self.wd = work_dir
        self.audio_path = audio_path
        self.src_lang, self.dst_lang = src_lang, dst_lang
        self.diary = []
        with open(work_dir+'/diary.pickle', 'rb') as file:
            self.diary = pickle.load(file)
        
        #Whisper pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    #Here we are matching transcript data with speaker diarization
    #Some voice generation models are limited on the size of prompt
    #So TODO will be more elaborate way of sentense separation
    #i.e. break down diary record into chunks using timestamps from transcript where each chunk size>200 chars
    def _resolveDiary(self, time, text):
        for rec in self.diary:
            if time>rec[0] and time<rec[1]:
                return rec
            
    def _FitTranscript(self, chunks):
        transcription = []
        if len(chunks)>0:
            for rec in self.diary:
                rec.append('')
            
            words = dict()
            seconds = dict()    
            for chunk in self.diary:
                words[chunk[2]]=0
                seconds[chunk[2]] =0
            #match speaker
            for chunk in chunks:
                avg_time = (chunk['end']+chunk['start'])/2
                speaker = next(filter(lambda x: x[0]<avg_time and x[1]>avg_time or x[0]>chunk['start'], self.diary), '')
                transcription.append([chunk['start'],chunk['end'],speaker[2],chunk['text']]) 
                if len(chunk['text'])>0:
                    words[speaker[2]]+=len(chunk['text'])
                    seconds[speaker[2]] += chunk['end']-chunk['start']
            wps = dict()
            for key, _ in words.items():
                wps[key] = words[key]/seconds[key]
    
            #stich sentense
            i=0
            while i < len(transcription)-1:
                text = transcription[i][3].strip()
                if text=='':
                    transcription.pop(i)
                    i-=1
                i+=1
            i=0

            while i < len(transcription)-1:
                cur_speaker = transcription[i][2]
                nxt_speaker = transcription[i+1][2]
                text = transcription[i][3].strip()
                nxt_text = transcription[i+1][3].strip()
                cur_wps = len(text)/(transcription[i][1]-transcription[i][0])
                speed_div = wps[cur_speaker]/cur_wps
                nxt_wps = len(nxt_text)/(transcription[i+1][1]-transcription[i+1][0])
                nxt_speed_div = wps[nxt_speaker]/nxt_wps
                merge = False
                pause = transcription[i+1][0]-transcription[i][1]
                if pause<1:
                    merge = speed_div>1.3 and nxt_speed_div<0.9 or pause<0.3
  
                length = len(text)+len(nxt_text)+1
                if (not(text.endswith('.') or text.endswith('?') or text.endswith('!')) or merge) and cur_speaker==nxt_speaker and not length>=5000:
                    text += ' '+nxt_text
                    transcription[i][3] = text.replace('.',' .')
                    transcription[i][1] = transcription[i+1][1]
                    transcription.pop(i+1)
                    i-=1
                i+=1
            self.diary = transcription
            
    def Transcribe(self):
        model = whisper.load_model("small")
        transcript = model.transcribe(
            word_timestamps=True,
            audio=self.audio_path
        )
        transcript['segments'][0]['start']=self.diary[0][0]
        self._FitTranscript(transcript['segments'])
        audio = AudioSegment.from_file(self.audio_path)
        for rec in self.diary:
            speaker = AudioSegment.silent(0,audio.frame_rate)
            speaker.export(self.wd+f'/{rec[2]}.wav', format='wav')
        i=0    
        for rec in self.diary:
            start = rec[0]
            end = rec[1]
            text = rec[3]
            speaker = rec[2]
            rec[3] = GoogleTranslator(source=self.src_lang, target=self.dst_lang).translate(text)
            
            referense_segment = audio[int(start*1000):int(end*1000)]
            if end-start<4:#short segments dont give good speaker referance
                speaker_path = self.wd+f'/{speaker}.wav'
                speaker_aud = AudioSegment.from_file(speaker_path)
                speaker_aud+=referense_segment
                speaker_aud.export(speaker_path, format="wav")
                rec.append(speaker_path)
            else:
                referense_segment.export(self.wd+f'/{i}.wav')
                rec.append(self.wd+f'/{i}.wav')
            i+=1            
        with open(self.wd+'/transcript.pickle', 'wb') as file:
            pickle.dump(self.diary, file, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Missing argument audio_path")
    else:
        transcriber = Transcriber(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        transcriber.Transcribe()