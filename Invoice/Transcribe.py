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

class Transcriber:
    def __init__(self, work_dir, audio_path, src_lang, dst_lang):
        
        self.wd = work_dir
        self.audio_path = audio_path
        self.src_lang, self.dst_lang = src_lang, dst_lang
        self.diary = []
        with open(work_dir+'\\diary.pickle', 'rb') as file:
            self.diary = pickle.load(file)
        
        #Whisper pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16
        model_id = "openai/whisper-small"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        self.whisper = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
    #Here we are matching transcript data with speaker diarization
    #Some voice generation models are limited on the size of prompt
    #So TODO will be more elaborate way of sentense separation
    #i.e. break down diary record into chunks using timestamps from transcript where each chunk size>200 chars
    def _resolveDiary(self, time, text):
        for rec in self.diary:
            if time>rec[0] and time<rec[1]:
                rec[3]+=text
                break
            
    def _FitTranscript(self, chunks):
        if len(chunks)>0:
            for rec in self.diary:
                rec.append('')
                
            #match speaker
            for chunk in chunks:
                avg_time = (chunk['timestamp'][1]+chunk['timestamp'][0])/2
                self._resolveDiary(avg_time,chunk['text'])
                
            #stich sentense
            i=0
            while i < len(self.diary)-1:
                text = self.diary[i][3].strip()
                if text=='':
                    self.diary.pop(i)
                    i-=1
                i+=1
            i=0
            while i < len(self.diary)-1:
                cur_speaker = self.diary[i][2]
                nxt_speaker = self.diary[i+1][2]
                text = self.diary[i][3].strip()
                nxt_text = self.diary[i+1][3].strip()
                if not(text.endswith('.') or text.endswith('?') or text.endswith('!')) and cur_speaker==nxt_speaker:
                    text += ' '+nxt_text
                    self.diary[i][3] = text
                    self.diary[i][1] = self.diary[i+1][1]
                    self.diary.pop(i+1)
                    i-=1
                i+=1
                
    def Transcribe(self):
        audio = AudioSegment.from_wav(self.audio_path)
        transcription = self.whisper(self.audio_path)
        
        self._FitTranscript(transcription['chunks'])
        for rec in self.diary:
            speaker = AudioSegment.silent(0,audio.frame_rate)
            speaker.export(self.wd+f'\\{rec[2]}.wav', format='wav')
        i=0    
        for rec in self.diary:
            start = rec[0]
            end = rec[1]
            text = rec[3]
            speaker = rec[2]
            rec[3] = GoogleTranslator(source=self.src_lang, target=self.dst_lang).translate(text)
            
            referense_segment = audio[int(start*1000):int(end*1000)]
            if end-start<10:#short segments dont give good speaker referance
                speaker_path = self.wd+f'\\{speaker}.wav'
                speaker_aud = AudioSegment.from_file(speaker_path)
                speaker_aud+=referense_segment
                speaker_aud.export(speaker_path, format="wav")
                rec.append(speaker_path)
            else:
                referense_segment.export(self.wd+f'\\{i}.wav')
                rec.append(self.wd+f'\\{i}.wav')
            i+=1            
        with open(self.wd+'\\transcript.pickle', 'wb') as file:
            pickle.dump(self.diary, file, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Missing argument audio_path")
    else:
        transcriber = Transcriber(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        transcriber.Transcribe()