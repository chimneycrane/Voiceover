import sys
import torch
import pickle
from TTS.api import TTS
from pydub import AudioSegment
from audiostretchy.stretch import stretch_audio

class Synthesis():
    def __init__(self, work_dir, accent, background):
        self.wd = work_dir
        self.accent = accent
        self.background = background
        #voice gen with TTS and xtts-v2 model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.tts.to(device)
        self.transcript = []
        with open(work_dir+'/transcript.pickle', 'rb') as file:
            self.transcript = pickle.load(file)
        
    def _squeeze_audio(self, audio_path, start_time, end_time):
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)
        length_ms = audio.duration_seconds
        desired_length = end_time-start_time
        speed_factor = desired_length/length_ms
        stretch_audio(audio_path, audio_path.replace('.wav','r.wav'), ratio=speed_factor)
        
    def Glue(self, result_path):
        output = AudioSegment.from_file(self.background)
        for record in self.transcript:
            if record[3]!='':
                audio = AudioSegment.from_file(record[5])
                resampled_audio = audio.set_frame_rate(output.frame_rate)
                output = output.overlay(audio, position=int(record[0]*1000))
        output.export(result_path, format='wav')                
        
    def Synthesize(self):
        i=0
        for record in self.transcript:
            if record[3]!='':
                if record[6]==1:
                    self.tts.tts_to_file(text=record[3].replace(';','.').replace('.',' .'),
                        file_path=record[5],
                        speaker_wav=record[4], temperature=0.7,
                        language=self.accent)
                if record[6]==1:
                    output = self._squeeze_audio(record[5],record[0],record[1])
                    record[5]= record[5].replace('.wav','r.wav')
            i+=1
        self.Glue(self.wd+'/result.wav')

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Missing argument audio_path")
    else:
        synth = Synthesis(sys.argv[1], sys.argv[2], sys.argv[3])
        synth.Synthesize()