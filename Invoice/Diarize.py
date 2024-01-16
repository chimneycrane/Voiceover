import os
import sys
import torch
import pickle
from pydub import AudioSegment
from pyannote.audio import Pipeline

class Diarizer:
    def __init__(self):
        self._diary = list()
        self._audio_path=''
        self.device = "cuda" 
        if not torch.cuda.is_available():
            device = "cpu"
        
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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", )
        
        self.pipeline.to(torch.device(self.device))
                
    def Diarize(self, audio_path):
        self._diary = list() 
        diary_folder = audio_path.split(".wav")[0]
        diarization = self.pipeline(audio_path)                
        audio = AudioSegment.from_wav(audio_path)
        os.mkdir(diary_folder)   
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start
            end = turn.end
            
            self._diary.append([start, end, speaker])
            segment = audio[start:end]
            
        with open(diary_folder+'/diary.pickle', 'wb') as file:
            pickle.dump(self._diary, file, protocol=pickle.HIGHEST_PROTOCOL)

#outputs separate audio chunks of speach in wav format and csv catalog into audio_path dir (first argument)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing argument audio_path")
    else:
        diarizer = Diarizer()
        diarizer.Diarize(sys.argv[1])
        