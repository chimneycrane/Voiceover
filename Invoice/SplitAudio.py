import os
import sys
import math
import shutil
from pydub import AudioSegment

class Viddub:
    def __init__(self) -> None:
        self.project_folder = ''
        self.vocals         = ''
        self.accompaniment  = ''

    def ExtractVoice(self):
        vocals = self.project_folder+'/vocals.wav'
        audio = AudioSegment.from_file(vocals)
        lower_sample_rate = 16000
        audio = audio.set_frame_rate(lower_sample_rate)
        audio.export(vocals, format='wav')
        return vocals, self.project_folder+'/accompaniment.wav'
        
    def dub(self, proj_dir):    
        self.project_folder =               proj_dir
        self.vocals, self.accompaniment = self.ExtractVoice()
    
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Missing arguments")
    else:
        dubber = Viddub()
        dubber.dub(sys.argv[1])
        