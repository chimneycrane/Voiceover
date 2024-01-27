import os
import sys
import math
import shutil
from pydub import AudioSegment

class Viddub:
    def __init__(self) -> None:
        self.project_folder = ''
        self.source_video   = ''
        self.source_audio   = ''
        self.vocals         = ''
        self.accompaniment  = ''

    def ExtractVoice(self):
        vocals = self.project_folder+'/vocals_sep.wav'
        audio = AudioSegment.from_file(vocals)
        lower_sample_rate = 16000
        audio = audio.set_frame_rate(lower_sample_rate)
        audio.export(vocals, format='wav')
        return vocals, self.project_folder+'/accompaniment_sep.wav'
        
    def dub(self, proj_dir, video_path, voice_left=0, voice_right=1):    
        self.source_video   =               video_path
        self.project_folder =               proj_dir
        self.source_audio   =               self.ExtractAudio(self.source_video, voice_left, voice_right)
        self.vocals, self.accompaniment = self.ExtractVoice()
    
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Missing arguments")
    else:
        dubber = Viddub()
        dubber.dub(sys.argv[1], sys.argv[2])
        