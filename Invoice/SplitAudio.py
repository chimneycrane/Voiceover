import os
import sys
import math
import shutil
import soundfile as sf
from pydub import AudioSegment
from spleeter.separator import Separator
from moviepy.editor import VideoFileClip

class Viddub:
    def __init__(self) -> None:
        self.project_folder = ''
        self.source_video   = ''
        self.source_audio   = ''
        self.vocals         = ''
        self.accompaniment  = ''

    def ExtractAudio(self, video_path, left_channel=0, right_channel=1):
        video = VideoFileClip(video_path)
        audio = video.audio
        stereo_audio_file = self.project_folder+f"/intermitent_stereo.wav"
        audio.write_audiofile(stereo_audio_file)
        return stereo_audio_file

    def _glue_files(self, output_folder, wildcard):
        audio         =     AudioSegment.silent(duration=0)
        wav_files     =     [f for f in os.listdir(output_folder) if (f.endswith(".wav") and wildcard in f and wildcard+'_sep.' not in f)]
        if len(wav_files)>0:
            wav_files.sort(key=lambda x: int(x.split('.')[0].split('_')[len(x.split('.')[0].split('_'))-1]))
            for wav_file in wav_files:
                segment   =     AudioSegment.from_wav(os.path.join(output_folder, wav_file))
                audio    +=     segment
            audio.export(output_folder+'/'+wildcard+f"_sep.wav", format="wav")
            for wav_file in wav_files:
                os.remove(os.path.join(output_folder, wav_file))
        return output_folder+'/'+wildcard+f"_sep.wav"
    
    def ExtractVoice(self):
        separator = Separator('spleeter:2stems')
        output_folder =     self.source_audio.split(".wav")[0]
        audio, sr         =     sf.read(self.source_audio)
        duration      =     len(audio)
        frames = (sr*120)
        num_segments  =     math.ceil(duration/frames)
        for i in range(num_segments):
            start      =    i * frames
            end        =    min((i + 1) * frames, duration)
            segment    =    audio[start:end]
            path = output_folder+f"_{i}"
            name = path+".wav"
            sf.write(name, segment, samplerate=sr)
            separator.separate_to_file(name, self.project_folder, synchronous=False)
            
        separator.join()
        for i in range(num_segments):
            path = output_folder+f"_{i}"
            name = path+".wav"
            os.remove(name)
            os.rename(path+"/vocals.wav", path+f"/vocals_{i}.wav")
            os.rename(path+"/accompaniment.wav", path+f"/accompaniment_{i}.wav")
            shutil.copy(path+f"/vocals_{i}.wav", self.project_folder)
            shutil.copy(path+f"/accompaniment_{i}.wav", self.project_folder)
            shutil.rmtree(path)
        self._glue_files(self.project_folder,"vocals"), self._glue_files(self.project_folder,"accompaniment")
        vocals = self.project_folder+'/vocals_sep.wav'
        audio = AudioSegment.from_file(vocals)
        lower_sample_rate = 24000
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
        dubber.dub(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[3])
        