import os
import sys
import subprocess
import shutil

#Arguments ProjectDirectory PathToVideo SouceLanguage DestinationLanguage AccentLanguage VoiceChannelLeft=0 VoiceChannelRight=1
#Language uses code ex: en, ua, fr, de etc.
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Missing arguments")
    else:
        #abstraction layer to free vram for each subroutine (some objects like Spleeter stay in vram even after exiting scope or autodisposal)
        left = 0 if len(sys.argv) else sys.argv[5]
        right = 1 if len(sys.argv) else sys.argv[6]
        subprocess.run('python '+os.getcwd()+f"\\SplitAudio.py {sys.argv[1]} {sys.argv[2]} {left} {right}")
        #this gave us vocals_sep.wav and accompaniment_sep.wav in project folder, now we need to diarize the vocals into own folder, yes individual speakers
        arg = sys.argv[1]+'\\vocals_sep.wav'
        subprocess.run('python '+os.getcwd()+f"\\Diarize.py {arg}")
        arg1 = sys.argv[1]+'\\vocals_sep'
        arg2 = sys.argv[1]+'\\accompaniment_sep.wav'
        arg3 = sys.argv[1]+'\\intermitent_stereo.wav'
        subprocess.run('python '+os.getcwd()+f"\\Transcribe.py {arg1} {arg} {sys.argv[3]} {sys.argv[4]}")
        subprocess.run('python '+os.getcwd()+f"\\Synthesize.py {arg1} {sys.argv[4]} {arg3}")
        subprocess.run('python '+os.getcwd()+f"\\RecoverVideo.py {sys.argv[2]} {sys.argv[1]} {left} {right}")
        shutil.rmtree(arg1)
        os.remove(arg)
        os.remove(arg2)
        os.remove(arg3)
        