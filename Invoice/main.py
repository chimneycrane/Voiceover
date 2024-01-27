import os
import sys
import subprocess
import shutil

#Arguments ProjectDirectory PathToVideo SouceLanguage DestinationLanguage AccentLanguage VoiceChannelLeft=0 VoiceChannelRight=1
#Language uses code ex: en, ua, fr, de etc.
def main():
    script_directory = os.path.dirname(__file__)
    os.chdir(script_directory)
    if len(sys.argv) < 6:
        print("Missing arguments")
    else:
        #abstraction layer to free vram for each subroutine (some objects like Spleeter stay in vram even after exiting scope or autodisposal)
        left = 0 if len(sys.argv) else sys.argv[5]
        right = 1 if len(sys.argv) else sys.argv[6]
        subprocess.run('python '+script_directory+f"/SplitAudio.py {sys.argv[1]} {sys.argv[2]} {left} {right}")
        #this gave us vocals_sep.wav and accompaniment_sep.wav in project folder, now we need to diarize the vocals into own folder, yes individual speakers
        arg = sys.argv[1]+'/vocals_sep.wav'
        subprocess.run('python '+script_directory+f"/Diarize.py {arg}")
        arg1 = sys.argv[1]+'/vocals_sep'
        arg2 = sys.argv[1]+'/accompaniment_sep.wav'
        arg3 = sys.argv[1]+'/intermitent_stereo.wav'
        subprocess.run('python '+script_directory+f"/Transcribe.py {arg1} {arg} {sys.argv[3]} {sys.argv[4]}")
        subprocess.run('python '+script_directory+f"/Translate.py {arg1} {sys.argv[3]} {sys.argv[4]}")        
        subprocess.run('python '+script_directory+f"/Synthesize.py {arg1} {sys.argv[5]} {arg2}")
        subprocess.run('python '+script_directory+f"/RecoverVideo.py {sys.argv[2]} {arg1} {sys.argv[1]}")
        shutil.rmtree(arg1)
        os.remove(arg)
        os.remove(arg2)
        os.remove(arg3)

if __name__ == "__main__":
    main()