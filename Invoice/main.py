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
        
        subprocess.run(['spleeter','separate','-o', sys.argv[1] ,'-p', 'spleeter:2stems', sys.argv[2]])
        vocals = sys.argv[2].split('.mp4')[0]
        arg = sys.argv[2].split('.mp4')[0]+'/vocals.wav'
        subprocess.run(['python',script_directory+f"/SplitAudio.py", vocals])
        subprocess.run(['python',script_directory+f"/Diarize.py",arg])
        arg2 = vocals+'/accompaniment.wav'
        subprocess.run(['python',script_directory+f"/Transcribe.py", vocals, arg, sys.argv[3], sys.argv[4]])
        subprocess.run(['python',script_directory+f"/Translate.py", vocals, sys.argv[3], sys.argv[4]])        
        subprocess.run(['python',script_directory+f"/Synthesize.py", vocals, sys.argv[5], arg2])
        subprocess.run(['python',script_directory+f"/RecoverVideo.py", sys.argv[2], vocals, sys.argv[1]])
        shutil.rmtree(vocals)
        
if __name__ == "__main__":
    main()