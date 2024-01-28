import os
import sys
import subprocess
import shutil

#Arguments ProjectDirectory VideoName SouceLanguage DestinationLanguage AccentLanguage VoiceChannelLeft=0 VoiceChannelRight=1
#Language uses code ex: en, ua, fr, de etc.
def main():
    script_directory = os.path.dirname(__file__)
    os.chdir(script_directory)
    if len(sys.argv) < 6:
        print("Missing arguments")
    else:
        #abstraction layer to free vram for each subroutine (some objects like Spleeter stay in vram even after exiting scope or autodisposal)
        video_path = sys.argv[1]+'/'+sys.argv[2]
        subprocess.run(['spleeter','separate','-o', sys.argv[1] ,'-p', 'spleeter:2stems', video_path])
        proj = video_path.split('.mp4')[0]
        arg = video_path.split('.mp4')[0]+'/vocals.wav'
        vocals = proj+'/vocals'
        arg2 = proj+'/accompaniment.wav'
        subprocess.run(['python',script_directory+f"/SplitAudio.py", proj])
        subprocess.run(['python',script_directory+f"/Diarize.py",arg])
        subprocess.run(['python',script_directory+f"/Transcribe.py", vocals, arg, sys.argv[3]])
        subprocess.run(['python',script_directory+f"/Translate.py", vocals, script_directory, sys.argv[3], sys.argv[4]])        
        subprocess.run(['python',script_directory+f"/synthesize.py", vocals, sys.argv[5], arg2])
        subprocess.run(['python',script_directory+f"/RecoverVideo.py", video_path, vocals, sys.argv[1]])
        shutil.rmtree(vocals)
        
if __name__ == "__main__":
    main()