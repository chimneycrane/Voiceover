import os
import sys
import subprocess

def replace_audio(video_path, audio_path, output_path, wd):
    command = ['ffmpeg', '-i', video_path, '-i', audio_path, '-c:v', 'libx264', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', output_path]
    subprocess.run(command)
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing argument audio_path")
    else:
        replace_audio(sys.argv[1], sys.argv[2]+'/result.wav', sys.argv[2]+'/../result.mp4', sys.argv[2])


