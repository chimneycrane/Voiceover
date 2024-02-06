from setuptools import setup, Command, find_packages

def run():
    import subprocess
    subprocess.call(['pip', 'install', 'protobuf==3.20.0','--force-install'])
      
setup(
    name='Invoice',  # Replace with your desired package name
    python_requires='>=3.8',
    version='1.0.0',  # Replace with your desired version
    description='Video dubbing package',
    author='Alex Don',
    author_email='alex.don.8096@gmail.com',
    packages=find_packages(),
    setup_requires=['setuptools_git'],
    entry_points={
        'console_scripts': [
            'invoice=Invoice.main:main',
        ],
    },
    dependency_links=[
        'https://download.pytorch.org/whl/cu121'  # Specify the index URL
    ],
    install_requires=[
        'typer'
       ,'rpy2==3.5.15'
       ,'pandas'
       ,'scipy'
       ,'xgboost'
       ,'scikit-learn'
       ,'numpy==1.22.0'
       ,'openai-whisper==20231117'
       ,'TTS'
       ,'torch==2.1.1'
       ,'torchaudio==2.1.1'
       ,'torchvision==0.16.1'
       ,'spleeter==2.4.0'
       ,'soundstretch'
       ,'httpx[http2]==0.19.0'
       ,'pytube==15.0.0'
       ,'pydub==0.25.1'
       ,'cffi==1.16.0'
       ,'accelerate==0.26.1'
       ,'pyannote.audio==3.1.1'
       ,'tensorflow'
       ,'transformers==4.36.2'
       ,'numba==0.58.1'
       ,'deep-translator==1.11.4'
       ,'ipython==7.34.0'
       ,'ffmpeg-python==0.2.0'
       ,'language-tool-python==2.7.1'
       ,'protobuf==3.20.0'
       ,'librosa'
    ],
    cmdclass={
        'custom_install': run,
    }
)
