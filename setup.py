from setuptools import setup, find_packages

setup(
    name='Invoice',  # Replace with your desired package name
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
    dependency_links=['git+https://github.com/m3hrdadfi/soxan.git'],
    install_requires=[
        'openai-whisper'
       ,'TTS'
       ,'torch==2.1.1'
       ,'torchaudio==2.1.1'
       ,'spleeter'
       ,'audiostretchy==1.3.5           '
       ,'httpx[http2]==0.19.0'
       ,'numpy==1.25.0'
       ,'pytube'
       ,'pydub'
       ,'cffi'
       ,'accelerate'
       ,'pyannote.audio'
       ,'tensorflow'
       ,'transformers==4.36.2'
       ,'numba'
       ,'deep-translator'
       ,'ipython'
       ,'ffmpeg-python==0.2.0'
       ,'language-tool-python'
    ]
)
