from setuptools import setup, find_packages

setup(
    name='Invoice',  # Replace with your desired package name
    version='1.0.0',  # Replace with your desired version
    description='Video dubbing package',
    author='Alex Don',
    author_email='alex.don.8096@gmail.com',
    packages=find_packages(),
    install_requires=[
        'accelerate==0.26.1             '
       ,'alembic==1.13.1                '
       ,'antlr4-python3-runtime==4.9.3  '
       ,'anyascii==0.3.2                '
       ,'asteroid-filterbanks==0.4.0    '
       ,'audiostretchy==1.3.5           '
       ,'bangla==0.0.2                  '
       ,'bnnumerizer==0.0.2             '
       ,'bnunicodenormalizer==0.1.6     '
       ,'click==7.1.2                   '
       ,'colorama==0.4.6                '
       ,'colorlog==6.8.2                '
       ,'coqpit==0.0.17                 '
       ,'dateparser==1.1.8              '
       ,'deep-translator==1.11.4        '
       ,'docopt==0.6.2                  '
       ,'einops==0.7.0                  '
       ,'encodec==0.1.1                 '
       ,'ffmpeg-python==0.2.0           '
       ,'fire==0.5.0                    '
       ,'flatbuffers==1.12              '
       ,'fsspec==2023.12.2              '
       ,'g2pkk==0.1.2                   '
       ,'gast==0.4.0                    '
       ,'google-auth-oauthlib==0.4.6    '
       ,'gruut==2.2.3                   '
       ,'gruut-ipa==0.13.0              '
       ,'gruut-lang-de==2.0.0           '
       ,'gruut-lang-en==2.0.0           '
       ,'gruut-lang-es==2.0.0           '
       ,'gruut-lang-fr==2.0.2           '
       ,'h11==0.12.0                    '
       ,'h2==4.1.0                      '
       ,'hangul-romanize==0.1.0         '
       ,'hpack==4.0.0                   '
       ,'httpcore==0.13.7               '
       ,'httpx==0.19.0                  '
       ,'hyperframe==6.0.1              '
       ,'HyperPyYAML==1.2.2             '
       ,'jamo==0.4.1                    '
       ,'jsonlines==1.2.0               '
       ,'julius==0.2.7                  '
       ,'keras==2.9.0                   '
       ,'Keras-Preprocessing==1.1.2     '
       ,'language-tool-python==2.7.1    '
       ,'librosa==0.10.0                '
       ,'lightning==2.1.3               '
       ,'lightning-utilities==0.10.1    '
       ,'Mako==1.3.0                    '
       ,'networkx==2.8.8                '
       ,'norbert==0.2.1                 '
       ,'num2words==0.5.13              '
       ,'numpy==1.26.3                  '
       ,'omegaconf==2.3.0               '
       ,'openai-whisper==20231117       '
       ,'optuna==3.5.0                  '
       ,'primePy==1.3                   '
       ,'protobuf==3.20.0               '
       ,'pyannote.audio==3.1.1          '
       ,'pyannote.core==5.0.0           '
       ,'pyannote.database==5.0.1       '
       ,'pyannote.metrics==3.2.1        '
       ,'pyannote.pipeline==3.0.1       '
       ,'pydub==0.25.1                  '
       ,'pynndescent==0.5.11            '
       ,'pypinyin==0.50.0               '
       ,'pysbd==0.3.4                   '
       ,'python-crfsuite==0.9.10        '
       ,'pytorch-lightning==2.1.3       '
       ,'pytorch-metric-learning==2.4.1 '
       ,'pytube==15.0.0                 '
       ,'regex==2023.12.25              '
       ,'rfc3986==1.5.0                 '
       ,'ruamel.yaml==0.18.5            '
       ,'ruamel.yaml.clib==0.2.8        '
       ,'safetensors==0.4.2             '
       ,'scikit-learn==1.4.0            '
       ,'semver==3.0.2                  '
       ,'sentencepiece==0.1.99          '
       ,'shellingham==1.5.4             '
       ,'speechbrain==0.5.16            '
       ,'spleeter==2.4.0                '
       ,'SudachiDict-core==20240109     '
       ,'SudachiPy==0.6.8               '
       ,'tensorboard==2.9.1             '
       ,'tensorboard-data-server==0.6.1 '
       ,'tensorboard-plugin-wit==1.8.1  '
       ,'tensorboardX==2.6.2.2          '
       ,'tensorflow==2.9.3              '
       ,'tensorflow-estimator==2.9.0    '
       ,'tiktoken==0.5.2                '
       ,'torch==2.1.1             '
       ,'torch-audiomentations==0.11.0  '
       ,'torch-pitch-shift==1.2.4       '
       ,'torchaudio==2.1.1       '
       ,'torchmetrics==1.3.0.post0      '
       ,'torchvision==0.16.1      '
       ,'trainer==0.0.36                '
       ,'transformers==4.36.2           '
       ,'TTS==0.22.0                    '
       ,'typer==0.3.2                   '
       ,'typing_extensions==4.9.0       '
       ,'umap-learn==0.5.5              '
       ,'Unidecode==1.3.8               '
       ,'urllib3==2.1.0                 '
    ],
    entry_points={
        'console_scripts': [
            'invoice=Invoice.main:main',
        ],
    },
    dependency_links=[
        'https://download.pytorch.org/whl/cu121'  # Specify the index URL
    ]
)
