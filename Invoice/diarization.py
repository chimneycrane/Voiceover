from pyannote.audio import Pipeline
import torch
from TTS.api import TTS

def Synthesize(diary):
    print(TTS().list_models())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
def Diarize(audio_path):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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")
    print(torch.cuda.get_arch_list())
    # send pipeline to GPU (when available)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # apply pretrained pipeline
    diarization = pipeline(audio_path)
    res = diarization.itertracks(yield_label=True)
    return res
    # print the result
    #for turn, _, speaker in diarization.itertracks(yield_label=True):
    #    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    # start=0.2s stop=1.5s speaker_0
    # start=1.8s stop=3.9s speaker_1
    # start=4.2s stop=5.7s speaker_0
# ...