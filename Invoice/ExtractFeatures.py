import os
from typing import List, Optional, Union, Dict

import tqdm
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Wav2Vec2Processor
)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        basedir: Optional[str] = None,
        sampling_rate: int = 16000,
        max_audio_len: int = 5,
    ):
        self.dataset = dataset
        self.basedir = basedir

        self.sampling_rate = sampling_rate
        self.max_audio_len = max_audio_len

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.dataset)

    def __getitem__(self, index):
        if self.basedir is None:
            filepath = self.dataset[index]
        else:
            filepath = os.path.join(self.basedir, self.dataset[index])
    
        speech_array, sr = torchaudio.load(filepath)
    
        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)
    
        if sr != self.sampling_rate:
            transform = torchaudio.transforms.Resample(sr, self.sampling_rate)
            speech_array = transform(speech_array)
            sr = self.sampling_rate
    
        len_audio = speech_array.shape[1]
    
        # Pad or truncate the audio to match the desired length
        if len_audio < self.max_audio_len * self.sampling_rate:
            # Pad the audio if it's shorter than the desired length
            padding = torch.zeros(1, self.max_audio_len * self.sampling_rate - len_audio)
            speech_array = torch.cat([speech_array, padding], dim=1)
        else:
            # Truncate the audio if it's longer than the desired length
            speech_array = speech_array[:, :self.max_audio_len * self.sampling_rate]
    
        speech_array = speech_array.squeeze().numpy()
    
        return {"input_values": speech_array, "attention_mask": None}


class CollateFunc:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        pad_to_multiple_of: Optional[int] = None,
        sampling_rate: int = 16000,
    ):
        self.padding = padding
        self.processor = processor
        self.max_length = max_length
        self.sampling_rate = sampling_rate
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List):
        input_features = []

        for audio in batch:
            input_tensor = self.processor(audio, sampling_rate=self.sampling_rate).input_values
            input_tensor = np.squeeze(input_tensor)
            input_features.append({"input_values": input_tensor})

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return batch


def predict(test_dataloader, model, device: torch.device):
    """
    Predict the class of the audio
    """
    model.to(device)
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader):
            input_values, attention_mask = batch['input_values'].to(device), batch['attention_mask'].to(device)

            logits = model(input_values, attention_mask=attention_mask).logits
            scores = F.softmax(logits, dim=-1)

            pred = torch.argmax(scores, dim=1).cpu().detach().numpy()

            preds.extend(pred)

    return preds



model_name_or_path = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
audio_paths = [] # Must be a list with absolute paths of the audios that will be used in inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label2id = {
    "female": 0,
    "male": 1
}

id2label = {
    0: "female",
    1: "male"
}

def get_voice_feature(audio_paths):
    num_labels = 2

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    test_dataset = CustomDataset(audio_paths, max_audio_len=5)  # for 5-second audio

    data_collator = CollateFunc(
        processor=feature_extractor,
        padding=True,
        sampling_rate=16000,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=10
    )

    preds = predict(test_dataloader=test_dataloader, model=model, device=device)

    return preds
