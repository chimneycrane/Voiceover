import sys
import pickle
from Predict import *
from language_tool_python import LanguageTool
from deep_translator import GoogleTranslator
from pydub import AudioSegment

diary =[]
with open(sys.argv[2]+'/transcript.pickle', 'rb') as file:
    diary = pickle.load(file)
grammar_modifier = dict()
for rec in diary:
    grammar_modifier[rec[2]]=''
tool = LanguageTool(sys.argv[3])
for rec in diary:
    speaker_aud = AudioSegment.from_file(rec[4])
    if grammar_modifier[rec[2]]=='':
        grammar_modifier[rec[2]]=predict(f"{rec[2]}.wav", sys.argv[2])
    feature = 'male' if grammar_modifier[rec[2]] == 'M' else 'female'
    translation = GoogleTranslator(source=sys.argv[3], target=sys.argv[4]).translate(f'({feature}): '+rec[3]).split('):')[1]
    rec[3] = tool.correct(translation)