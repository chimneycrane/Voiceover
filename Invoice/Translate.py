import sys
import pickle
from Predict import *
from language_tool_python import LanguageTool
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from langdetect import detect
from num2words import num2words

def replace_numbers_with_words(text):
    words = text.split()
    new_words = []
    for word in words:
        if word.isdigit():
            new_words.append(num2words(int(word), lang=sys.argv[3]))
        else:
            new_words.append(word)
    return " ".join(new_words)

diary =[]
with open(sys.argv[1]+'/transcript.pickle', 'rb') as file:
    diary = pickle.load(file)
grammar_modifier = dict()
for rec in diary:
    grammar_modifier[rec[2]]=''
tool = LanguageTool(sys.argv[3])
for rec in diary:
    language = detect(rec[3])
    if language != sys.argv[4]:
        speaker_aud = AudioSegment.from_file(rec[4])
        if grammar_modifier[rec[2]]=='':
            grammar_modifier[rec[2]]='male'#predict(sys.argv[1]+f"/{rec[2]}.wav", sys.argv[2])
        feature = grammar_modifier[rec[2]]
        rec[3] = replace_numbers_with_words(rec[3])
        translation = GoogleTranslator(source=sys.argv[3], target=sys.argv[4]).translate(f'({feature}): '+rec[3])
        rec[3] = tool.correct(translation).split('):')[1]
        rec.append(1)
    else:
        rec.append(0)
    print(rec[3])

with open(sys.argv[1]+'/transcript.pickle', 'wb') as file:
    pickle.dump(diary, file, protocol=pickle.HIGHEST_PROTOCOL)