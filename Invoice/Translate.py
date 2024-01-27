import sys
import pickle
import GrammarHelper
from language_tool_python import LanguageTool
from deep_translator import GoogleTranslator
from pydub import AudioSegment

diary =[]
with open(sys.argv[1]+'/transcript.pickle', 'rb') as file:
    diary = pickle.load(file)
grammar_modifier = dict()
for rec in diary:
    grammar_modifier[rec[2]]=''
tool = LanguageTool(sys.argv[3])
for rec in diary:
    speaker_aud = AudioSegment.from_file(rec[4])
    if grammar_modifier[rec[2]]=='':
        grammar_modifier[rec[2]]=GrammarHelper.predict(rec[4])
    feature = 'male' if grammar_modifier[rec[2]] == 'M' else 'female'
    translation = GoogleTranslator(source=sys.argv[2], target=sys.argv[3]).translate(f'({feature}): '+rec[3]).split('):')[1]
    rec[3] = tool.correct(translation)
