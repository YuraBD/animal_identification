from pydub import AudioSegment
import os

def convert_to_mp3(input_file, output_file):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="mp3")

directory = 'crow'

i = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        print(f)
        convert_to_mp3(f, f"crows_mp3_renamed/crow_{i}.mp3")
    i+=1
