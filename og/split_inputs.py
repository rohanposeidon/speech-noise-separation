from pydub import AudioSegment
from td_utils import *
sound = AudioSegment.from_file('../1Aero/Train_Set_1/FS_P01_dev_001.wav')
n = len(sound)
print(n)
i = 0
sounds = []
while i < n:
    sounds.append(sound[i:i+20000])
    i = i+20000

#print(sounds)
for i in range(90):
    sounds[i].export("../1/output{}.wav".format(i), format="wav")
    print(graph_spectrogram("../1/output{}.wav".format(i)).shape)