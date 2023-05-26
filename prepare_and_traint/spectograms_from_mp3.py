import matplotlib.image
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def cut(signal):
    signal /= np.max(signal)
    plt.plot(np.abs(signal))
    orig_signal = signal
    signal = np.abs(signal)

    means = []
    for i in range(0, len(signal)):

        try:
            curSig = signal[i:i + 100]
            means.append(np.mean(curSig))

        except:
            continue

    means = np.array(means)

    start = 0
    stop = 0

    max_ind = np.argmax(signal)

    cur_ind = max_ind
    while (means[cur_ind] > 0.01):
        cur_ind += 1
    stop = cur_ind

    cur_ind = max_ind
    while (means[cur_ind] > 0.01):
        cur_ind -= 1
    start = cur_ind

    final_signal = np.zeros(70000)

    signal_useful = orig_signal[start:stop]
    num = int(len(final_signal) / (len(signal_useful)))
    ind = 0
    for i in range(num):
        final_signal[ind:ind + len(signal_useful)] = signal_useful
        ind = ind + len(signal_useful)

    final_signal[ind:] = signal_useful[:70000 - ind]
    return final_signal

def create_spectogram_image(path, out_path):
    signal, sr = librosa.load(path)
    signal = cut(signal)
    n_fft = 2048
    hop_length = 512

    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length,
                                                n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(8, 7))
    ax = plt.axes()
    ax.set_facecolor('black')
    img = librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hop_length)
    plt.xlim([0, 3])
    plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.savefig("out.png")
    plt.clf()
    image = mpimg.imread("out.png")
    image = image[85:624, 100:721]
    print(type(image))
    matplotlib.image.imsave(out_path, image)

directory = 'crows_mp3_renamed'

i = 0
for filename in os.listdir(directory):
    try:
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            print(f)
            create_spectogram_image(f, f"crows_spectograms1/crow{i}.png")
        i += 1
        print(i)
    except:
        continue