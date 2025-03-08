import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

if __name__ == "__main__":
    sample_rate, raw_data = wavfile.read("star_noisy.wav")
    raw_data = raw_data[0:16000]
    
    fig, ax = plt.subplots(3, 1)
    fig.set_figwidth(10)
    fig.set_figheight(8)
    fig.canvas.manager.set_window_title("D1009212 - HW02")
    plt.subplots_adjust(hspace=0.5)
    
    # (a) Waveform of the star_noisy.wav
    norm_data = raw_data / np.max(np.abs(raw_data))
    norm_data = norm_data - np.mean(norm_data)
    plt.subplot(3, 1, 1)
    plt.plot(norm_data, linewidth=0.5, color='blue')
    plt.axvline(x=6000, color='red', linewidth=0.5)
    plt.axvline(x=6256, color='red', linewidth=0.5)
    plt.grid(linestyle='--')
    plt.title("(a) Waveform of the star_noisy.wav")
    plt.xlim(0, 16000)
    plt.ylim(-1, 1)
    
    # (b) Frame (blue) and the fitted line of third-order polynomial (red)
    frame_data = norm_data[6000:6000 + 256]
    x = np.arange(256)
    coefs = np.polyfit(x, frame_data, 3)
    coefs_point = np.polyval(coefs, x)
    plt.subplot(3, 1, 2)
    plt.plot(frame_data, linewidth=1, color='blue')
    plt.plot(coefs_point, linewidth=1, color='red')
    plt.grid(linestyle='--')
    plt.title("(b) Frame (blue) and the fitted line of third-order polynomial (red)")
    plt.xlim(0, 256)
    
    # (c) Frame after zero-justification
    zj_frame_data = frame_data - coefs_point
    plt.subplot(3, 1, 3)
    plt.plot(zj_frame_data, linewidth=1, color='blue')
    plt.grid(linestyle='--')
    plt.title("(c) Frame after zero-justification")
    plt.xlim(0, 256)
    
    plt.show()