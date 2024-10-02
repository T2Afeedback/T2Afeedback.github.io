import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 加载音频文件
# file_path = 'G:/dataset/picoaudio/data/multi_event_train/syn_3492.wav'
# file_path = 'G:/dataset/picoaudio/data/multi_event_train/syn_1796.wav'
file_path = r'D:\WeChat Files\WeChat Files\wxid_y5ck9wiewtzm22\FileStorage\File\2024-10\T2Afeedback.github.io\T2Afeedback.github.io\res\quality\1\A woman speaking continuously.wav'
y, sr = librosa.load(file_path, sr=None)

# 计算短时傅里叶变换 (Short-Time Fourier Transform, STFT)
D = librosa.stft(y)

# 将幅度谱转换为分贝（dB）
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# 绘制频谱图
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')

# 添加颜色条，标题和标签
plt.colorbar(format='%+2.0f dB')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
plt.show()
