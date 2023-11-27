import serial
from joblib import load
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from pyts.image import GramianAngularField
import pandas as pd
import numpy as np
import pandas as pd
import cv2
from scipy.signal import butter, lfilter
from scipy.fft import fft, ifft

clf = load('C:/Users/aless/Desktop/opencv competition/clf.joblib')
scaler2 = load('C:/Users/aless/Desktop/opencv competition/scaler2.joblib')

#---------------------------------------------------------------------------------------
# Initialize communication with Arduino
arduino = serial.Serial('COM9', 9600)

# Create a vector to store signal
signal = []

start_time = time.time()
while time.time() - start_time < 5:
    # Read a line from Arduino
    data = arduino.readline()
    if data:
        string = data.decode()
        string2 = string.strip()
        num = int(string2)
        signal.append(num)

# Close communication with Arduino
#---------------------------------------------------------------------------------------
arduino.close()
print("Signal Received")
#---------------------------------------------------------------------------------------
# 1) Clean signal
signal = np.array(signal)
signal = -signal
def butter_bandpass_filter(data, lowcut, highcut, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data
lowcut = 85
highcut = 255
filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, sampling_rate=1000)
#---------------------------------------------------------------------------------------
# 2) Denoise signal
spectrum = fft(signal)
mean_spectrum = np.mean(np.abs(spectrum))
spectrum_denoised = spectrum - mean_spectrum
signal_denoised = ifft(spectrum_denoised).real
#---------------------------------------------------------------------------------------
# 3) Standardize
mean = np.mean(signal_denoised)
sd_sign = signal_denoised-mean
sd_sign = sd_sign[10:]
#---------------------------------------------------------------------------------------
# 4) Gramian Angular Field
signal = pd.DataFrame(sd_sign)
array = signal.values
array = np.transpose(array)
gasf = GramianAngularField(method='summation', image_size=128)
img = gasf.transform(array)
img = (img - img.min()) / (img.max() - img.min()) * 255 #ensure range [0, 255]
img = img.astype(np.uint8)
img = img.transpose(1,2,0)
#---------------------------------------------------------------------------------------
# 5) HOG
win_size = (128, 128)
cell_size = (4, 4)
block_size = (8, 8)
block_stride = (4, 4)
num_bins = 18
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
image = cv2.resize(img, (128, 128))
features = hog.compute(image)
#---------------------------------------------------------------------------------------
# 6) Min-Max Scaling
features = features[np.newaxis, :]
scaled = scaler2.transform(features)
#----------------------------------------------------------------------------------------
# 7) Linear Support Vector Machine
prediction = clf.predict(scaled)
if prediction == 0:
    prediction = 'Healthy'
else:
    prediction = 'Pathological'
#----------------------------------------------------------------------------------------
# 8) Display
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

axes[0].plot(sd_sign)
axes[0].set_title('Recorded 5 seconds of vowel signal')

axes[1].imshow(image, cmap='gray')
axes[1].axis('off')
axes[1].set_title('GAF Image')

# Reshape the HOG features into a 2D array
hog_features = features.reshape(-1, num_bins)
# Create a blank image to visualize HOG features
hog_image = np.zeros((128, 128), dtype=float)
# Iterate through HOG features and draw HOG descriptors as gradients
cell_size_x, cell_size_y = cell_size
for i in range(hog_features.shape[0]):
    for j in range(num_bins):
        magnitude = hog_features[i][j]
        angle_radians = j * np.pi / num_bins
        x = int(i % (128 / cell_size_x)) * cell_size_x + cell_size_x // 2
        y = int(i // (128 / cell_size_y)) * cell_size_y + cell_size_y // 2
        x1 = x + int(magnitude * np.cos(angle_radians) * cell_size_x / 2)
        y1 = y + int(magnitude * np.sin(angle_radians) * cell_size_y / 2)
        cv2.line(hog_image, (y, x), (y1, x1), 255, 1)

axes[2].imshow(hog_image, cmap='gray')
axes[2].axis('off')
axes[2].set_title('HOG Features')

plt.suptitle(str(prediction))
plt.show(block=True)