import pyaudio
import sys
import pygame

import numpy as np


def gray(im):
    im = 255 * (im / im.max())
    w = im.shape[0]
    h = im.shape[1]
    ret = np.zeros((w, h, 3), dtype=np.uint8)
    ret[..., 2] = ret[..., 1] = ret[..., 0] = im
    return ret


# No. of input samples used for a single FFT
CHUNK_SIZE = 2048

# Sampling rate
RATE = 44100

# Spectrogram's width in pixels
WINDOW_WIDTH = int(700)

# Spectrogram's height in pixels
HEIGHT = int(CHUNK_SIZE / 2)

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                input_device_index=3 # if default don't work, set this to
                # an audio device given by "python -m sounddevice"!
                )

spectrogram = np.zeros([WINDOW_WIDTH, HEIGHT], dtype='uint16')

screen = pygame.display.set_mode((WINDOW_WIDTH, HEIGHT))

i2 = 0
while (True):
    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
    data = np.fromstring(data, 'int16')
    freq = np.fft.rfft(data)
    print("SHAPEEEEE", freq.shape)

    tmp = np.zeros([WINDOW_WIDTH, HEIGHT], dtype='uint16')

    # Copy LAST WIDTH-1 columns from spectogram
    # to the FIRST WIDTH-1 columns in tmp
    tmp[0:WINDOW_WIDTH - 1, 0:HEIGHT] = spectrogram[1:WINDOW_WIDTH, 0:HEIGHT]

    for i in range(1, HEIGHT):
        # rvalue = abs(int(np.real(freq[i])))
        idx = (CHUNK_SIZE/2) * i**2/(((CHUNK_SIZE / 2))**2)
        diff = idx - np.floor(idx)
        r_v1 = abs(int(np.real(freq[int(idx)])))
        r_v2 = abs(int(np.real(freq[int(idx) + 1])))
        rvalue = r_v1 * diff + r_v2 * (1 - diff)
        # rvalue = abs(int(np.real(freq[int(idx)])))
        tmp[-1, HEIGHT - i] = rvalue

    spectrogram = tmp

    image = gray(spectrogram)
    surface = pygame.surfarray.make_surface(image)

    screen.blit(surface, (0, 0))
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_ESCAPE]:
            pygame.quit()
            break

stream.stop_stream()
stream.close()
p.terminate()
