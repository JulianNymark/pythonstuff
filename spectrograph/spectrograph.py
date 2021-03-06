import pyaudio
import sys
import pygame
import math

import numpy as np


def gray(im):
    im = 255 * (im / im.max())
    w = im.shape[0]
    h = im.shape[1]
    ret = np.zeros((w, h, 3), dtype=np.uint8)
    ret[..., 2] = ret[..., 1] = ret[..., 0] = im
    return ret


# No. of input samples used for a single FFT
CHUNK_SIZE = 1024

# Sampling rate
RATE = 44100

# Spectrogram's width in pixels
WINDOW_WIDTH = int(1400)

# Spectrogram's height in pixels
HEIGHT = int(CHUNK_SIZE / 2)

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                input_device_index=12  # if default don't work, set this to
                # an audio device given by "python -m sounddevice"!
                )

spectrogram = np.zeros([WINDOW_WIDTH, HEIGHT], dtype='uint16')

screen = pygame.display.set_mode((WINDOW_WIDTH, HEIGHT))

while (True):
    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
    data = np.fromstring(data, 'int16')
    freq = np.fft.rfft(data)

    tmp = np.zeros([WINDOW_WIDTH, HEIGHT], dtype='uint16')

    # Copy LAST WIDTH-1 columns from spectogram
    # to the FIRST WIDTH-1 columns in tmp
    tmp[0:WINDOW_WIDTH - 1, 0:HEIGHT] = spectrogram[1:WINDOW_WIDTH, 0:HEIGHT]

    for i in range(1, HEIGHT):
        idx = round((float(i**1.5) / HEIGHT**1.5) * HEIGHT)
        # replace idx with i if you want linear
        rvalue = abs(int(np.real(freq[idx])))
        tmp[-1, HEIGHT - i] = rvalue

    spectrogram = tmp

    image = gray(spectrogram)
    surface = pygame.surfarray.make_surface(image)

    screen.blit(surface, (0, 0))
    pygame.display.flip()

    pygame.event.pump()
    # event = pygame.event.wait() # ?? only redraws when mouse over?
    for event in pygame.event.get():
        if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_ESCAPE]:
            pygame.display.quit()
            break

stream.stop_stream()
stream.close()
p.terminate()
