# spectrograph

- a simple spectrograph using pygame + pyaudio + numpy
- every frame, a column in a 2d numpy array is added from FFT of an audio stream
- loosely based on code from https://github.com/lgeek/Python-Spectrograph/blob/master/spectrograph.py (python 2 + old opencv)
![spectrograph image](/spectrograph/spectrograph.png "spectrograph")

## notes:
tried matplotlib & PIL (couldn't get opencv to compile with gtk, but it _should_ be possible to get realtime image updates / redraws in a window with opencv)
