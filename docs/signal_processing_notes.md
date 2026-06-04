# Signal Processing Notes

## PCM

PCM stores audio as a sequence of sampled amplitude values. For 16-bit PCM WAV files, each sample is usually an integer from `-32768` to `32767`. Most speech processing code converts these values to floating point in the approximate range `[-1.0, 1.0]`.

## RMS

RMS means root mean square:

```text
RMS = sqrt(mean(x^2))
```

It measures average signal energy. Directly averaging waveform samples is not useful because positive and negative waveform values cancel each other.

## dBFS

dBFS means decibels relative to full scale. In digital audio, `0 dBFS` is the maximum representable level. Anything above this cannot be represented and may become clipping.

For amplitude values:

```text
dBFS = 20 * log10(amplitude)
```

For power values:

```text
dB = 10 * log10(power)
```

## Short-Time Analysis

Speech is not stationary over long periods. A common strategy is to analyze short overlapping frames:

```text
frame length: 25 ms
hop length:   10 ms
```

At 16 kHz:

```text
25 ms = 400 samples
10 ms = 160 samples
```

This setup is widely used for short-time energy, STFT, mel spectrograms, MFCCs, and VAD.

## STFT

STFT means short-time Fourier transform. The workflow is:

```text
waveform -> frames -> window function -> FFT -> magnitude -> dB -> spectrogram
```

A Hann window reduces sharp frame-boundary discontinuities and makes the frequency spectrum cleaner.

