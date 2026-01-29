# img2sound

Convert images and video to audio using sonification.

## Features

- **Real-time sync**: Video audio matches video duration exactly
- **Auto-scaling**: Analyzes brightness range for optimal dynamics
- **Auto-versioning**: Output files as `YYMMDD/YYMMDD_<filename>_<mode>_v001.wav`
- **Progress bar**: Visual feedback during video rendering
- **Three modes**: Scanline, Spectral, and Additive synthesis

## Installation

```bash
cd img2sound
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Modes

### Scanline
Direct spatial mapping. Each column = moment in time, brightness = amplitude.
Best for: glitchy, raw, waveform-like sounds. Now with bass boost and filtering!

### Spectral
Treats image as spectrogram. Y = frequency, X = time, brightness = magnitude.
Best for: rich, textural, ambient sounds with bass.

### Additive (Best for visual sync)
Each image row = a sine oscillator. Most direct visual-to-audio mapping.
- Horizontal lines = sustained tones
- Vertical lines = clicks/transients
- Diagonal lines = pitch sweeps
- Bright areas = loud frequencies

## Usage

```bash
# Scanline with bass boost
img2sound scanline video.mp4 --bass-boost 2.0
img2sound scanline video.mp4 --bass-boost 2.5 --lowpass 4000

# Spectral - rich textures
img2sound spectral video.mp4
img2sound spectral photo.jpg --bass-boost 3.0

# Additive - best visual sync
img2sound additive video.mp4
img2sound additive artwork.png --duration 10
```

## Output

Files auto-version with mode name:
```
260129/260129_video_scanline_v001.wav
260129/260129_video_spectral_v001.wav
260129/260129_video_additive_v001.wav
```

Progress bar shows during video processing:
```
  Rendering [████████████████████░░░░░░░░░░░░░░░░░░░░]  52.3% (157/300)
```

## Options

### Global
| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | auto | Output path |
| `--sample-rate` | 44100 | Sample rate Hz |
| `--bit-depth` | 16 | 16 or 32 bit |
| `-v, --verbose` | off | Show details |

### Scanline
| Option | Default | Description |
|--------|---------|-------------|
| `--direction` | horizontal | horizontal or vertical |
| `--channel-mode` | mono | mono, stereo, rgb |
| `--invert` | off | Dark = loud |
| `--bass-boost` | 1.0 | Bass boost (1.0 = none, 2.0 = double) |
| `--lowpass` | none | Lowpass filter cutoff Hz |
| `--highpass` | none | Highpass filter cutoff Hz |
| `--smooth` | 0 | Smoothing (0 = off, try 3-11) |

### Spectral
| Option | Default | Description |
|--------|---------|-------------|
| `--freq-min` | 30 | Min frequency Hz |
| `--freq-max` | 10000 | Max frequency Hz |
| `--bass-boost` | 2.5 | Bass boost (1.0 = none) |
| `--phase` | random | random, griffin-lim, zero |
| `--log-freq` | on | Logarithmic frequency scale |

### Additive
| Option | Default | Description |
|--------|---------|-------------|
| `--freq-min` | 40 | Min frequency Hz |
| `--freq-max` | 4000 | Max frequency Hz |
| `--oscillators` | 64 | Number of sine oscillators |
| `--log-freq` | on | Logarithmic frequency scale |

## Examples

```bash
# Scanline with warm bass
img2sound scanline video.mp4 --bass-boost 2.5 --lowpass 6000

# Remove harsh highs, boost bass
img2sound scanline video.mp4 --lowpass 4000 --bass-boost 2.0 --smooth 5

# Spectral with extra bass
img2sound spectral video.mp4 --bass-boost 3.5 --freq-min 20

# Musical additive range (A1 to A5)
img2sound additive video.mp4 --freq-min 55 --freq-max 880
```

## Supported Formats

**Images**: PNG, JPG, JPEG, TIFF, BMP, WebP
**Video**: MP4, MOV, AVI, MKV, WebM, GIF

## License

MIT
