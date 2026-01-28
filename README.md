# img2sound

Convert images and video to audio using sonification.

## Features

- **Real-time sync**: Video audio matches video duration exactly
- **Auto-scaling**: Analyzes brightness range for optimal dynamics
- **Auto-versioning**: Output files as `YYMMDD/YYMMDD_<filename>_v001.wav`
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
Best for: glitchy, raw, waveform-like sounds.

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
# Additive - best visual sync
img2sound additive video.mp4
img2sound additive artwork.png --duration 10

# Spectral - rich textures with bass
img2sound spectral video.mp4
img2sound spectral photo.jpg --bass-boost 3.0

# Scanline - raw/glitchy
img2sound scanline video.mp4
img2sound scanline image.png --channel-mode stereo
```

## Output

Files auto-version to prevent overwrites:
```
260128/260128_video_v001.wav
260128/260128_video_v002.wav
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

### Scanline
| Option | Default | Description |
|--------|---------|-------------|
| `--direction` | horizontal | horizontal or vertical |
| `--channel-mode` | mono | mono, stereo, rgb |
| `--invert` | off | Dark = loud |

## Supported Formats

**Images**: PNG, JPG, JPEG, TIFF, BMP, WebP
**Video**: MP4, MOV, AVI, MKV, WebM, GIF

## License

MIT
