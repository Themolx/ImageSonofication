"""Command-line interface for img2sound."""

import sys
import click
from pathlib import Path

from img2sound import __version__
from img2sound.core.scanline import scanline_image, scanline_video, process_scanline_audio
from img2sound.core.spectral import spectral_image, spectral_video
from img2sound.core.additive import additive_image, additive_video
from img2sound.core.audio import normalize_audio, write_wav, resample_audio
from img2sound.io.image import load_image, validate_image
from img2sound.io.video import get_video_info
from img2sound.utils.helpers import get_input_type, format_duration, generate_output_path, write_nfo, Img2SoundError


class ProgressBar:
    """Simple progress bar for video processing."""

    def __init__(self, total: int, width: int = 40):
        self.total = total
        self.width = width
        self.current = 0

    def update(self, current: int, total: int):
        self.current = current
        self.total = total
        self._render()

    def _render(self):
        if self.total == 0:
            return

        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = "█" * filled + "░" * (self.width - filled)
        percent_str = f"{percent * 100:5.1f}%"

        sys.stdout.write(f"\r  Rendering [{bar}] {percent_str} ({self.current}/{self.total})")
        sys.stdout.flush()

        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


@click.group()
@click.version_option(version=__version__)
def cli():
    """img2sound - Convert images and video to audio.

    Three sonification modes available:

    \b
    scanline  - Direct spatial mapping of brightness to amplitude
    spectral  - Treat image as spectrogram, reconstruct via inverse FFT
    additive  - Each image row = sine oscillator (best visual sync)

    Output files are auto-versioned: YYMMDD/YYMMDD_<filename>_<mode>_v001.wav
    Settings saved to matching .nfo file.
    """
    pass


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output WAV file path (auto-versioned if not specified)")
@click.option("--sample-rate", default=44100, type=int, help="Sample rate in Hz")
@click.option(
    "--bit-depth",
    type=click.Choice(["16", "32"]),
    default="16",
    help="Bit depth: 16 (int) or 32 (float)",
)
@click.option(
    "--direction",
    type=click.Choice(["horizontal", "vertical"]),
    default="horizontal",
    help="Scan direction: horizontal (L→R) or vertical (T→B)",
)
@click.option(
    "--channel-mode",
    type=click.Choice(["mono", "stereo", "rgb"]),
    default="mono",
    help="Audio channel mode",
)
@click.option(
    "--duration",
    type=float,
    default=None,
    help="Force output duration in seconds (images only)",
)
@click.option("--invert", is_flag=True, help="Invert brightness (dark = loud)")
@click.option("--bass-boost", default=1.0, type=float, help="Bass boost factor (1.0 = none, 2.0 = double)")
@click.option("--lowpass", default=None, type=int, help="Lowpass filter cutoff in Hz (e.g. 8000)")
@click.option("--highpass", default=None, type=int, help="Highpass filter cutoff in Hz (e.g. 20)")
@click.option("--smooth", default=0, type=int, help="Smoothing window size (0 = off, try 3-11)")
@click.option("-v", "--verbose", is_flag=True, help="Print processing info")
def scanline(
    input,
    output,
    sample_rate,
    bit_depth,
    direction,
    channel_mode,
    duration,
    invert,
    bass_boost,
    lowpass,
    highpass,
    smooth,
    verbose,
):
    """Scanline sonification mode.

    Scan image/frame column-by-column. Each column becomes a moment in time.
    Brightness values become amplitude values.

    \b
    Audio Processing:
      --bass-boost  Amplify low frequencies (try 1.5-3.0)
      --lowpass     Remove high frequencies above cutoff
      --highpass    Remove low frequencies below cutoff
      --smooth      Reduce harshness (try 3-11)

    \b
    Examples:
      img2sound scanline photo.jpg
      img2sound scanline video.mp4 --bass-boost 2.0
      img2sound scanline video.mp4 --lowpass 4000 --bass-boost 2.5
      img2sound scanline image.png --channel-mode stereo --direction vertical
    """
    try:
        input_type = get_input_type(input)

        if output is None:
            output = str(generate_output_path(input, mode="scanline"))

        if verbose:
            click.echo(f"Input: {input} ({input_type})")
            click.echo(f"Mode: scanline")
            click.echo(f"Direction: {direction}")
            click.echo(f"Channel mode: {channel_mode}")
            click.echo(f"Sample rate: {sample_rate} Hz")
            if bass_boost > 1.0:
                click.echo(f"Bass boost: {bass_boost}x")
            if lowpass:
                click.echo(f"Lowpass: {lowpass} Hz")
            if highpass:
                click.echo(f"Highpass: {highpass} Hz")
            if smooth > 0:
                click.echo(f"Smooth: {smooth}")

        if input_type == "image":
            image = load_image(input)
            validate_image(image)

            if verbose:
                h, w = image.shape[:2]
                click.echo(f"Image size: {w}x{h}")

            audio = scanline_image(image, direction, channel_mode, invert)

            if duration is not None:
                target_samples = int(duration * sample_rate)
                if verbose:
                    click.echo(f"Resampling to {duration}s")
                audio = resample_audio(audio, target_samples)

            audio = process_scanline_audio(
                audio,
                sample_rate,
                bass_boost=bass_boost,
                lowpass=lowpass,
                highpass=highpass,
                smooth=smooth,
            )
        else:
            info = get_video_info(input)
            if verbose:
                click.echo(f"Video: {info['width']}x{info['height']} @ {info['fps']:.2f} fps")
                click.echo(f"Duration: {format_duration(info['duration'])}")
                click.echo(f"Frames: {info['frame_count']}")

            progress = ProgressBar(info["frame_count"])

            audio = scanline_video(
                input,
                direction=direction,
                channel_mode=channel_mode,
                sample_rate=sample_rate,
                invert=invert,
                bass_boost=bass_boost,
                lowpass=lowpass,
                highpass=highpass,
                smooth=smooth,
                progress_callback=progress.update,
            )

        audio = normalize_audio(audio)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(str(output_path), audio, sample_rate, int(bit_depth))

        # Write NFO file with settings
        output_duration = len(audio) / sample_rate
        settings = {
            "mode": "scanline",
            "input": input,
            "input_type": input_type,
            "output": str(output_path),
            "sample_rate": f"{sample_rate} Hz",
            "bit_depth": bit_depth,
            "duration": f"{output_duration:.2f}s",
            "direction": direction,
            "channel_mode": channel_mode,
            "invert": invert,
            "bass_boost": bass_boost if bass_boost > 1.0 else None,
            "lowpass": f"{lowpass} Hz" if lowpass else None,
            "highpass": f"{highpass} Hz" if highpass else None,
            "smooth": smooth if smooth > 0 else None,
        }
        write_nfo(str(output_path), settings)

        click.echo(f"Written {output_path} ({format_duration(output_duration)})")

    except Img2SoundError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Processing failed: {e}")


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output WAV file path (auto-versioned if not specified)")
@click.option("--sample-rate", default=44100, type=int, help="Sample rate in Hz")
@click.option(
    "--bit-depth",
    type=click.Choice(["16", "32"]),
    default="16",
    help="Bit depth: 16 (int) or 32 (float)",
)
@click.option("--duration", type=float, default=5.0, help="Output duration in seconds (images only)")
@click.option("--freq-min", default=30, type=int, help="Minimum frequency in Hz")
@click.option("--freq-max", default=10000, type=int, help="Maximum frequency in Hz")
@click.option(
    "--phase",
    type=click.Choice(["random", "griffin-lim", "zero"]),
    default="random",
    help="Phase reconstruction method",
)
@click.option(
    "--griffin-iterations",
    default=32,
    type=int,
    help="Iterations for Griffin-Lim algorithm",
)
@click.option("--log-freq/--linear-freq", default=True, help="Use logarithmic frequency scale")
@click.option("--bass-boost", default=2.5, type=float, help="Bass boost factor (1.0 = none)")
@click.option("-v", "--verbose", is_flag=True, help="Print processing info")
def spectral(
    input,
    output,
    sample_rate,
    bit_depth,
    duration,
    freq_min,
    freq_max,
    phase,
    griffin_iterations,
    log_freq,
    bass_boost,
    verbose,
):
    """Spectral sonification mode.

    Treat the image as a spectrogram: X = time, Y = frequency, brightness = magnitude.
    Reconstruct audio using inverse STFT. Produces rich, textural sounds.

    \b
    Phase modes:
      random      - Washy/dreamy textures (fast)
      griffin-lim - Cleaner, more defined (slower)
      zero        - Harsh/clicky digital artifacts

    \b
    Examples:
      img2sound spectral artwork.png --duration 10
      img2sound spectral video.mp4
      img2sound spectral photo.jpg --phase griffin-lim
      img2sound spectral dark.png --freq-min 20 --freq-max 200  # Bass rumble
    """
    try:
        input_type = get_input_type(input)

        if output is None:
            output = str(generate_output_path(input, mode="spectral"))

        if verbose:
            click.echo(f"Input: {input} ({input_type})")
            click.echo(f"Mode: spectral")
            click.echo(f"Frequency range: {freq_min}-{freq_max} Hz")
            click.echo(f"Phase mode: {phase}")
            click.echo(f"Log frequency: {log_freq}")
            click.echo(f"Bass boost: {bass_boost}x")
            click.echo(f"Sample rate: {sample_rate} Hz")

        if input_type == "image":
            image = load_image(input)
            validate_image(image)

            if verbose:
                h, w = image.shape[:2]
                click.echo(f"Image size: {w}x{h}")
                click.echo(f"Target duration: {duration}s")
                if phase == "griffin-lim":
                    click.echo(f"Griffin-Lim iterations: {griffin_iterations}")

            audio = spectral_image(
                image,
                duration=duration,
                sample_rate=sample_rate,
                freq_min=freq_min,
                freq_max=freq_max,
                phase_mode=phase,
                griffin_iterations=griffin_iterations,
                log_freq=log_freq,
                bass_boost=bass_boost,
            )
        else:
            info = get_video_info(input)
            if verbose:
                click.echo(f"Video: {info['width']}x{info['height']} @ {info['fps']:.2f} fps")
                click.echo(f"Duration: {format_duration(info['duration'])}")
                click.echo(f"Frames: {info['frame_count']}")
                if phase == "griffin-lim":
                    click.echo(f"Griffin-Lim iterations: {griffin_iterations}")

            progress = ProgressBar(info["frame_count"])

            audio = spectral_video(
                input,
                sample_rate=sample_rate,
                freq_min=freq_min,
                freq_max=freq_max,
                phase_mode=phase,
                griffin_iterations=griffin_iterations,
                log_freq=log_freq,
                bass_boost=bass_boost,
                progress_callback=progress.update,
            )

        audio = normalize_audio(audio)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(str(output_path), audio, sample_rate, int(bit_depth))

        # Write NFO file with settings
        output_duration = len(audio) / sample_rate
        settings = {
            "mode": "spectral",
            "input": input,
            "input_type": input_type,
            "output": str(output_path),
            "sample_rate": f"{sample_rate} Hz",
            "bit_depth": bit_depth,
            "duration": f"{output_duration:.2f}s",
            "freq_min": f"{freq_min} Hz",
            "freq_max": f"{freq_max} Hz",
            "phase": phase,
            "griffin_iterations": griffin_iterations if phase == "griffin-lim" else None,
            "log_freq": log_freq,
            "bass_boost": bass_boost,
        }
        write_nfo(str(output_path), settings)

        click.echo(f"Written {output_path} ({format_duration(output_duration)})")

    except Img2SoundError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Processing failed: {e}")


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output WAV file path (auto-versioned if not specified)")
@click.option("--sample-rate", default=44100, type=int, help="Sample rate in Hz")
@click.option(
    "--bit-depth",
    type=click.Choice(["16", "32"]),
    default="16",
    help="Bit depth: 16 (int) or 32 (float)",
)
@click.option("--duration", type=float, default=5.0, help="Output duration in seconds (images only)")
@click.option("--freq-min", default=40, type=int, help="Minimum frequency in Hz")
@click.option("--freq-max", default=4000, type=int, help="Maximum frequency in Hz")
@click.option("--log-freq/--linear-freq", default=True, help="Use logarithmic frequency scale")
@click.option("--oscillators", default=64, type=int, help="Number of frequency bands/oscillators")
@click.option("-v", "--verbose", is_flag=True, help="Print processing info")
def additive(
    input,
    output,
    sample_rate,
    bit_depth,
    duration,
    freq_min,
    freq_max,
    log_freq,
    oscillators,
    verbose,
):
    """Additive synthesis mode - best visual-to-audio sync.

    Each row of the image becomes a sine wave oscillator:
    - Y position = frequency (bottom = bass, top = treble)
    - X position = time
    - Brightness = amplitude (volume) of that frequency

    This creates the most direct visual-to-audio mapping:
    - Horizontal lines = sustained tones
    - Vertical lines = clicks/transients
    - Diagonal lines = pitch sweeps
    - Bright areas = loud frequencies

    \b
    Examples:
      img2sound additive artwork.png --duration 10
      img2sound additive video.mp4
      img2sound additive gradient.png --freq-min 55 --freq-max 880  # Musical range
      img2sound additive photo.jpg --oscillators 128  # More detail
    """
    try:
        input_type = get_input_type(input)

        if output is None:
            output = str(generate_output_path(input, mode="additive"))

        if verbose:
            click.echo(f"Input: {input} ({input_type})")
            click.echo(f"Mode: additive")
            click.echo(f"Frequency range: {freq_min}-{freq_max} Hz")
            click.echo(f"Log frequency: {log_freq}")
            click.echo(f"Oscillators: {oscillators}")
            click.echo(f"Sample rate: {sample_rate} Hz")

        if input_type == "image":
            image = load_image(input)
            validate_image(image)

            if verbose:
                h, w = image.shape[:2]
                click.echo(f"Image size: {w}x{h}")
                click.echo(f"Target duration: {duration}s")

            audio = additive_image(
                image,
                duration=duration,
                sample_rate=sample_rate,
                freq_min=freq_min,
                freq_max=freq_max,
                log_freq=log_freq,
                n_oscillators=oscillators,
            )
        else:
            info = get_video_info(input)
            if verbose:
                click.echo(f"Video: {info['width']}x{info['height']} @ {info['fps']:.2f} fps")
                click.echo(f"Duration: {format_duration(info['duration'])}")
                click.echo(f"Frames: {info['frame_count']}")

            progress = ProgressBar(info["frame_count"])

            audio = additive_video(
                input,
                sample_rate=sample_rate,
                freq_min=freq_min,
                freq_max=freq_max,
                log_freq=log_freq,
                n_oscillators=oscillators,
                progress_callback=progress.update,
            )

        audio = normalize_audio(audio)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(str(output_path), audio, sample_rate, int(bit_depth))

        # Write NFO file with settings
        output_duration = len(audio) / sample_rate
        settings = {
            "mode": "additive",
            "input": input,
            "input_type": input_type,
            "output": str(output_path),
            "sample_rate": f"{sample_rate} Hz",
            "bit_depth": bit_depth,
            "duration": f"{output_duration:.2f}s",
            "freq_min": f"{freq_min} Hz",
            "freq_max": f"{freq_max} Hz",
            "log_freq": log_freq,
            "oscillators": oscillators,
        }
        write_nfo(str(output_path), settings)

        click.echo(f"Written {output_path} ({format_duration(output_duration)})")

    except Img2SoundError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Processing failed: {e}")


if __name__ == "__main__":
    cli()
