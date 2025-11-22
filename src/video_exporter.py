
"""video_exporter

Unified video generation utilities:
 - Beat-synced random clip montage driven by kick timestamps.
 - Optional initial offset section.
 - Lyric word overlay using precomputed word_timings (start/end seconds relative to audio start).
 - Final audio mux via ffmpeg.

Expose a single function generate_beat_synced_lyrics_video(...) for use in main.
"""

from typing import List, Dict, Optional, Sequence, Tuple
import cv2
import numpy as np
import math
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont
import os
import random

def generate_beat_synced_lyrics_video(
    audio_file: str,
    video_folder: str,
    output_final: str,
    word_timings: Sequence[Dict],
    kick_timestamps: Sequence[float],
    sr: int,
    total_audio_samples: int,
    audio_offset: float = 0.0,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    flash_effect: bool = True,
    font_path: Optional[str] = None,
    font_size_px: int = 120,
    hershey_scale: float = 2.5,
    font_color: Tuple[int,int,int] = (255,255,255),
    stroke_color: Tuple[int,int,int] = (0,0,0),
    font_thickness: int = 3,
    stroke_thickness: int = 6,
    lyric_time_shift: float = 0.0,
    random_seed: Optional[int] = None,
    temp_dir: Optional[str] = None,
    codec: str = "mp4v"
) -> str:
    """Generate a beat-synced montage video with lyric overlays.

    Parameters
    ----------
    audio_file : Path to source audio (final muxed track).
    video_folder : Folder containing source videos (mp4/mov/avi/mkv).
    output_final : Path for final muxed video (mp4).
    word_timings : List of dicts {word,start,end} relative to audio start (seconds).
    kick_timestamps : Iterable of kick times (seconds) relative to audio start.
    sr : Sample rate of original audio (for computing end duration).
    total_audio_samples : Total samples in the full track (len(y)).
    audio_offset : Seconds of offset frames to prepend (video lead-in before audio words appear).
    fps : Target frames per second.
    width,height : Output resolution.
    flash_effect : If True, apply flash on first few frames after each kick.
    font_path : Optional .ttf/.otf font; if None uses OpenCV Hershey.
    font_size_px : Font size when using PIL TrueType.
    hershey_scale : Scale when using Hershey font.
    font_color, stroke_color : RGB colors for text and stroke.
    font_thickness, stroke_thickness : Thickness values.
    lyric_time_shift : Shift applied to lyric timestamps (positive delays lyrics, negative advances).
    random_seed : Seed for reproducible video selection.
    temp_dir : Optional directory for temp file placement.
    codec : FourCC codec (e.g. 'mp4v', 'avc1').

    Returns
    -------
    Path to final muxed video file.
    """

    if random_seed is not None:
        random.seed(random_seed)

    video_folder_path = Path(video_folder)
    if not video_folder_path.is_dir():
        raise FileNotFoundError(f"Video folder not found: {video_folder}")

    # Collect video files
    video_files = [str(p) for p in video_folder_path.iterdir() if p.suffix.lower() in {'.mp4','.mov','.avi','.mkv'}]
    if not video_files:
        raise RuntimeError(f"No video files found in {video_folder}")
    print(f"Found {len(video_files)} video files")

    kick_ts = list(kick_timestamps)
    print(f"Found {len(kick_ts)} kicks")

    # Prepare temp paths
    temp_dir_path = Path(temp_dir) if temp_dir else Path(output_final).resolve().parent
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    temp_video_path = temp_dir_path / (Path(output_final).stem + "_temp.mp4")

    fourcc = cv2.VideoWriter.fourcc(*codec)
    writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    # Font pipeline choice
    use_pil_font = bool(font_path) and Path(font_path).exists()
    if font_path and not use_pil_font:
        print(f"Warning: font_path not found: {font_path}. Falling back to Hershey font.")

    def draw_centered_text(img: np.ndarray, text: str):
        if not text:
            return
        if not use_pil_font:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size, _ = cv2.getTextSize(text, font, hershey_scale, font_thickness)
            tw, th = text_size
            x = (width - tw) // 2
            y = (height + th) // 2
            if stroke_thickness > 0:
                cv2.putText(img, text, (x, y), font, hershey_scale, stroke_color, stroke_thickness, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, hershey_scale, font_color, font_thickness, cv2.LINE_AA)
        else:
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            try:
                fnt = ImageFont.truetype(str(font_path), font_size_px)
            except Exception as e:
                print(f"Failed to load TrueType font '{font_path}': {e}. Using Hershey.")
                return draw_centered_text(img, text)  # retry via Hershey
            bbox = draw.textbbox((0, 0), text, font=fnt)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (width - tw) // 2
            y = (height - th) // 2
            if stroke_thickness > 0:
                r = max(1, stroke_thickness // 2)
                for dx, dy in [(-r,0),(r,0),(0,-r),(0,r),(-r,-r),(-r,r),(r,-r),(r,r)]:
                    draw.text((x+dx, y+dy), text, font=fnt, fill=stroke_color)
            draw.text((x, y), text, font=fnt, fill=font_color)
            img[:] = np.array(pil_img)

    # Sanitize word timings
    wt = []
    for w in word_timings:
        try:
            s = float(w.get('start', 0.0))
            e = float(w.get('end', s))
            if e < s:  # guard
                e = s + 0.05
            wt.append({'word': str(w.get('word','')), 'start': s + lyric_time_shift, 'end': e + lyric_time_shift})
        except Exception:
            continue
    if not wt:
        print("Warning: empty word timings; lyrics overlay disabled.")
    else:
        print(f"Loaded {len(wt)} word timings (shift={lyric_time_shift:.2f}s)")
        print("Word timings preview:")
        for i, w in enumerate(wt[:10]):  # show first 10
            print(f"  {i}: '{w['word']}' {w['start']:.3f}s - {w['end']:.3f}s")

    # Helper to get active word for given audio time
    def active_word(audio_t: float) -> Optional[str]:
        if not wt:
            return None
        # Search for the word whose time range contains audio_t
        # Handle zero-duration words by checking if we're at the exact timestamp
        for w in wt:
            if w['start'] == w['end']:  # zero-duration word
                # Display if we're within a small window around the timestamp
                if abs(audio_t - w['start']) < 0.05:  # 50ms window
                    return w['word']
            elif w['start'] <= audio_t < w['end']:
                return w['word']
        return None

    print("Generating video...")

    total_audio_duration = total_audio_samples / sr
    previous_video_path = None
    total_frames_written = 0

    # Prepend offset frames (no lyrics) if audio_offset > 0
    if audio_offset > 0:
        offset_frames = int(audio_offset * fps)
        offset_video = random.choice(video_files)
        cap_offset = cv2.VideoCapture(offset_video)
        print(f"Adding offset lead-in: {offset_frames} frames ({audio_offset:.2f}s) from {Path(offset_video).name}")
        for i in range(offset_frames):
            ret, frame = cap_offset.read()
            if not ret:
                cap_offset.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap_offset.read()
            frame = cv2.resize(frame, (width, height))
            # overlay nothing (audio time negative)
            writer.write(frame)
            total_frames_written += 1
        cap_offset.release()
        print("Offset section complete")

    # Process each kick segment
    for i, kt in enumerate(kick_ts):
        if i < len(kick_ts) - 1:
            segment_end_t = kick_ts[i + 1]
        else:
            segment_end_t = total_audio_duration
        duration = max(0.0, segment_end_t - kt)
        target_total_frames = int((audio_offset + segment_end_t) * fps)
        frames_needed = target_total_frames - total_frames_written
        if frames_needed <= 0:
            continue

        # choose clip
        if previous_video_path is None:
            video_path = random.choice(video_files)
        else:
            avail = [v for v in video_files if v != previous_video_path] or video_files
            video_path = random.choice(avail)
        previous_video_path = video_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: failed to open {video_path}, skipping segment {i}")
            continue
        print(f"Segment {i+1}/{len(kick_ts)}: {frames_needed} frames from {Path(video_path).name}")

        frame_idx_seg = 0
        while frame_idx_seg < frames_needed:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: empty video {video_path}")
                    break
            frame = cv2.resize(frame, (width, height))
            # flash effect for first few frames of segment
            if flash_effect and frame_idx_seg < 5:
                flash_intensity = int(50 * (1 - frame_idx_seg / 5))
                frame = cv2.add(frame, np.full(frame.shape, flash_intensity, dtype=np.uint8))
            # Determine audio time corresponding to this frame
            video_time = total_frames_written / fps
            audio_time = video_time - audio_offset  # lyrics anchored to audio start
            wtxt = active_word(audio_time)
            if wtxt:
                draw_centered_text(frame, wtxt)
            writer.write(frame)
            total_frames_written += 1
            frame_idx_seg += 1
        cap.release()

    writer.release()
    print(f"Video frames written to {temp_video_path}")
    expected_frames = int((audio_offset + total_audio_duration) * fps)
    print(f"Total frames: {total_frames_written} (expected ~{expected_frames})")

    # Mux audio
    print("Muxing audio with ffmpeg...")
    cmd = [
        'ffmpeg','-y',
        '-i', str(temp_video_path),
        '-i', str(audio_file),
        '-c:v','copy',
        '-c:a','aac',
        '-shortest',
        str(output_final)
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Final video saved to {output_final}")
        # Remove temporary video file after successful mux
        try:
            if temp_video_path.exists():
                temp_video_path.unlink()
                print(f"Cleaned up temp file: {temp_video_path}")
        except Exception as cleanup_err:
            print(f"Warning: failed to remove temp file {temp_video_path}: {cleanup_err}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH. Install FFmpeg.")

    return str(output_final)


__all__ = ["generate_beat_synced_lyrics_video"]
