import lyrics_transcriber as lt
import beat_timer as bt
import video_exporter as ve
from pathlib import Path

song_path = Path(r"E:\Documents\CodeStuff\BLAiRE\src\shoobie.wav")  # audio file path
lyrics_text = """I spend my life\ndoing anything you like\ncome on, and love me like you used to""".strip()

bpm = 101
kick_percentile_threshold = 95  # Percentile for low-band flux peaks (0-100)
video_switch_frequency = 4      # Higher value -> more frequent switching (fraction of beat)

video_folder = Path(__file__).resolve().parent / "video"  # folder of source clips
# Export final video to project root /out directory
project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "out"
output_final = output_dir / "output_with_audio.mp4"  # final muxed output path

# Video settings
fps = 30
width, height = 1920, 1080
# Video effects
flash_effect = True
enable_lyrics = True

def main():
    print("=== BLAiRE Lyric-Synced Video Generator ===")
    if not song_path.is_file():
        print(f"[ERROR] Audio file not found: {song_path}")
        return
    if not video_folder.is_dir():
        print(f"[ERROR] Video folder not found: {video_folder}")
        return
    # Ensure output directory exists
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Info] Created output directory: {output_dir}")
        except Exception as e:
            print(f"[ERROR] Could not create output directory '{output_dir}': {e}")
            return

    # 1. Align lyrics FIRST (single pass, no persistence)
    if enable_lyrics:
        print("[Step 1] Aligning lyrics with WhisperX...")
        word_timings = lt.align_lyrics_to_audio(str(song_path), lyrics_text, lt.use_vocal_separation, reuse_models=True)
    else:
        print("[Info] Lyrics overlay disabled.")
        word_timings = []

    # 2. Process song for flux and trim silence
    print("[Step 2] Processing audio for beat/kick detection...")
    processed = bt.process_song(str(song_path))
    import librosa
    y, sr = librosa.load(song_path, sr=None)
    total_audio_samples = len(y)
    audio_start_offset = processed.get('audio_start_offset', 0.0)

    # 3. Detect kicks
    print("[Step 3] Detecting kicks...")
    processed['sr'] = sr
    kick_times, kick_binary = bt.detect_kicks(
        processed,
        bpm=bpm,
        kick_percentile_threshold=kick_percentile_threshold,
        video_switch_frequency=video_switch_frequency
    )
    if len(kick_times) == 0:
        print("[WARN] No kicks detected; using single segment.")
        kick_times = [0.0]

    # 4. Generate video
    print("[Step 4] Generating beat-synced video with lyrics overlay...")
    try:
        ve.generate_beat_synced_lyrics_video(
            audio_file=str(song_path),
            video_folder=str(video_folder),
            output_final=str(output_final),  # already a Path; cast to str for exporter
            word_timings=word_timings,
            kick_timestamps=kick_times,
            sr=int(sr),
            total_audio_samples=total_audio_samples,
            audio_offset=audio_start_offset,
            fps=fps,
            width=width,
            height=height,
            flash_effect=flash_effect,
            font_path=str(Path(__file__).resolve().parent / 'KGRedHands.ttf'),
            font_size_px=120,
            hershey_scale=2.5,
            lyric_time_shift=0.0,
            random_seed=42
        )
    except Exception as e:
        print(f"[ERROR] Video generation failed: {e}")
        return

    print("=== Done ===")


def align_words():
    """Run alignment once and print timings (no caching)."""
    wt = lt.align_lyrics_to_audio(str(song_path), lyrics_text, lt.use_vocal_separation, reuse_models=True)
    print("\nYour lyrics with timings:")
    for i, word_data in enumerate(wt):
        print(f"{i}: '{word_data['word']}' - {word_data['start']:.2f}s to {word_data['end']:.2f}s")
    return wt

if __name__ == "__main__":
    main()