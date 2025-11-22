import tempfile
import os
import whisperx
import re
import difflib
from pathlib import Path

# Install: pip install demucs
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Options
use_vocal_separation = True
whisperx_model_size = "base.en"
device = "cpu"
short_word_bias = 0.175  # 0.0 = no bias (pure ASR durations); higher shrinks short words / extends long words within phrase
min_word_length = 0.5  # minimum duration for any word in seconds
_CACHED_ASR_MODEL = None
_CACHED_ALIGN_MODEL = None
_CACHED_ALIGN_METADATA = None
_CACHED_MODEL_SIZE = None

def _get_asr_model(model_size: str, device: str):
    global _CACHED_ASR_MODEL, _CACHED_MODEL_SIZE
    if _CACHED_ASR_MODEL is None or _CACHED_MODEL_SIZE != model_size:
        print(f"[Cache] Loading ASR model '{model_size}'...")
        _CACHED_ASR_MODEL = whisperx.load_model(model_size, device, compute_type="float32")
        _CACHED_MODEL_SIZE = model_size
    else:
        print(f"[Cache] Reusing ASR model '{model_size}'.")
    return _CACHED_ASR_MODEL

def _get_align_model(device: str):
    global _CACHED_ALIGN_MODEL, _CACHED_ALIGN_METADATA
    if _CACHED_ALIGN_MODEL is None:
        print("[Cache] Loading alignment model (en)...")
        _CACHED_ALIGN_MODEL, _CACHED_ALIGN_METADATA = whisperx.load_align_model(language_code="en", device=device)
    else:
        print("[Cache] Reusing alignment model.")
    return _CACHED_ALIGN_MODEL, _CACHED_ALIGN_METADATA
# Alignment tuning knobs
min_word_duration = 0.15  # enforce a floor so words never collapse to 0 and remain visible (reduced from 0.25 for tighter fit)
similarity_primary_threshold = 0.60  # required similarity between lyric word and concatenated ASR tokens to accept a group
similarity_fallback_threshold = 0.55  # lower threshold used after expansion attempts fail
max_merge_factor = 1.75  # do not merge ASR tokens if merged length exceeds lyric length * this factor (avoid overswallowing)
final_word_stretch_fraction = 0.15  # allow last word in phrase to stretch leftover slack up to this fraction of phrase duration


def _verify_audio(path_str: str):
    p = Path(path_str)
    if not p.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {p.resolve()}\n"
            "Ensure the WAV exists and path is correct."
        )
    size = p.stat().st_size
    if size < 1024:
        raise RuntimeError(
            f"Audio file is too small ({size} bytes): {p.resolve()} - likely corrupt."
        )


def extract_vocals(audio_path):
    """Separate vocals using Demucs (no DLL issues on Windows)"""
    _verify_audio(audio_path)
    print("Separating vocals from music with Demucs...")
    
    # Load model
    model = get_model('htdemucs')
    model.cpu()
    model.eval()
    
    # Load audio
    try:
        wav, sr = torchaudio.load(audio_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load audio '{audio_path}' with torchaudio: {e}\n"
            "Verify the file is a valid PCM WAV. Try converting:\n"
            "  ffmpeg -y -i input.wav -ac 2 -ar 44100 -vn -c:a pcm_s16le conviction.wav"
        ) from e
    
    # Apply separation
    with torch.no_grad():
        sources = apply_model(model, wav[None], device='cpu')[0]
    
    # Extract vocals (index 3)
    vocals = sources[3]
    
    # Save to temp file - FIXED: Use NamedTemporaryFile instead of mktemp
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        temp_vocals = tmp.name
    
    torchaudio.save(temp_vocals, vocals, sr)

    
    print(f"Vocals extracted to: {temp_vocals}")
    return temp_vocals


def _tokenize_words(text: str):
    # keep letters and apostrophes for words like it's, you're
    return re.findall(r"[A-Za-z']+", text.lower())


def _simple_map_lyrics_to_asr(lyrics_text: str, asr_words: list):
    """
    Simple approach: Extract all lyric words, map them 1:1 to ASR timestamps.
    If more lyrics than ASR words, extend timeline. If fewer, use subset of ASR.
    Returns list of {word, start, end} for YOUR lyrics with ASR timing.
    Makes lyrics continuous (no gaps) by adjusting end times.
    """
    lyric_words = _tokenize_words(lyrics_text)
    if not lyric_words:
        return []
    if not asr_words:
        # No ASR, make synthetic timing
        return [{"word": w, "start": i * 0.5, "end": (i + 1) * 0.5} for i, w in enumerate(lyric_words)]
    
    n_lyrics = len(lyric_words)
    n_asr = len(asr_words)
    
    timings = []
    
    if n_lyrics <= n_asr:
        # Map each lyric word to an ASR timestamp
        for i, lyric in enumerate(lyric_words):
            asr = asr_words[i]
            timings.append({
                "word": lyric,
                "start": float(asr['start']),
                "end": float(asr['end'])
            })
    else:
        # More lyrics than ASR words - need to extend
        # Map first n_asr lyrics to ASR words
        for i in range(n_asr):
            asr = asr_words[i]
            timings.append({
                "word": lyric_words[i],
                "start": float(asr['start']),
                "end": float(asr['end'])
            })
        # Extend remaining lyrics after last ASR word
        last_end = float(asr_words[-1]['end'])
        avg_duration = sum(float(w['end']) - float(w['start']) for w in asr_words) / len(asr_words)
        avg_duration = max(0.3, avg_duration)  # at least 0.3s per word
        
        for i in range(n_asr, n_lyrics):
            start = last_end
            end = start + avg_duration
            timings.append({
                "word": lyric_words[i],
                "start": start,
                "end": end
            })
            last_end = end
    
    # Make continuous: adjust each word's end to match next word's start
    for i in range(len(timings) - 1):
        # If there's a gap, extend current word to fill it
        if timings[i]['end'] < timings[i + 1]['start']:
            timings[i]['end'] = timings[i + 1]['start']
        # If there's overlap, trim current word
        elif timings[i]['end'] > timings[i + 1]['start']:
            timings[i]['end'] = timings[i + 1]['start']
    
    # Ensure no word has zero or negative duration
    for w in timings:
        if w['end'] <= w['start']:
            w['end'] = w['start'] + 0.1
    
    return timings


def _split_lyrics_into_phrases(lyrics_text: str):
    # Split by lines; drop empties
    lines = [ln.strip() for ln in lyrics_text.splitlines()]
    lines = [ln for ln in lines if ln]
    phrases = []
    for ln in lines:
        words = _tokenize_words(ln)
        if words:
            phrases.append({"text": ln, "words": words})
    return phrases


def _group_asr_into_phrases(whisperx_words, gap_threshold: float = 0.1):
    # Group contiguous ASR words separated by gaps > threshold
    phrases = []
    if not whisperx_words:
        return phrases
    current = {"words": [], "start": whisperx_words[0]['start'], "end": whisperx_words[0]['end']}
    for w in whisperx_words:
        if not current["words"]:
            current["words"].append(w)
            current["start"] = w["start"]
            current["end"] = w["end"]
            continue
        gap = w["start"] - current["words"][-1]["end"]
        if gap > gap_threshold:
            phrases.append(current)
            current = {"words": [w], "start": w["start"], "end": w["end"]}
        else:
            current["words"].append(w)
            current["end"] = w["end"]
    if current["words"]:
        phrases.append(current)
    return phrases


def _norm(s: str) -> str:
    return re.sub(r"[^a-z']+", "", s.lower())


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _segment_asr_groups(lyric_words, asr_words_in_phrase):
    """Map lyric words to ASR tokens using fuzzy grouping with safeguards.

    Strategy:
    - Start at current ASR index, attempt to match lyric word by progressively concatenating ASR tokens.
    - Evaluate similarity; stop expanding when similarity decreases meaningfully or length exceeds max_merge_factor.
    - Accept group if similarity >= primary threshold; else if expansion stalls, accept best group if it passes fallback threshold.
    - If no acceptable match, assign a minimal synthetic duration after previous word.
    - Extend last group's end to consume any trailing ASR tokens for phrase coverage.
    Returns list of dicts: {word, base_start, base_end, len_norm, sim}
    """
    groups = []
    awords = asr_words_in_phrase
    n = len(awords)
    idx = 0
    prev_end = awords[0]['start'] if awords else 0.0
    for lyric in lyric_words:
        target = _norm(lyric)
        if idx >= n:
            # allocate synthetic minimal slot
            synthetic_start = prev_end
            synthetic_end = synthetic_start + min_word_duration
            groups.append({"word": lyric, "base_start": synthetic_start, "base_end": synthetic_end, "len_norm": len(target) or 1, "sim": 0.0})
            prev_end = synthetic_end
            continue
        best_sim = -1.0
        best_end = idx
        concat = ""
        # progressive expansion
        for j in range(idx, n):
            concat += _norm(awords[j]['word'])
            length_ratio = len(concat) / max(1, len(target))
            sim = _sim(concat, target) if target else 1.0
            # track best
            if sim > best_sim:
                best_sim = sim
                best_end = j
            # stop if overly long without improving
            if length_ratio > max_merge_factor and sim < similarity_primary_threshold:
                break
            # early stop if similarity dips significantly after good match
            if best_sim >= similarity_primary_threshold and sim < best_sim - 0.08:
                break
        start_time = float(awords[idx]['start'])
        end_time = float(awords[best_end]['end'])
        accept = best_sim >= similarity_primary_threshold or (best_sim >= similarity_fallback_threshold and len(target) >= 4)
        if not accept:
            # fallback synthetic minimal slot
            start_time = prev_end
            end_time = start_time + min_word_duration
            best_sim = 0.0
        groups.append({"word": lyric, "base_start": start_time, "base_end": end_time, "len_norm": len(target) or 1, "sim": best_sim})
        prev_end = end_time
        if accept:
            idx = best_end + 1
    # extend last real group to cover remaining ASR tokens
    if idx < n and groups:
        groups[-1]['base_end'] = float(awords[-1]['end'])
    return groups


def _apply_duration_bias(groups, phrase_start, phrase_end, bias: float):
    """
    Adjust durations per lyric word around their ASR-based base durations using a
    length-based bias. Preserves total phrase duration and ordering.
    bias in [0,1]: 0 = no change; higher = stronger long>short emphasis.
    """
    if not groups:
        return []
    # Base durations from ASR grouping
    base_durs = [max(min_word_duration, g['base_end'] - g['base_start']) for g in groups]
    total_base = sum(base_durs)
    if total_base <= 1e-6:
        # fallback: equal tiny splits
        step = (phrase_end - phrase_start) / len(groups)
        t = phrase_start
        out = []
        for g in groups:
            out.append({"word": g['word'], "start": t, "end": t + step})
            t += step
        out[-1]['end'] = phrase_end
        return out

    # Compute weights based on word length relative to mean
    mean_len = max(1.0, sum(g['len_norm'] for g in groups) / len(groups))
    weights = []
    for g in groups:
        ln = max(1.0, float(g['len_norm']))
        rel = ln / mean_len
        # weight factor: 1 blended toward rel**gamma by bias
        gamma = 1.0 + 1.0 * bias
        w = (1 - bias) + bias * (rel ** gamma)
        weights.append(w)

    # Apply weights to base durations, then renormalize to keep total the same
    adjusted = [d * w for d, w in zip(base_durs, weights)]
    sum_adj = sum(adjusted)
    if sum_adj <= 1e-6:
        adjusted = base_durs[:]  # fallback
        sum_adj = total_base
    scale = total_base / sum_adj
    adjusted = [a * scale for a in adjusted]

    # Lay out sequentially within the phrase window, preserving total window
    # Align the start to min(base_starts, phrase_start) for stability
    t = max(phrase_start, min(g['base_start'] for g in groups))
    # If the first ASR starts after phrase_start, use that; otherwise phrase_start
    t = phrase_start
    out = []
    for i, (g, dur) in enumerate(zip(groups, adjusted)):
        start = t
        # Calculate desired end
        end = start + dur
        # Enforce minimum duration BEFORE clamping to phrase_end
        if end - start < min_word_duration:
            end = start + min_word_duration
        # Only clamp if we're at the last word
        if i == len(groups) - 1:
            end = phrase_end
        out.append({"word": g['word'], "start": start, "end": max(start + 0.01, end)})  # guarantee non-zero
        t = end
    
    # Final safeguard: ensure no word has zero duration
    for w in out:
        if w['end'] <= w['start']:
            w['end'] = w['start'] + min_word_length
    
    return out


def _segment_asr_to_lyrics_with_bias(lyrics_phrases, asr_phrases, bias: float):
    timings = []
    pairs = min(len(lyrics_phrases), len(asr_phrases))
    for i in range(pairs):
        lp = lyrics_phrases[i]
        ap = asr_phrases[i]
        lwords = lp["words"]
        awords = ap["words"]
        s, e = float(ap['start']), float(ap['end'])
        if not lwords or e <= s:
            continue
        groups = _segment_asr_groups(lwords, awords)
        timings.extend(_apply_duration_bias(groups, s, e, bias))
    # handle extra lyric phrases if any
    if len(lyrics_phrases) > pairs:
        t = timings[-1]['end'] if timings else (asr_phrases[-1]['end'] if asr_phrases else 0.0)
        for i in range(pairs, len(lyrics_phrases)):
            for w in lyrics_phrases[i]['words']:
                end = t + 0.3
                timings.append({"word": w, "start": t, "end": end})
                t = end
    return timings


def align_lyrics_to_audio(audio_path, lyrics_text, use_vocal_separation=True, reuse_models: bool = True):
    """
    Align YOUR lyrics to the audio timing using WhisperX: first group ASR tokens to lyric
    words (preserving ASR-based durations), then apply a short-word bias that slightly
    reduces durations of short/function words and increases durations of longer words,
    while preserving each phrase's total duration.

    Returns a list of word timings for YOUR lyrics.
    """
    # Preflight
    _verify_audio(audio_path)
    # Step 1: Extract vocals if enabled
    if use_vocal_separation:
        try:
            audio_to_use = extract_vocals(audio_path)
        except Exception as e:
            print(f"[Warning] Vocal separation failed: {e}. Falling back to original audio.")
            audio_to_use = audio_path
    else:
        audio_to_use = audio_path
    
    # Step 2: Load WhisperX model
    if reuse_models:
        model = _get_asr_model(whisperx_model_size, device)
    else:
        print(f"Loading WhisperX model ({whisperx_model_size}) (no cache)...")
        model = whisperx.load_model(whisperx_model_size, device, compute_type="float32")
    
    # Step 3: Load audio
    audio = whisperx.load_audio(audio_to_use)
    
    # Step 4: Transcribe to get initial segments (disable VAD for music)
    print("Transcribing audio...")
    result = model.transcribe(audio, batch_size=16, language="en")
    
    # Step 5: Load alignment model for word-level timestamps
    if reuse_models:
        model_a, metadata = _get_align_model(device=device)
    else:
        print("Loading alignment model (no cache)...")
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    
    # Step 6: Align to get precise word timings
    print("Aligning words...")
    result_aligned = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio, 
        device,
        return_char_alignments=False
    )
    
    # Step 7: Extract word timings from WhisperX
    whisperx_words = []
    for segment in result_aligned.get("segments", []):
        for word_info in segment.get("words", []):
            if word_info.get('word') is None:
                continue
            whisperx_words.append({
                'word': word_info['word'].strip().lower(),
                'start': float(word_info['start']),
                'end': float(word_info['end'])
            })
    print(f"Extracted {len(whisperx_words)} words from WhisperX")
    # Optional debug (first 50)
    for idx, w in enumerate(whisperx_words[:50]):
        print(f"{idx}: '{w['word']}' - {w['start']:.2f}s to {w['end']:.2f}s")

    # Step 8: Simple direct mapping (no complex fuzzy matching)
    print("\n[Simple Mode] Mapping your lyrics directly to ASR timestamps...")
    word_timings = _simple_map_lyrics_to_asr(lyrics_text, whisperx_words)

    print(f"\nMapped {len(word_timings)} lyric words to ASR timing (simple 1:1 mapping).")
    return word_timings

