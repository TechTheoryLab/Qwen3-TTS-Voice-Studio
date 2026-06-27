import os
import sys
import json
import torch
import soundfile as sf
import gradio as gr
from qwen_tts import Qwen3TTSModel
import shutil
import datetime
import re
import numpy as np
import librosa
import whisper
import gc
import subprocess
import tempfile
from deep_translator import GoogleTranslator

# =========================================================================
# 🕵️ SMART ENVIRONMENT DETECTOR & SETUP
# =========================================================================
IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
SAVE_MODELS_TO_DRIVE = False

if IS_COLAB:
    print("🌐 Environment Detected: Google Colab")
    if SAVE_MODELS_TO_DRIVE:
        os.environ["HF_HOME"] = "/content/drive/MyDrive/Ai/qwen_models"
    else:
        os.environ.pop("HF_HOME", None)
    base_output_path = "/content/drive/MyDrive/Ai/output/voice"
else:
    print("💻 Environment Detected: Local PC / Hugging Face Spaces")
    os.environ.pop("HF_HOME", None)
    base_output_path = "./TechTheoryLab_Output/voice"

voice_folder = base_output_path
presets_master_folder = os.path.join(voice_folder, "Presets")
clone_audio_folder = os.path.join(presets_master_folder, "clone_audio")

os.makedirs(voice_folder, exist_ok=True)
os.makedirs(presets_master_folder, exist_ok=True)
os.makedirs(clone_audio_folder, exist_ok=True)

presets_file = os.path.join(presets_master_folder, "voice_presets.json")
clone_presets_file = os.path.join(presets_master_folder, "clone_presets.json")

FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"

# =========================================================================
# 🧠 VRAM TELEMETRY & AUTO-SWAPPER (OPTIMIZED)
# =========================================================================
model_base = None
model_design = None
current_model_loaded = None

def get_vram_status():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated(0) / 1e9
        r = torch.cuda.memory_reserved(0) / 1e9
        return f"🟢 GPU VRAM: {a:.2f} GB Used / {r:.2f} GB Reserved"
    return "🖥️ Running on CPU (No VRAM)"

def manual_clear_vram():
    global model_base, model_design, current_model_loaded
    if model_base is not None: del model_base
    if model_design is not None: del model_design
    model_base = None
    model_design = None
    current_model_loaded = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return f"🧹 GPU Memory Cleared! Ready to load fresh models.\n{get_vram_status()}"

def _safe_from_pretrained(model_id, device, dtype):
    try:
        return Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype, attn_implementation="sdpa")
    except Exception as e:
        if "cuda" in str(device).lower() and dtype == torch.bfloat16:
            try:
                return Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=torch.float16, attn_implementation="sdpa")
            except Exception:
                pass
        return Qwen3TTSModel.from_pretrained(model_id, device_map="cpu")

def switch_vram_model(target):
    global model_base, model_design, current_model_loaded
    if current_model_loaded == target: return

    print(f"🔄 VRAM Manager: Loading Qwen3 {target.upper()}...")
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if target == "base":
            if model_design is not None: del model_design
            global model_design; model_design = None
            gc.collect(); 
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            model_base = _safe_from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device, dtype)
        elif target == "design":
            if model_base is not None: del model_base
            global model_base; model_base = None
            gc.collect(); 
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            model_design = _safe_from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device, dtype)

        current_model_loaded = target
    except Exception as e:
        current_model_loaded = None
        raise e

# Deferred model loading: We removed the immediate `switch_vram_model` call here so the app opens instantly.
print("Loading Whisper for Subtitles...")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

# =========================================================================
# 🛠️ PRO-FINANCE PRE-PROCESSOR & HELPERS
# =========================================================================
QWEN_LANGUAGES = ["Auto", "English", "Chinese", "Japanese", "Korean", "French", "German", "Spanish"]

def master_text_processor(text, lexicon_data):
    if not text: return text
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*trillion', r'\1 trillion dollars', text, flags=re.IGNORECASE)
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*billion', r'\1 billion dollars', text, flags=re.IGNORECASE)
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*million', r'\1 million dollars', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\.(\d+)%', r'\1 point \2 percent', text)
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    
    if lexicon_data:
        for row in lexicon_data:
            if len(row) >= 2 and str(row[0]).strip() and str(row[1]).strip():
                t = str(row[0]).strip()
                r = str(row[1]).strip()
                text = re.sub(r'\b' + re.escape(t) + r'\b', r, text, flags=re.IGNORECASE)
    return text

def split_text_into_chunks(text):
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\be\.g)(?<!\bi\.e)(?<=[.!?]) +', text)
    valid_chunks = [s.strip() + ("." if not s.strip()[-1] in ".!?" else "") for s in sentences if len(s.strip()) > 0]
    return valid_chunks

def preview_chunks(text):
    if not text.strip(): return gr.update(visible=True, value="⚠️ Please enter a script first.")
    chunks = split_text_into_chunks(text)
    return gr.update(visible=True, value="\n\n---\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(chunks)]))

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = round((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def write_srt_file(segments, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            f.write(f"{i+1}\n{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n{segment['text'].strip()}\n\n")

def mix_background_music(voice_y, voice_sr, bgm_path, bgm_vol):
    bgm_y, bgm_sr = librosa.load(bgm_path, sr=voice_sr)
    if len(bgm_y) < len(voice_y): bgm_y = np.tile(bgm_y, int(np.ceil(len(voice_y) / len(bgm_y))))
    mixed = voice_y + (bgm_y[:len(voice_y)] * bgm_vol)
    max_amp = np.max(np.abs(mixed))
    if max_amp > 1.0: mixed = mixed / max_amp * 0.95
    return mixed

def convert_audio_format(wav_file, target_format):
    target = target_format.lower().strip()
    if target == "wav": return wav_file
    out_file = wav_file.rsplit('.', 1)[0] + f".{target}"
    try:
        subprocess.run([FFMPEG_BIN, "-y", "-i", wav_file, "-q:a", "0", out_file], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(wav_file) and os.path.exists(out_file):
            os.remove(wav_file)
        return out_file
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}. Falling back to WAV.")
        return wav_file

def process_audio_studio(audio_path, do_norm, do_trim, speed, do_srt, bgm_path, bgm_vol, out_format="WAV", progress=gr.Progress()):
    if not audio_path: return None, None, "⚠️ Please upload an audio file."
    try:
        status = []
        progress(0.1, desc="Loading Audio...")
        y, sr = librosa.load(audio_path, sr=None)

        if do_trim:
            y, _ = librosa.effects.trim(y, top_db=30)
            status.append("✂️ Silences trimmed.")
        
        progress(0.3, desc="Applying FX...")
        if speed != 1.0:
            y = librosa.effects.time_stretch(y, rate=speed)
            status.append(f"⏱️ Speed changed to {speed}x.")
        
        if bgm_path:
            y = mix_background_music(y, sr, bgm_path, bgm_vol)
            status.append("🎵 Cinematic Background Music mixed in.")

        if do_norm:
            max_amp = np.max(np.abs(y))
            if max_amp > 0: y = y / max_amp * 0.95
            status.append("🔊 Volume normalized.")

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_folder = os.path.join(voice_folder, "Polished")
        os.makedirs(out_folder, exist_ok=True)
        temp_wav_path = os.path.join(out_folder, f"{base_name}_Polished.wav")
        sf.write(temp_wav_path, y, sr)

        files_to_download = []
        if do_srt:
            progress(0.7, desc="Generating SRT Captions...")
            result = whisper_model.transcribe(temp_wav_path)
            srt_path = os.path.join(out_folder, f"{base_name}_Captions.srt")
            write_srt_file(result["segments"], srt_path)
            files_to_download.append(srt_path)
            status.append("✅ SRT generated successfully.")

        progress(0.9, desc="Converting Format...")
        final_export_path = convert_audio_format(temp_wav_path, out_format)
        files_to_download.insert(0, final_export_path)

        progress(1.0, desc="Done!")
        return final_export_path, files_to_download, "\n".join(status)
    except Exception as e: return None, None, f"❌ Error: {str(e)}"

# =========================================================================
# 💾 ATOMIC PRESET LOGIC (Prevents File Corruption)
# =========================================================================
def atomic_write_json(path, data):
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise e

def load_presets():
    if os.path.exists(presets_file):
        with open(presets_file, "r") as f: return json.load(f)
    return {}

def save_preset_to_drive(name, desc, lang):
    name = name.strip()
    if not name: return gr.update(), "⚠️ Provide a preset name."
    presets = load_presets()
    presets[name] = {"desc": desc, "lang": lang}
    atomic_write_json(presets_file, presets)
    return gr.update(choices=list(presets.keys()), value=name), f"✅ Saved preset: {name}"

def load_clone_presets():
    if os.path.exists(clone_presets_file):
        with open(clone_presets_file, "r") as f: return json.load(f)
    return {}

def save_clone_preset(name, audio_path, ref_text, lang):
    name = name.strip()
    if not name or not audio_path: return gr.update(), "⚠️ Need name and audio."
    safe_name = "".join(x for x in name if x.isalnum() or x in "._- ")
    ext = os.path.splitext(audio_path)[1] or ".wav"
    permanent_audio_path = os.path.join(clone_audio_folder, f"{safe_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{ext}")
    shutil.copy(audio_path, permanent_audio_path)
    presets = load_clone_presets()
    presets[name] = {"audio_path": permanent_audio_path, "ref_text": ref_text if ref_text else "", "lang": lang}
    atomic_write_json(clone_presets_file, presets)
    return gr.update(choices=list(presets.keys()), value=name), f"✅ Saved: {name}"

def apply_preset(name):
    presets = load_presets()
    if name in presets: return presets[name]["desc"], presets[name]["lang"]
    return "", "Auto"

def apply_clone_preset(name):
    presets = load_clone_presets()
    if name in presets: return presets[name]["audio_path"], presets[name]["ref_text"], presets[name]["lang"]
    return None, "", "Auto"

# =========================================================================
# 🎙️ GENERATION ENGINES (Master, Podcast, Batch)
# =========================================================================
def master_generate(text, lang, model_type, custom_filename, use_chunking, out_format, desc=None, ref_audio=None, ref_text=None, x_vector=False, do_norm=True, do_trim=True, do_srt=True, lexicon_data=None, progress=None):
    try:
        text = master_text_processor(text, lexicon_data)
        if progress: progress(0.05, desc="Loading Model to VRAM...")
        switch_vram_model("design" if model_type == "design" else "base")

        base_name = custom_filename.strip() if custom_filename.strip() else f"Generated_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        final_drive_path = os.path.join(voice_folder, f"{base_name}.wav")
        chunks_to_process = split_text_into_chunks(text) if use_chunking else [text]
        generated_audio_segments = []
        sample_rate = 24000

        for idx, chunk in enumerate(chunks_to_process):
            if not chunk.strip(): continue
            if progress: progress(0.1 + (0.6 * (idx / max(1, len(chunks_to_process)))), desc=f"Generating chunk {idx+1}/{len(chunks_to_process)}...")
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        if model_type == "design": wavs, sr = model_design.generate_voice_design(text=chunk, language=lang, instruct=desc, non_streaming_mode=True)
                        else: wavs, sr = model_base.generate_voice_clone(language=lang, text=chunk, ref_audio=ref_audio, ref_text=ref_text if not x_vector else None)
                else:
                    if model_type == "design": wavs, sr = model_design.generate_voice_design(text=chunk, language=lang, instruct=desc, non_streaming_mode=True)
                    else: wavs, sr = model_base.generate_voice_clone(language=lang, text=chunk, ref_audio=ref_audio, ref_text=ref_text if not x_vector else None)
            
            sample_rate = sr
            generated_audio_segments.append(wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0])

        if generated_audio_segments:
            silence = np.zeros(int(0.3 * sample_rate), dtype=generated_audio_segments[0].dtype)
            final_audio_list = []
            for i, seg in enumerate(generated_audio_segments):
                final_audio_list.append(seg)
                if i < len(generated_audio_segments) - 1: final_audio_list.append(silence)
            sf.write(final_drive_path, np.concatenate(final_audio_list), sample_rate)

        downloads = []
        status_msg = f"✅ Audio generated successfully!"

        if do_norm or do_trim or do_srt:
            if progress: progress(0.8, desc="Applying Polish & Captions...")
            y, sr_load = librosa.load(final_drive_path, sr=None)
            if do_trim: y, _ = librosa.effects.trim(y, top_db=30)
            if do_norm:
                max_amp = np.max(np.abs(y))
                if max_amp > 0: y = y / max_amp * 0.95
            if do_trim or do_norm:
                sf.write(final_drive_path, y, sr_load)
            if do_srt:
                res = whisper_model.transcribe(final_drive_path)
                srt_path = final_drive_path.replace(".wav", ".srt")
                write_srt_file(res["segments"], srt_path)
                downloads.append(srt_path)

        if progress: progress(0.95, desc="Converting Format...")
        final_export_path = convert_audio_format(final_drive_path, out_format)
        downloads.insert(0, final_export_path)
        status_msg += f"\n📂 Saved to Output Directory: {os.path.basename(final_export_path)}"
        if progress: progress(1.0, desc="Done!")
        
        return final_export_path, downloads, status_msg
    except Exception as e: return None, None, f"❌ Error: {str(e)}"

def btn_design_wrapper(txt, lng, fn, chunk, fmt, desc, norm, trim, srt, lex, progress=gr.Progress()):
    return master_generate(txt, lng, "design", fn, chunk, fmt, desc=desc, do_norm=norm, do_trim=trim, do_srt=srt, lexicon_data=lex, progress=progress)

def btn_clone_wrapper(txt, lng, fn, chunk, fmt, ra, rt, xv, norm, trim, srt, lex, progress=gr.Progress()):
    return master_generate(txt, lng, "clone", fn, chunk, fmt, ref_audio=ra, ref_text=rt, x_vector=xv, do_norm=norm, do_trim=trim, do_srt=srt, lexicon_data=lex, progress=progress)

def run_podcast_engine(script_text, lang, do_norm, do_trim, do_srt, out_format, lex,
                       s1_n, s1_p, s1_audio, s1_ref, s2_n, s2_p, s2_audio, s2_ref,
                       s3_n, s3_p, s3_audio, s3_ref, s4_n, s4_p, s4_audio, s4_ref, progress=gr.Progress()):
    try:
        script_text = master_text_processor(script_text, lex)
        progress(0.05, desc="Loading Podcast Setup...")
        switch_vram_model("base")
        presets = load_clone_presets()

        speaker_map = {}
        def map_speaker(name, preset, audio, ref_text):
            if name.strip(): speaker_map[name.strip()] = {"type": "upload", "audio": audio, "ref_text": ref_text} if audio else {"type": "preset", "preset_name": preset}

        map_speaker(s1_n, s1_p, s1_audio, s1_ref)
        map_speaker(s2_n, s2_p, s2_audio, s2_ref)
        map_speaker(s3_n, s3_p, s3_audio, s3_ref)
        map_speaker(s4_n, s4_p, s4_audio, s4_ref)

        lines = script_text.split('\n')
        segments, sr_final = [], 24000
        status_log = ["🎙️ Starting Podcast Generation..."]
        yield None, None, "\n".join(status_log)

        for i, line in enumerate(lines):
            if not line.strip(): continue
            match = re.match(r'^\[(.*?)\]:\s*(.*)', line.strip())
            if match:
                raw_speaker, text = match.group(1).strip(), match.group(2).strip()
                if raw_speaker not in speaker_map:
                    if raw_speaker in presets: speaker_info = {"type": "preset", "preset_name": raw_speaker}
                    else: continue
                else: speaker_info = speaker_map[raw_speaker]

                progress(0.1 + (0.7 * (i / max(1, len(lines)))), desc=f"Dubbing [{raw_speaker}]...")
                
                audio_path = speaker_info["audio"] if speaker_info["type"] == "upload" else presets[speaker_info["preset_name"]]["audio_path"]
                ref_text = speaker_info["ref_text"] if speaker_info["type"] == "upload" else presets[speaker_info["preset_name"]]["ref_text"]

                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            wavs, sr = model_base.generate_voice_clone(language=lang, text=text, ref_audio=audio_path, ref_text=ref_text)
                    else:
                        wavs, sr = model_base.generate_voice_clone(language=lang, text=text, ref_audio=audio_path, ref_text=ref_text)
                
                sr_final = sr
                segments.append(wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0])
                segments.append(np.zeros(int(0.5 * sr_final), dtype=segments[-1].dtype))

        if not segments:
            yield None, None, "❌ No valid dialogue found."
            return

        final_drive_path = os.path.join(voice_folder, f"Podcast_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        sf.write(final_drive_path, np.concatenate(segments), sr_final)

        downloads = []
        if do_norm or do_trim or do_srt:
            progress(0.85, desc="Final Mix & Captions...")
            y, sr_load = librosa.load(final_drive_path, sr=None)
            if do_trim: y, _ = librosa.effects.trim(y, top_db=30)
            if do_norm:
                max_amp = np.max(np.abs(y))
                if max_amp > 0: y = y / max_amp * 0.95
            if do_trim or do_norm: sf.write(final_drive_path, y, sr_load)
            if do_srt:
                res = whisper_model.transcribe(final_drive_path)
                srt_path = final_drive_path.replace(".wav", ".srt")
                write_srt_file(res["segments"], srt_path)
                downloads.append(srt_path)

        progress(0.95, desc="Converting Format...")
        final_export_path = convert_audio_format(final_drive_path, out_format)
        downloads.insert(0, final_export_path)
        status_log.append(f"✅ Podcast Ready!")
        progress(1.0, desc="Podcast Ready!")
        yield final_export_path, downloads, "\n".join(status_log)

    except Exception as e: yield None, None, f"❌ Error: {str(e)}"

def run_batch(mode, text_list, lang, desc, clone_preset, custom_prefix, do_srt, target_langs, out_format, lex, progress=gr.Progress()):
    try:
        progress(0.05, desc="Initializing Batch Engine...")
        switch_vram_model("design" if mode == "Voice Design" else "base")

        lines = [line.strip() for line in text_list.split('\n') if line.strip()]
        if not lines: yield None, "⚠️ Please paste some text.", []; return

        prefix = custom_prefix.strip() if custom_prefix.strip() else "Batch"
        batch_folder = os.path.join(voice_folder, f"{prefix}_Batch_{datetime.datetime.now().strftime('%H%M%S')}")
        os.makedirs(batch_folder, exist_ok=True)

        audio_path, ref_text = None, None
        if mode == "Voice Clone":
            if not clone_preset: yield None, "⚠️ Please select a Clone Preset.", []; return
            presets = load_clone_presets()
            audio_path, ref_text = presets[clone_preset]["audio_path"], presets[clone_preset]["ref_text"]

        lang_map = {"English": "en", "Chinese": "zh-CN", "Japanese": "ja", "Korean": "ko", "French": "fr", "German": "de", "Spanish": "es"}
        status_log = [f"🚀 Starting Batch: {len(lines)} scripts detected."]
        all_downloadable_files = []
        grid_data = []
        yield None, "\n".join(status_log), grid_data

        for i, raw_line in enumerate(lines):
            line = master_text_processor(raw_line, lex)
            progress((i+1)/len(lines), desc=f"Processing {i+1}/{len(lines)}...")
            wav_filepath = os.path.join(batch_folder, f"{prefix}_{str(i+1).zfill(2)}_{lang}.wav")
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        if mode == "Voice Design": wavs, sr = model_design.generate_voice_design(text=line, language=lang, instruct=desc, non_streaming_mode=True)
                        else: wavs, sr = model_base.generate_voice_clone(language=lang, text=line, ref_audio=audio_path, ref_text=ref_text)
                else:
                    if mode == "Voice Design": wavs, sr = model_design.generate_voice_design(text=line, language=lang, instruct=desc, non_streaming_mode=True)
                    else: wavs, sr = model_base.generate_voice_clone(language=lang, text=line, ref_audio=audio_path, ref_text=ref_text)
            
            sf.write(wav_filepath, wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0], sr)

            if do_srt:
                res = whisper_model.transcribe(wav_filepath)
                srt_path = wav_filepath.replace(".wav", ".srt")
                write_srt_file(res["segments"], srt_path)
                all_downloadable_files.append(srt_path)

            final_base_file = convert_audio_format(wav_filepath, out_format)
            all_downloadable_files.append(final_base_file)
            grid_data.append([os.path.basename(final_base_file), lang, f"{os.path.getsize(final_base_file)/1024:.1f} KB"])
            yield None, f"⏳ Processed {i+1}/{len(lines)}...", grid_data

            for g_lang in target_langs:
                if g_lang == lang: continue
                code = lang_map.get(g_lang, "en")
                translated = GoogleTranslator(source='auto', target=code).translate(raw_line)
                translated = master_text_processor(translated, lex)

                g_wav_filepath = os.path.join(batch_folder, f"{prefix}_{str(i+1).zfill(2)}_{g_lang}.wav")
                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            if mode == "Voice Design": g_wavs, sr = model_design.generate_voice_design(text=translated, language=g_lang, instruct=desc, non_streaming_mode=True)
                            else: g_wavs, sr = model_base.generate_voice_clone(language=g_lang, text=translated, ref_audio=audio_path, ref_text=ref_text)
                    else:
                        if mode == "Voice Design": g_wavs, sr = model_design.generate_voice_design(text=translated, language=g_lang, instruct=desc, non_streaming_mode=True)
                        else: g_wavs, sr = model_base.generate_voice_clone(language=g_lang, text=translated, ref_audio=audio_path, ref_text=ref_text)
                
                sf.write(g_wav_filepath, g_wavs[0].cpu().numpy() if torch.is_tensor(g_wavs[0]) else g_wavs[0], sr)

                if do_srt:
                    g_res = whisper_model.transcribe(g_wav_filepath)
                    g_srt_path = g_wav_filepath.replace(".wav", ".srt")
                    write_srt_file(g_res["segments"], g_srt_path)
                    all_downloadable_files.append(g_srt_path)

                final_dub_file = convert_audio_format(g_wav_filepath, out_format)
                all_downloadable_files.append(final_dub_file)
                grid_data.append([os.path.basename(final_dub_file), g_lang, f"{os.path.getsize(final_dub_file)/1024:.1f} KB"])
                yield None, f"⏳ Dubbing {i+1} into {g_lang}...", grid_data

        progress(0.95, desc="Zipping all files...")
        zip_path = shutil.make_archive(batch_folder, 'zip', batch_folder)
        all_downloadable_files.insert(0, zip_path)

        progress(1.0, desc="Batch Complete!")
        status_log.append(f"✅ Batch Complete! All files zipped for easy download.")
        yield all_downloadable_files, "\n".join(status_log), grid_data

    except Exception as e: yield None, f"❌ Batch Error: {str(e)}", []


# =========================================================================
# 🎨 BUILD THE UI
# =========================================================================
initial_presets = list(load_presets().keys())
initial_clone_presets = list(load_clone_presets().keys())

gr.close_all()

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as app:

    gr.HTML("""
    <style>
        .yt-subscribe-btn {
            display: inline-block; padding: 14px 28px; background-color: #ff0000; color: #ffffff !important;
            font-weight: 800; text-decoration: none; border-radius: 8px; font-size: 18px;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); box-shadow: 0 4px 10px rgba(255, 0, 0, 0.4); cursor: pointer;
        }
        .yt-subscribe-btn:hover { background-color: #cc0000; transform: scale(1.05); box-shadow: 0 6px 18px rgba(255, 0, 0, 0.6); }
        .pro-console textarea { border: 1px solid #7b61ff; box-shadow: inset 0 0 5px rgba(123,97,255,0.2); }
    </style>
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://github.com/TechTheoryLab/TechTheoryLab-Assets/blob/main/Tech%20Theory%20Lab%20Banner.png?raw=true"
             style="max-width: 100%; height: auto; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <br><br>
        <a href="https://www.youtube.com/@TechTheoryLab?sub_confirmation=1" target="_blank" class="yt-subscribe-btn">
            ▶️ SUBSCRIBE TO THE CHANNEL
        </a>
    </div>
    """)

    with gr.Row():
        clear_vram_btn = gr.Button("🧹 Clear GPU Memory", variant="stop", scale=1)
        vram_refresh_btn = gr.Button("🔄 Refresh Stats", scale=1)
        vram_status = gr.Textbox(value=get_vram_status(), interactive=False, scale=4, show_label=False)
        
    clear_vram_btn.click(fn=manual_clear_vram, outputs=[vram_status])
    vram_refresh_btn.click(fn=get_vram_status, outputs=[vram_status])

    global_lexicon = gr.State([["PCE", "P C E"], ["Fed", "Federal Reserve"]])

    with gr.Tabs():

        # --- TAB 1: VOICE DESIGN ---
        with gr.Tab("Voice Design"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Script & Settings")
                    vd_text = gr.Textbox(label="Script", lines=5)
                    
                    with gr.Row():
                        vd_lang = gr.Dropdown(choices=QWEN_LANGUAGES, value="Auto", label="Language")
                        vd_filename = gr.Textbox(label="File Name", placeholder="Optional")
                    
                    vd_desc = gr.Textbox(label="Voice Description", lines=2, placeholder="e.g., A confident female news anchor.")

                    with gr.Accordion("⚙️ Advanced Processing Options", open=False):
                        with gr.Row():
                            vd_chunking = gr.Checkbox(label="🔪 Auto-Chunking", value=False, info="Splits long text to avoid crashes.")
                            vd_norm = gr.Checkbox(label="🔊 Auto-Normalize", value=True, info="Boosts overall audio volume.")
                        with gr.Row():
                            vd_trim = gr.Checkbox(label="✂️ Auto-Trim", value=True, info="Removes dead silence.")
                            vd_srt = gr.Checkbox(label="📝 Generate .SRT", value=True)
                        vd_format = gr.Dropdown(choices=["WAV", "MP3", "FLAC", "OGG"], value="MP3", label="Output Format")
                    
                    with gr.Row():
                        vd_preview_btn = gr.Button("🔍 Preview Chunks", scale=1)
                        vd_btn = gr.Button("✨ Generate Audio", variant="primary", scale=3)
                        vd_stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1)
                    
                    vd_chunk_preview = gr.Textbox(label="Chunking Preview Sandbox", lines=3, interactive=False, visible=False)

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 💾 Presets Management")
                        with gr.Row(equal_height=True):
                            vd_preset_dropdown = gr.Dropdown(choices=initial_presets, label="Load Saved Voice Preset", scale=2)
                            vd_preset_name = gr.Textbox(label="New Preset Name", scale=2)
                        vd_save_preset_btn = gr.Button("💾 Save Current Configuration")

                    gr.Markdown("### 🎧 Output Dashboard")
                    vd_audio = gr.Audio(label="Audio Output", type="filepath")
                    vd_download = gr.File(label="📥 Download Files", file_count="multiple", interactive=False)
                    vd_status = gr.Textbox(label="Status Console", lines=3, elem_classes=["pro-console"])

            vd_preview_btn.click(fn=preview_chunks, inputs=[vd_text], outputs=[vd_chunk_preview])
            vd_preset_dropdown.change(fn=lambda n: apply_preset(n), inputs=[vd_preset_dropdown], outputs=[vd_desc, vd_lang])
            vd_save_preset_btn.click(fn=save_preset_to_drive, inputs=[vd_preset_name, vd_desc, vd_lang], outputs=[vd_preset_dropdown, vd_status])
            vd_event = vd_btn.click(fn=btn_design_wrapper, inputs=[vd_text, vd_lang, vd_filename, vd_chunking, vd_format, vd_desc, vd_norm, vd_trim, vd_srt, global_lexicon], outputs=[vd_audio, vd_download, vd_status])
            vd_stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[vd_event])

        # --- TAB 2: VOICE CLONE ---
        with gr.Tab("Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎤 Source Audio & Target Script")
                    vc_ref_audio = gr.Audio(type="filepath", label="Reference Audio")
                    vc_target_text = gr.Textbox(label="Target Script to Clone", lines=4)

                    with gr.Accordion("⚙️ Advanced Reference & Processing Options", open=False):
                        vc_ref_text = gr.Textbox(label="Reference Transcript (Optional)", lines=2, info="Provide exact words spoken in audio.")
                        vc_xvector = gr.Checkbox(label="Ignore Transcript (x-vector mode only)")
                        with gr.Row():
                            vc_lang = gr.Dropdown(choices=QWEN_LANGUAGES, value="Auto", label="Language")
                            vc_filename = gr.Textbox(label="File Name", placeholder="Optional")
                        with gr.Row():
                            vc_chunking = gr.Checkbox(label="🔪 Auto-Chunking", value=False)
                            vc_norm = gr.Checkbox(label="🔊 Auto-Normalize", value=True)
                        with gr.Row():
                            vc_trim = gr.Checkbox(label="✂️ Auto-Trim", value=True)
                            vc_srt = gr.Checkbox(label="📝 Generate .SRT", value=True)
                        vc_format = gr.Dropdown(choices=["WAV", "MP3", "FLAC", "OGG"], value="MP3", label="Output Format")
                    
                    with gr.Row():
                        vc_preview_btn = gr.Button("🔍 Preview Chunks", scale=1)
                        vc_btn = gr.Button("🧬 Clone Voice", variant="primary", scale=3)
                        vc_stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1)

                    vc_chunk_preview = gr.Textbox(label="Chunking Preview Sandbox", lines=3, interactive=False, visible=False)

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 💾 Clone Presets Management")
                        with gr.Row(equal_height=True):
                            vc_preset_dropdown = gr.Dropdown(choices=initial_clone_presets, label="Load Preset", scale=2)
                            vc_preset_name = gr.Textbox(label="New Preset Name", scale=2)
                        vc_save_preset_btn = gr.Button("💾 Save Voice as Preset")

                    gr.Markdown("### 🎧 Output Dashboard")
                    vc_audio = gr.Audio(label="Audio Output")
                    vc_download = gr.File(label="📥 Download Files", file_count="multiple", interactive=False)
                    vc_status = gr.Textbox(label="Status Console", lines=3, elem_classes=["pro-console"])

            vc_preview_btn.click(fn=preview_chunks, inputs=[vc_target_text], outputs=[vc_chunk_preview])
            vc_preset_dropdown.change(fn=apply_clone_preset, inputs=[vc_preset_dropdown], outputs=[vc_ref_audio, vc_ref_text, vc_lang])
            vc_save_preset_btn.click(fn=save_clone_preset, inputs=[vc_preset_name, vc_ref_audio, vc_ref_text, vc_lang], outputs=[vc_preset_dropdown, vc_status])
            vc_event = vc_btn.click(fn=btn_clone_wrapper, inputs=[vc_target_text, vc_lang, vc_filename, vc_chunking, vc_format, vc_ref_audio, vc_ref_text, vc_xvector, vc_norm, vc_trim, vc_srt, global_lexicon], outputs=[vc_audio, vc_download, vc_status])
            vc_stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[vc_event])

        # --- TAB 3: VIRAL BATCH GENERATOR ---
        with gr.Tab("Batch Generator"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🚀 Multi-Script Batch Processing")
                    batch_mode = gr.Radio(choices=["Voice Design", "Voice Clone"], value="Voice Clone", label="Generation Mode")
                    batch_text = gr.Textbox(label="Paste multiple scripts (Press Enter between each)", lines=6)

                    with gr.Group(visible=False) as design_row:
                        batch_desc = gr.Textbox(label="Voice Description", lines=2)
                    with gr.Group(visible=True) as clone_row:
                        batch_clone_preset = gr.Dropdown(choices=initial_clone_presets, label="Select Clone Preset")

                    with gr.Row():
                        batch_lang = gr.Dropdown(choices=QWEN_LANGUAGES, value="Auto", label="Base Language")
                        batch_prefix = gr.Textbox(label="File Prefix", placeholder="e.g., Tech_Shorts")
                    
                    with gr.Accordion("🌍 Global Localization Studio", open=False):
                        batch_target_langs = gr.CheckboxGroup(
                            choices=["English", "Chinese", "Japanese", "Korean", "French", "German", "Spanish"],
                            label="Select Languages for Auto-Dubbing"
                        )
                        batch_srt = gr.Checkbox(label="📝 Auto-Generate .SRT for EVERY file", value=True)
                        batch_format = gr.Dropdown(choices=["WAV", "MP3", "FLAC", "OGG"], value="MP3", label="Output Format")

                    with gr.Row():
                        batch_btn = gr.Button("⚡ Run Super-Batch", variant="primary", scale=3)
                        batch_stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1)

                with gr.Column(scale=1):
                    batch_downloads = gr.File(label="📥 Download ZIP Archive", interactive=False)
                    batch_status = gr.Textbox(label="Batch Console", interactive=False, lines=2, elem_classes=["pro-console"])
                    
                    gr.Markdown("### 📊 Live Batch Overview")
                    batch_grid = gr.Dataframe(headers=["Generated File", "Language", "Size"], interactive=False, col_count=(3, "fixed"))

            def toggle_batch_mode(mode): return (gr.update(visible=True), gr.update(visible=False)) if mode == "Voice Design" else (gr.update(visible=False), gr.update(visible=True))
            batch_mode.change(fn=toggle_batch_mode, inputs=[batch_mode], outputs=[design_row, clone_row])
            batch_event = batch_btn.click(fn=run_batch, inputs=[batch_mode, batch_text, batch_lang, batch_desc, batch_clone_preset, batch_prefix, batch_srt, batch_target_langs, batch_format, global_lexicon], outputs=[batch_downloads, batch_status, batch_grid])
            batch_stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[batch_event])

        # --- TAB 4: PODCAST STUDIO ---
        with gr.Tab("Podcast Studio"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("👥 Cast Members", open=True):
                        with gr.Group():
                            with gr.Row():
                                s1_name = gr.Textbox(label="Speaker 1 Tag", scale=1)
                                s1_preset = gr.Dropdown(choices=initial_clone_presets, label="Preset", scale=1)
                            with gr.Accordion("Direct Upload Override", open=False):
                                s1_audio = gr.Audio(type="filepath", label="Direct Audio")
                                s1_ref = gr.Textbox(label="Ref Text", lines=1)
                        with gr.Group():
                            with gr.Row():
                                s2_name = gr.Textbox(label="Speaker 2 Tag", scale=1)
                                s2_preset = gr.Dropdown(choices=initial_clone_presets, label="Preset", scale=1)
                            with gr.Accordion("Direct Upload Override", open=False):
                                s2_audio = gr.Audio(type="filepath", label="Direct Audio")
                                s2_ref = gr.Textbox(label="Ref Text", lines=1)
                        with gr.Accordion("Expand for Speakers 3 & 4", open=False):
                            with gr.Group():
                                with gr.Row():
                                    s3_name = gr.Textbox(label="Speaker 3 Tag", scale=1)
                                    s3_preset = gr.Dropdown(choices=initial_clone_presets, label="Preset", scale=1)
                                s3_audio = gr.Audio(type="filepath", label="Direct Audio")
                                s3_ref = gr.Textbox(label="Ref Text", lines=1)
                            with gr.Group():
                                with gr.Row():
                                    s4_name = gr.Textbox(label="Speaker 4 Tag", scale=1)
                                    s4_preset = gr.Dropdown(choices=initial_clone_presets, label="Preset", scale=1)
                                s4_audio = gr.Audio(type="filepath", label="Direct Audio")
                                s4_ref = gr.Textbox(label="Ref Text", lines=1)

                with gr.Column(scale=1):
                    pod_script = gr.Textbox(label="Master Podcast Script", lines=12, placeholder="[Host]: Hello!\n[Guest]: Hi.")
                    with gr.Accordion("⚙️ Master Export Settings", open=False):
                        with gr.Row():
                            pod_lang = gr.Dropdown(choices=QWEN_LANGUAGES, value="Auto", label="Language")
                            pod_format = gr.Dropdown(choices=["WAV", "MP3"], value="MP3", label="Format")
                        with gr.Row():
                            pod_norm = gr.Checkbox(label="🔊 Normalize", value=True)
                            pod_trim = gr.Checkbox(label="✂️ Trim", value=True)
                            pod_srt = gr.Checkbox(label="📝 Gen .SRT", value=True)

                    with gr.Row():
                        pod_btn = gr.Button("🎧 Compile Episode", variant="primary", scale=3)
                        pod_stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1)
                        
                    pod_audio = gr.Audio(label="Podcast Output")
                    pod_download = gr.File(label="📥 Assets", file_count="multiple", interactive=False)
                    pod_status = gr.Textbox(label="Console", lines=4, elem_classes=["pro-console"])

            pod_event = pod_btn.click(fn=run_podcast_engine, inputs=[pod_script, pod_lang, pod_norm, pod_trim, pod_srt, pod_format, global_lexicon, s1_name, s1_preset, s1_audio, s1_ref, s2_name, s2_preset, s2_audio, s2_ref, s3_name, s3_preset, s3_audio, s3_ref, s4_name, s4_preset, s4_audio, s4_ref], outputs=[pod_audio, pod_download, pod_status])
            pod_stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[pod_event])

        # --- TAB 5: FX MIXER ---
        with gr.Tab("FX Mixer"):
            with gr.Row():
                with gr.Column(scale=1):
                    pp_audio_in = gr.Audio(type="filepath", label="Upload Voice Audio")
                    with gr.Accordion("🎵 Background Music Track", open=False):
                        pp_bgm_in = gr.Audio(type="filepath", label="BGM Audio")
                        pp_bgm_vol = gr.Slider(minimum=0.0, maximum=1.0, value=0.15, step=0.05, label="BGM Volume")
                    with gr.Accordion("🎚️ Mastering Tools", open=False):
                        with gr.Row():
                            pp_norm = gr.Checkbox(label="🔊 Normalize", value=True)
                            pp_trim = gr.Checkbox(label="✂️ Trim", value=True)
                            pp_srt = gr.Checkbox(label="📝 Gen .SRT", value=True)
                        pp_speed = gr.Slider(minimum=0.8, maximum=1.5, value=1.0, step=0.05, label="Speed Adjust")
                    pp_format = gr.Dropdown(choices=["WAV", "MP3"], value="MP3", label="Output Format")
                    
                    with gr.Row():
                        pp_btn = gr.Button("⚡ Apply Mix", variant="primary", scale=3)
                        pp_stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1)

                with gr.Column(scale=1):
                    pp_audio_out = gr.Audio(label="Final Mix")
                    pp_downloads = gr.File(label="📥 Downloads", file_count="multiple", interactive=False)
                    pp_status = gr.Textbox(label="Mixer Console", lines=8, elem_classes=["pro-console"])

            pp_event = pp_btn.click(fn=process_audio_studio, inputs=[pp_audio_in, pp_norm, pp_trim, pp_speed, pp_srt, pp_bgm_in, pp_bgm_vol, pp_format], outputs=[pp_audio_out, pp_downloads, pp_status])
            pp_stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[pp_event])

        # --- TAB 6: GLOBAL STUDIO SETTINGS ---
        with gr.Tab("Global Studio Settings"):
            gr.Markdown("### 📖 Custom Pronunciation Lexicon")
            gr.Markdown("Teach the AI how to perfectly pronounce acronyms, metrics, or custom names. *(e.g., Target: **PCE**, Replacement: **P C E**)*. This applies automatically across **all** generation tabs.")
            
            lexicon_table = gr.Dataframe(
                headers=["Target Text", "Phonetic Replacement"], 
                value=[["PCE", "P C E"], ["Fed", "Federal Reserve"]], 
                col_count=(2, "fixed"), 
                interactive=True
            )
            
            gr.Markdown("*Note: Large numbers starting with `$` and ending in `trillion/billion/million` or ending in `%` are now automatically normalized natively by the Pro Engine.*")
            lexicon_table.change(fn=lambda x: x, inputs=[lexicon_table], outputs=[global_lexicon])

print("Starting Tech Theory Lab Pro Studio...")
app.launch(inline=True, share=True, debug=True, show_error=True)
