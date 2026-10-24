import os
import io
import wave
import struct
import math
import numpy as np
from scipy import signal

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "audio_samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

def generate_sine_wave(freq: float, duration: float, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)

def generate_rain_sound(duration: float, sr: int = 16000, intensity: float = 0.6) -> np.ndarray:
    """
    Synthesize realistic rain soundscape:
    - Band-passed noise between 600 Hz and 7000 Hz.
    - Intermittent droplet clicks with exponential decay.
    """
    n = int(sr * duration)
    # Bandpass white noise for water rushing/rain texture
    nyq = sr / 2.0
    low = 600.0 / nyq
    high = min(7500.0 / nyq, 0.99)
    b, a = signal.butter(3, [low, high], btype='bandpass')
    white = np.random.randn(n)
    rain_base = signal.lfilter(b, a, white)
    rain_base = rain_base / (np.max(np.abs(rain_base)) + 1e-6)

    # Add droplet transients
    droplets = np.zeros(n)
    num_drops = int(duration * 60 * intensity)
    drop_indices = np.random.randint(0, max(1, n - 300), size=num_drops)
    for idx in drop_indices:
        drop_len = min(250, n - idx)
        t_drop = np.arange(drop_len) / sr
        freq = np.random.uniform(2200, 5000)
        decay = np.exp(-t_drop * np.random.uniform(25, 60))
        droplets[idx:idx + drop_len] += np.sin(2 * np.pi * freq * t_drop) * decay * np.random.uniform(0.3, 0.7)

    rain = (0.5 * rain_base + 0.5 * droplets) * intensity
    return rain

def generate_siren_sound(duration: float, sr: int = 16000, amp: float = 0.25) -> np.ndarray:
    """Synthesize European/US wailing emergency siren (650Hz - 950Hz oscillation)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 0.4 Hz cycle
    freq_mod = 800 + 200 * np.sin(2 * np.pi * 0.35 * t)
    phase = 2 * np.pi * np.cumsum(freq_mod) / sr
    return amp * np.sin(phase)

def generate_traffic_noise(duration: float, sr: int = 16000, amp: float = 0.3) -> np.ndarray:
    """Synthesize low-frequency urban traffic rumble + occasional car horns."""
    n = int(sr * duration)
    nyq = sr / 2.0
    # Lowpass filter for diesel engine & tire road roar (50 Hz - 400 Hz)
    b, a = signal.butter(3, min(450.0 / nyq, 0.99), btype='lowpass')
    white = np.random.randn(n)
    rumble = signal.lfilter(b, a, white)
    rumble = rumble / (np.max(np.abs(rumble)) + 1e-6) * amp

    # Add 2 car honks
    t = np.linspace(0, duration, n, endpoint=False)
    horn = np.zeros(n)
    if duration > 3.0:
        # Dual-tone horn (420 Hz & 500 Hz)
        h_start = int(sr * 1.5)
        h_end = int(sr * 2.2)
        if h_end <= n:
            h_t = t[h_start:h_end] - t[h_start]
            horn[h_start:h_end] = 0.22 * (np.sin(2 * np.pi * 420 * h_t) + np.sin(2 * np.pi * 504 * h_t))

    return rumble + horn

def generate_electrical_arc(duration: float, sr: int = 16000, amp: float = 0.4) -> np.ndarray:
    """Synthesize 50Hz/60Hz mains hum with harsh harmonics and intermittent chaotic sparks."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # 50 Hz base hum with 3rd, 5th, 7th harmonics (buzzing transformer)
    hum = (
        0.3 * np.sin(2 * np.pi * 50 * t) +
        0.25 * np.sin(2 * np.pi * 150 * t) +
        0.2 * np.sin(2 * np.pi * 250 * t) +
        0.15 * np.sin(2 * np.pi * 350 * t)
    )
    # Chaotic electrical crackles / sparks
    sparks = np.zeros(n)
    num_sparks = int(duration * 12)
    spark_locs = np.random.randint(0, max(1, n - 200), size=num_sparks)
    for loc in spark_locs:
        sp_len = min(150, n - loc)
        sparks[loc:loc + sp_len] += np.random.randn(sp_len) * np.random.uniform(0.4, 0.8)

    arc = (0.6 * hum + 0.4 * sparks) * amp
    return arc

def generate_speech_formants(phonemes: list, duration: float, sr: int = 16000, amp: float = 0.6) -> np.ndarray:
    """
    Synthesize stylized speech vocal tract formants for speech simulation
    F0 pitch ~130Hz modulated with typical vowel formants (F1, F2).
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # Pitch fundamental with vibrato
    f0 = 135 + 8 * np.sin(2 * np.pi * 3.5 * t)
    # Glottal pulse train
    glottal = signal.sawtooth(2 * np.pi * np.cumsum(f0) / sr, width=0.1)

    # Formant filtering (simulating urgent human voice vowels: /a/, /i/, /u/)
    nyq = sr / 2.0
    b_f1, a_f1 = signal.butter(2, [600 / nyq, 900 / nyq], btype='bandpass')
    b_f2, a_f2 = signal.butter(2, [1400 / nyq, 1900 / nyq], btype='bandpass')
    f1 = signal.lfilter(b_f1, a_f1, glottal)
    f2 = signal.lfilter(b_f2, a_f2, glottal)

    voice = (0.6 * f1 + 0.4 * f2)
    voice = voice / (np.max(np.abs(voice)) + 1e-6)

    # Speech cadence / pause envelope (words and pauses)
    envelope = np.zeros(n)
    chunk = int(sr * 0.4)
    for i in range(0, n, chunk * 2):
        end = min(n, i + chunk)
        envelope[i:end] = np.hanning(end - i)

    return voice * envelope * amp

def save_wav(filename: str, audio: np.ndarray, sr: int = 16000) -> str:
    path = os.path.join(SAMPLES_DIR, filename)
    # Normalize to -1 dB
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.89
    int16_audio = (audio * 32767).astype(np.int16)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_audio.tobytes())
    return path

def build_all_sample_audio_files():
    """Generates 4 rich benchmark acoustic hazard recordings with deterministic ground truth."""
    np.random.seed(42)
    sr = 16000
    duration = 5.0

    samples = {}

    # Sample 1: Rainstorm Emergency (Heavy Rain, Water Gushing, Distant Siren, Urgent Voice)
    rain = generate_rain_sound(duration, sr, intensity=0.9)
    siren = generate_siren_sound(duration, sr, amp=0.2)
    voice = generate_speech_formants(["open_drain_alert"], duration, sr, amp=0.65)
    s1_audio = rain + siren + voice
    s1_path = save_wav("sample_rainstorm_open_drain.wav", s1_audio, sr)
    samples["rainstorm_open_drain"] = {
        "id": "rainstorm_open_drain",
        "title": "Severe Storm: Submerged Open Manhole & Flooded School Road",
        "filename": "sample_rainstorm_open_drain.wav",
        "path": s1_path,
        "duration_sec": duration,
        "ground_truth_speech": "Emergency! The manhole lid outside St. Mary's Primary School has washed away in the floodwater! The current is pulling debris into the open pit. Children and parents are stranded, please send rescue dewatering immediately!",
        "ground_truth_summary": "Submerged missing manhole cover creating dangerous vortex suction hazard outside primary school entrance during flash flooding.",
        "ground_truth_urgency": "CRITICAL",
        "expected_environmental_cues": [
            {"cue": "Torrential Rain / Downpour", "category": "WEATHER", "confidence": 0.97},
            {"cue": "Rushing Floodwater / Gutter Inundation", "category": "WEATHER", "confidence": 0.94},
            {"cue": "Emergency Siren (Distant)", "category": "URBAN_TRAFFIC", "confidence": 0.91}
        ],
        "expected_scene": "Outdoor Monsoon Storm Surge & Flooded Arterial Street"
    }

    # Sample 2: Fallen Sparking Live Wire (Electrical Arc, Traffic Rumble, Rain Drizzle)
    arc = generate_electrical_arc(duration, sr, amp=0.55)
    rain_drizzle = generate_rain_sound(duration, sr, intensity=0.35)
    traffic = generate_traffic_noise(duration, sr, amp=0.25)
    voice2 = generate_speech_formants(["live_wire_sparking"], duration, sr, amp=0.6)
    s2_audio = arc + rain_drizzle + traffic + voice2
    s2_path = save_wav("sample_sparking_power_wire.wav", s2_audio, sr)
    samples["sparking_power_wire"] = {
        "id": "sparking_power_wire",
        "title": "High Voltage Live Wire Snapped Across Roadway",
        "filename": "sample_sparking_power_wire.wav",
        "path": s2_path,
        "duration_sec": duration,
        "ground_truth_speech": "Watch out! A 440-volt live wire has snapped off the utility pole near Ward 4 market. It is sparking violently on the wet road and smoking. We need the power grid cut right now before someone steps on it!",
        "ground_truth_summary": "Snapped 440V high-voltage power line arcing violently on wet roadway near busy market junction.",
        "ground_truth_urgency": "CRITICAL",
        "expected_environmental_cues": [
            {"cue": "Electrical Arcing / High-Voltage Buzz", "category": "STRUCTURAL_HAZARD", "confidence": 0.98},
            {"cue": "Rain Drizzle / Wet Surface", "category": "WEATHER", "confidence": 0.88},
            {"cue": "Urban Traffic Rumble", "category": "URBAN_TRAFFIC", "confidence": 0.85}
        ],
        "expected_scene": "Wet Urban Commercial Street with Active Electrical Discharge"
    }

    # Sample 3: Traffic Accident & Flood Gridlock (Car Horns, Heavy Engine, Splash)
    traffic_dense = generate_traffic_noise(duration, sr, amp=0.6)
    siren3 = generate_siren_sound(duration, sr, amp=0.3)
    voice3 = generate_speech_formants(["traffic_gridlock"], duration, sr, amp=0.6)
    s3_audio = traffic_dense + siren3 + voice3
    s3_path = save_wav("sample_traffic_gridlock_accident.wav", s3_audio, sr)
    samples["traffic_gridlock"] = {
        "id": "traffic_gridlock",
        "title": "Underpass Inundation & Multi-Vehicle Collision Gridlock",
        "filename": "sample_traffic_gridlock_accident.wav",
        "path": s3_path,
        "duration_sec": duration,
        "ground_truth_speech": "Two cars collided inside the flooded subway underpass due to zero visibility. Traffic is completely backed up for two kilometers and ambulance is stuck behind horns. We need traffic police and tow trucks immediately.",
        "ground_truth_summary": "Two-vehicle collision inside submerged underpass creating severe 2km traffic gridlock and obstructing emergency response.",
        "ground_truth_urgency": "HIGH",
        "expected_environmental_cues": [
            {"cue": "Vehicle Horns / Gridlock Honking", "category": "URBAN_TRAFFIC", "confidence": 0.96},
            {"cue": "Heavy Diesel Engine Roar", "category": "URBAN_TRAFFIC", "confidence": 0.90},
            {"cue": "Emergency Siren", "category": "URBAN_TRAFFIC", "confidence": 0.92}
        ],
        "expected_scene": "Congested Urban Subway Underpass / Highway"
    }

    # Sample 4: Calm Pothole / Road Crater Report (Quiet Residential Alley, Faint Birds)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Gentle background birds
    birds = np.zeros(int(sr * duration))
    for b_idx in [int(sr * 0.8), int(sr * 2.5), int(sr * 4.1)]:
        if b_idx + 1000 < len(birds):
            chirp_t = np.linspace(0, 0.08, 1000)
            chirp = 0.15 * np.sin(2 * np.pi * (3200 + 800 * np.sin(2 * np.pi * 30 * chirp_t)) * chirp_t)
            birds[b_idx:b_idx + 1000] += chirp
    voice4 = generate_speech_formants(["calm_pothole"], duration, sr, amp=0.7)
    s4_audio = birds + voice4 + 0.05 * np.random.randn(len(t))
    s4_path = save_wav("sample_quiet_pothole_report.wav", s4_audio, sr)
    samples["quiet_pothole_report"] = {
        "id": "quiet_pothole_report",
        "title": "Routine Civic Maintenance: Road Surface Crater",
        "filename": "sample_quiet_pothole_report.wav",
        "path": s4_path,
        "duration_sec": duration,
        "ground_truth_speech": "Hi, I am calling from 7th Cross Street. There is a deep pothole that formed after yesterday's rain near house number 42. It is about two feet wide. Two-wheelers are skidding. Please schedule a road patch crew when convenient.",
        "ground_truth_summary": "Two-foot wide road crater posing skidding hazard for two-wheelers in quiet residential street.",
        "ground_truth_urgency": "MEDIUM",
        "expected_environmental_cues": [
            {"cue": "Quiet Ambient Residential Environment", "category": "DOMESTIC", "confidence": 0.95},
            {"cue": "Distant Bird Chirping", "category": "WILDLIFE_NATURE", "confidence": 0.88}
        ],
        "expected_scene": "Quiet Residential Neighborhood (Low Ambient Noise)"
    }

    return samples

if __name__ == "__main__":
    res = build_all_sample_audio_files()
    print(f"Generated {len(res)} realistic acoustic hazard audio files in {SAMPLES_DIR}")
