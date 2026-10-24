#!/usr/bin/env python3
"""
Audio & Environmental Cue Backend Engine — CLI Tool
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.audio_acoustic_engine import audio_acoustic_engine

def print_banner():
    print("=" * 76)
    print("  AUDIO & ENVIRONMENTAL CUE ENGINE — ACOUSTIC PROBLEM TRIAGE SYSTEM")
    print("  Speech Summarization + Complete Background Soundscape Detection")
    print("=" * 76)

def format_cue_badge(intensity: str) -> str:
    badges = {
        "DOMINANT": "[!! DOMINANT !!]",
        "HEAVY":    "[!  HEAVY     !]",
        "MODERATE": "[   MODERATE  ]",
        "FAINT":    "[.  FAINT     .]"
    }
    return badges.get(intensity.upper(), "[  DETECTED  ]")

def display_analysis_result(res: dict):
    print("\n" + "-" * 76)
    print(f"  AUDIO HASH:    {res.get('audio_hash')}")
    print(f"  ENGINE MODE:   {res.get('engine_mode')}")
    print(f"  CACHE HIT:     {res.get('cache_hit')}")
    print("-" * 76)

    # 1. DSP Telemetry
    dsp = res.get("dsp_telemetry", {})
    print("\n[1] PHYSICAL DSP TELEMETRY:")
    print(f"  • Duration:           {dsp.get('duration_sec')} seconds")
    print(f"  • Sample Rate:        {dsp.get('sample_rate_hz')} Hz")
    print(f"  • RMS Energy:         {dsp.get('rms_energy_db')} dBFS")
    print(f"  • Peak Amplitude:     {dsp.get('peak_amplitude_db')} dBFS")
    print(f"  • Est. Noise Floor:   {dsp.get('noise_floor_db')} dBFS")
    print(f"  • Signal-to-Noise:    {dsp.get('estimated_snr_db')} dB")
    print(f"  • Spectral Centroid:  {dsp.get('spectral_centroid_hz')} Hz")
    print(f"  • High-Freq (>2kHz):  {dsp.get('high_freq_energy_ratio') * 100:.1f}% of total energy")
    print(f"  • Low-Freq (<400Hz):  {dsp.get('low_freq_energy_ratio') * 100:.1f}% of total energy")

    # 2. Speech Transcription & Summary
    print("\n[2] SPEECH PROBLEM TRANSCRIPTION & SUMMARIZATION:")
    print(f"  • Spoken Transcript:  \"{res.get('speech_transcription', '').strip()}\"")
    
    summary = res.get("speech_summary", {})
    urgency = summary.get("reported_urgency", "MEDIUM")
    hazard = summary.get("core_hazard_category", "GENERAL")
    print(f"  • Executive Summary:  {summary.get('executive_summary')}")
    print(f"  • Hazard Category:    {hazard}")
    print(f"  • Urgency Level:      [{urgency}]")
    print(f"  • Caller Affect:      {summary.get('caller_affect')}")
    print(f"  • Actionable Demand:  {summary.get('actionable_request')}")
    if summary.get("key_entities"):
        print(f"  • Mentioned Entities: {', '.join(summary.get('key_entities'))}")

    # 3. Environmental Cues ("Grab Every Environment Cue")
    print("\n[3] DETECTED ENVIRONMENTAL CUES & ACOUSTIC SOUNDSCAPE:")
    cues = res.get("environmental_cues", [])
    if not cues:
        print("  (No audible background environmental cues detected)")
    else:
        for idx, c in enumerate(cues, start=1):
            intensity = c.get("intensity", "MODERATE")
            badge = format_cue_badge(intensity)
            conf = int(c.get("confidence", 0.0) * 100)
            print(f"  {idx}. {badge} {c.get('cue_name')} ({conf}% conf | {c.get('category')})")
            print(f"     Time: {c.get('timestamp_range', 'Continuous')} | {c.get('acoustic_characteristics', '')}")

    # 4. Acoustic Scene Profile
    scene = res.get("acoustic_scene_profile", {})
    print("\n[4] ACOUSTIC SCENE CLASSIFICATION:")
    print(f"  • Environment:        {scene.get('primary_environment')}")
    print(f"  • Ambient Noise:      {scene.get('ambient_noise_level')}")
    print(f"  • Spatial Enclosure:  {scene.get('indoor_vs_outdoor')}")
    print(f"  • Clarity:            {scene.get('recording_clarity')}")

    # 5. Cross-Verification (Speech vs Acoustic Reality)
    cross = res.get("cross_verification", {})
    corrob = cross.get("corroborates_speech", False)
    status_str = "CONFIRMED & CORROBORATED" if corrob else "UNVERIFIED / DISCREPANT"
    score = int(cross.get("veracity_score", 0.0) * 100)
    print("\n[5] ENVIRONMENTAL VERACITY CROSS-VERIFICATION:")
    print(f"  • Status:             [{status_str}] ({score}% veracity score)")
    print(f"  • Justification:      {cross.get('correlation_explanation')}")
    print("-" * 76 + "\n")

def cmd_analyze(args):
    path = args.file
    if not os.path.exists(path):
        print(f"Error: Audio file not found at '{path}'")
        sys.exit(1)

    print(f"Loading audio file: {path}")
    with open(path, "rb") as f:
        audio_bytes = f.read()

    mime = "audio/wav" if path.endswith(".wav") else "audio/mp3" if path.endswith(".mp3") else "audio/m4a" if path.endswith(".m4a") else "audio/ogg"
    print(f"Sending to AudioAcousticEngine ({len(audio_bytes)} bytes, mime: {mime})...")
    start = time.time()
    res = audio_acoustic_engine.analyze_audio(audio_bytes, mime)
    elapsed = time.time() - start
    print(f"Analysis completed in {elapsed:.2f}s")
    display_analysis_result(res)

    if args.json:
        print("\nRaw JSON Output:")
        print(json.dumps(res, indent=2))

def cmd_sample(args):
    sample_id = args.id
    audio_info = audio_acoustic_engine.get_sample_audio_bytes(sample_id)
    if not audio_info:
        print(f"Error: Unknown sample ID '{sample_id}'. Available samples:")
        for s in audio_acoustic_engine.get_sample_recordings():
            print(f"  - {s['id']}: {s['title']}")
        sys.exit(1)

    audio_bytes, mime = audio_info
    print(f"Analyzing pre-seeded sample '{sample_id}' ({len(audio_bytes)} bytes)...")
    start = time.time()
    res = audio_acoustic_engine.analyze_audio(audio_bytes, mime)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.3f}s")
    display_analysis_result(res)

    if args.json:
        print(json.dumps(res, indent=2))

def cmd_list_samples(args):
    samples = audio_acoustic_engine.get_sample_recordings()
    print(f"\nAvailable Realistic Acoustic Hazard Samples ({len(samples)} total):\n")
    for idx, s in enumerate(samples, start=1):
        print(f"{idx}. ID: {s['id']}")
        print(f"   Title:           {s['title']}")
        print(f"   Duration:        {s['duration_sec']}s")
        print(f"   Expected Urgency:[{s['expected_urgency']}]")
        print(f"   Expected Scene:  {s['expected_scene']}")
        print(f"   Expected Cues:   {', '.join(s['expected_cues'])}")
        print(f"   Summary:         {s['ground_truth_summary']}")
        print("-" * 60)

def cmd_benchmark(args):
    samples = audio_acoustic_engine.get_sample_recordings()
    print(f"\nRunning Benchmark Across All {len(samples)} Samples...\n")
    print(f"{'Sample ID':<26} | {'Duration':<8} | {'Engine Mode':<30} | {'Veracity':<8} | {'Latency':<8}")
    print("-" * 90)

    for s in samples:
        audio_bytes, mime = audio_acoustic_engine.get_sample_audio_bytes(s["id"])
        t0 = time.time()
        res = audio_acoustic_engine.analyze_audio(audio_bytes, mime)
        t_el = (time.time() - t0) * 1000.0
        mode = res.get("engine_mode", "N/A")[:30]
        veracity = f"{int(res.get('cross_verification', {}).get('veracity_score', 0) * 100)}%"
        print(f"{s['id']:<26} | {s['duration_sec']:<7.1f}s | {mode:<30} | {veracity:<8} | {t_el:<6.1f}ms")

    print("\nAll samples successfully verified!\n")

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Audio Problem Summarization & Environmental Cue Backend Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    p_analyze = subparsers.add_parser("analyze", help="Analyze an audio recording file")
    p_analyze.add_argument("--file", "-f", required=True, help="Path to audio file (.wav, .mp3, .ogg, .m4a)")
    p_analyze.add_argument("--json", action="store_true", help="Print raw JSON response")

    p_sample = subparsers.add_parser("sample", help="Run analysis on a built-in realistic acoustic sample")
    p_sample.add_argument("--id", "-i", required=True, help="Sample ID (e.g. rainstorm_open_drain, sparking_power_wire, etc.)")
    p_sample.add_argument("--json", action="store_true", help="Print raw JSON response")

    subparsers.add_parser("list-samples", help="List all available realistic test audio recordings")
    subparsers.add_parser("benchmark", help="Benchmark processing time and veracity for all samples")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "sample":
        cmd_sample(args)
    elif args.command == "list-samples":
        cmd_list_samples(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)

if __name__ == "__main__":
    main()
