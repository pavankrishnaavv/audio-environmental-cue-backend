import unittest
import os
import io
import wave
import numpy as np

from engine.audio_acoustic_engine import AudioAcousticEngine
from engine.audio_generator import build_all_sample_audio_files

class TestAudioAcousticEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_all_sample_audio_files()
        cls.engine = AudioAcousticEngine()

    def test_samples_exist(self):
        samples = self.engine.get_sample_recordings()
        self.assertGreaterEqual(len(samples), 4)
        sample_ids = [s["id"] for s in samples]
        self.assertIn("rainstorm_open_drain", sample_ids)
        self.assertIn("sparking_power_wire", sample_ids)
        self.assertIn("traffic_gridlock", sample_ids)
        self.assertIn("quiet_pothole_report", sample_ids)

    def test_dsp_feature_extraction(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sine = (0.5 * np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(sine.tobytes())

        dsp, samples, extracted_sr = self.engine.extract_dsp_features(buf.getvalue())
        self.assertEqual(extracted_sr, 16000)
        self.assertAlmostEqual(dsp["duration_sec"], 1.0, places=1)
        self.assertIn("rms_energy_db", dsp)
        self.assertIn("spectral_centroid_hz", dsp)
        self.assertIn("estimated_snr_db", dsp)

    def test_rainstorm_sample_analysis(self):
        audio_bytes, mime = self.engine.get_sample_audio_bytes("rainstorm_open_drain")
        self.assertIsNotNone(audio_bytes)
        self.assertGreater(len(audio_bytes), 1000)

        result = self.engine.analyze_audio(audio_bytes, mime)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("speech_summary", result)
        self.assertIn("environmental_cues", result)
        
        summary = result["speech_summary"]
        self.assertIn("executive_summary", summary)
        self.assertIn("core_hazard_category", summary)
        self.assertIn("reported_urgency", summary)

        cues = result["environmental_cues"]
        self.assertGreaterEqual(len(cues), 1)
        cue_names = [c["cue_name"].lower() for c in cues]
        has_rain_cue = any("rain" in name or "water" in name for name in cue_names)
        self.assertTrue(has_rain_cue, f"Expected rain cue in {cue_names}")

        self.assertIn("acoustic_scene_profile", result)
        self.assertIn("cross_verification", result)
        self.assertTrue(result["cross_verification"]["corroborates_speech"])

    def test_sparking_wire_sample_analysis(self):
        audio_bytes, mime = self.engine.get_sample_audio_bytes("sparking_power_wire")
        result = self.engine.analyze_audio(audio_bytes, mime)
        self.assertEqual(result["status"], "SUCCESS")
        cues = result["environmental_cues"]
        cue_names = [c["cue_name"].lower() for c in cues]
        has_electric_cue = any("arc" in name or "wire" in name or "buzz" in name or "spark" in name for name in cue_names)
        self.assertTrue(has_electric_cue, f"Expected electrical cue in {cue_names}")

    def test_deterministic_dsp_fallback(self):
        fake_dsp = {
            "duration_sec": 3.0,
            "sample_rate_hz": 16000,
            "high_freq_energy_ratio": 0.48,
            "low_freq_energy_ratio": 0.15,
            "rms_energy_db": -22.0,
            "noise_floor_db": -40.0,
            "spectral_centroid_hz": 2800.0
        }
        fallback_res = self.engine._deterministic_dsp_analysis(fake_dsp)
        self.assertIn("environmental_cues", fallback_res)
        self.assertEqual(fallback_res["engine_mode"], "DETERMINISTIC_DSP_FALLBACK")
        cues = fallback_res["environmental_cues"]
        self.assertTrue(any("rain" in c["cue_name"].lower() for c in cues))

if __name__ == "__main__":
    unittest.main()
