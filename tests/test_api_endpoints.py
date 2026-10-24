import unittest
import json
from fastapi.testclient import TestClient
from main import app

class TestAudioApiEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_audio_health(self):
        res = self.client.get("/api/audio/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertEqual(data["dsp_spectral_pipeline"], "OPERATIONAL")
        self.assertGreaterEqual(data["samples_ready"], 4)

    def test_list_audio_samples(self):
        res = self.client.get("/api/audio/samples")
        self.assertEqual(res.status_code, 200)
        samples = res.json()
        self.assertGreaterEqual(len(samples), 4)
        sample_ids = [s["id"] for s in samples]
        self.assertIn("rainstorm_open_drain", sample_ids)
        self.assertIn("sparking_power_wire", sample_ids)

    def test_stream_audio_sample(self):
        res = self.client.get("/api/audio/samples/rainstorm_open_drain/stream")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "audio/wav")
        self.assertGreater(len(res.content), 1000)

    def test_analyze_audio_endpoint(self):
        sample_res = self.client.get("/api/audio/samples/rainstorm_open_drain/stream")
        self.assertEqual(sample_res.status_code, 200)

        files = {"file": ("test.wav", sample_res.content, "audio/wav")}
        res = self.client.post("/api/audio/analyze", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("speech_summary", data)
        self.assertIn("environmental_cues", data)
        self.assertIn("dsp_telemetry", data)

        cues = data["environmental_cues"]
        cue_names = [c["cue_name"].lower() for c in cues]
        self.assertTrue(any("rain" in n or "water" in n for n in cue_names))

    def test_record_and_triage_endpoint(self):
        res = self.client.post("/api/audio/record-and-triage", data={
            "sample_id": "rainstorm_open_drain",
            "latitude": 19.0760,
            "longitude": 72.8777,
            "is_anonymous": True
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "TRIAGED_AND_VERIFIED")
        self.assertIn("incident_id", data)
        self.assertIn("speech_summary", data)
        self.assertGreaterEqual(len(data["detected_environmental_cues"]), 1)

if __name__ == "__main__":
    unittest.main()
