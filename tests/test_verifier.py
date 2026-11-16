import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import Verifier, draw_status, is_no_face_error  # noqa: E402

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def wait_for(verifier, status, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if verifier.snapshot()[0] == status:
            return True
        time.sleep(0.01)
    return False


def test_match_and_no_match():
    v = Verifier(lambda f, r: {"verified": True, "distance": 0.2, "threshold": 0.5}, None)
    assert v.snapshot()[0] == "STARTING"
    assert v.submit(FRAME) and wait_for(v, "MATCH")
    assert v.snapshot()[1:] == (0.2, 0.5)

    v = Verifier(lambda f, r: {"verified": False, "distance": 0.9, "threshold": 0.5}, None)
    assert v.submit(FRAME) and wait_for(v, "NO MATCH")


def test_no_face_when_detector_finds_nobody():
    def verify(frame, ref):
        raise ValueError("Face could not be detected. Please confirm that the picture is a face photo.")
    v = Verifier(verify, None)
    assert v.submit(FRAME) and wait_for(v, "NO FACE")


def test_real_errors_are_reported_not_hidden_as_no_face(capsys):
    def verify(frame, ref):
        raise ValueError("An exception occurred while downloading weights")
    v = Verifier(verify, None)
    assert v.submit(FRAME) and wait_for(v, "NO FACE")
    time.sleep(0.05)
    assert "downloading weights" in capsys.readouterr().err
    assert is_no_face_error(ValueError("Face could not be detected")) is True
    assert is_no_face_error(ValueError("something else")) is False
    assert is_no_face_error(RuntimeError("Face could not be detected")) is False


def test_only_one_check_runs_at_a_time():
    gate, started = threading.Event(), threading.Event()

    def slow(frame, ref):
        started.set()
        gate.wait(2)
        return {"verified": True, "distance": 0.1, "threshold": 0.5}

    v = Verifier(slow, None)
    assert v.submit(FRAME)
    assert started.wait(1)
    assert v.submit(FRAME) is False          # busy: this frame is skipped
    gate.set()
    assert wait_for(v, "MATCH")
    time.sleep(0.05)
    assert v.submit(FRAME) is True           # free again


def test_draw_status_does_not_crash_and_marks_the_frame():
    frame = FRAME.copy()
    draw_status(frame, "MATCH", 0.21, 0.5)
    assert frame.any()
    draw_status(frame, "NO FACE", None, None)
