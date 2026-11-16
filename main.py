"""Live face verification: is the person in front of the webcam the person in reference.jpg?

    python main.py                          # webcam vs reference.jpg
    python main.py --reference me.jpg       # use another photo
    python main.py --image photo.jpg        # check one still image instead of the webcam

Press q (or Esc) in the video window to quit.
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import cv2

GREEN, RED, AMBER, GREY = (0, 200, 0), (0, 0, 255), (0, 170, 255), (160, 160, 160)


def is_no_face_error(e: Exception) -> bool:
    """DeepFace signals "nobody in the picture" with a ValueError ("Face could not be detected ...").

    Any other error (a failed model download, a bad image, ...) is a real problem and must not be
    reported as a harmless NO FACE.
    """
    return isinstance(e, ValueError) and "could not be detected" in str(e)


class Verifier:
    """Compares frames with a reference face on a background thread.

    Face verification takes a fraction of a second, far longer than one video frame, so it runs
    off the main loop and at most one check is in flight at a time (frames that arrive while it is
    busy are simply skipped). The video never stutters; the label updates as results arrive.
    """

    def __init__(self, verify_fn, reference):
        self._verify = verify_fn
        self._reference = reference
        self._busy = threading.Lock()
        self._lock = threading.Lock()
        self.status = "STARTING"          # STARTING | MATCH | NO MATCH | NO FACE
        self.distance = None
        self.threshold = None

    def submit(self, frame) -> bool:
        """Start a check on `frame` unless one is already running. Returns True if started."""
        if not self._busy.acquire(blocking=False):
            return False
        threading.Thread(target=self._run, args=(frame.copy(),), daemon=True).start()
        return True

    def _run(self, frame) -> None:
        try:
            result = self._verify(frame, self._reference)
            status = "MATCH" if result["verified"] else "NO MATCH"
            distance, threshold = result.get("distance"), result.get("threshold")
        except Exception as e:             # keep the camera alive, but don't hide real problems
            if not is_no_face_error(e):
                print(f"verification error: {e}", file=sys.stderr)
            status, distance, threshold = "NO FACE", None, None
        finally:
            self._busy.release()
        with self._lock:
            self.status, self.distance, self.threshold = status, distance, threshold

    def snapshot(self):
        with self._lock:
            return self.status, self.distance, self.threshold


def deepface_verify(model_name: str, detector: str):
    """The real verify function, built on DeepFace (imported lazily: it pulls in TensorFlow)."""
    from deepface import DeepFace

    def verify(frame, reference):
        return DeepFace.verify(frame, reference, model_name=model_name, detector_backend=detector,
                               enforce_detection=True)
    return verify


def draw_status(frame, status, distance, threshold):
    colour = {"MATCH": GREEN, "NO MATCH": RED, "NO FACE": AMBER}.get(status, GREY)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), colour, 6)
    cv2.putText(frame, status, (20, h - 24), cv2.FONT_HERSHEY_DUPLEX, 1.6, colour, 3, cv2.LINE_AA)
    if distance is not None:
        cv2.putText(frame, f"distance {distance:.2f} (match if under {threshold:.2f})", (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)


def open_camera(index: int):
    # DirectShow starts much faster on Windows; other platforms use the default backend.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def run_webcam(verifier: Verifier, camera: int, every: int) -> int:
    cap = open_camera(camera)
    if not cap.isOpened():
        print(f"Could not open camera {camera}. Try --camera 1, and close other apps that use it.", file=sys.stderr)
        return 1
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("The camera stopped sending frames.", file=sys.stderr)
                return 1
            if frame_no % every == 0:
                verifier.submit(frame)
            frame_no += 1
            draw_status(frame, *verifier.snapshot())
            cv2.imshow("Face verification  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_image(verify_fn, reference, path: str) -> int:
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image: {path}", file=sys.stderr)
        return 1
    try:
        r = verify_fn(img, reference)
    except Exception as e:
        if not is_no_face_error(e):
            print(f"Verification failed: {e}", file=sys.stderr)
            return 1
        print("NO FACE: no face was found in the image.")
        return 2
    print(f"{'MATCH' if r['verified'] else 'NO MATCH'}  (distance {r['distance']:.3f}, threshold {r['threshold']:.3f})")
    return 0 if r["verified"] else 3


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):      # DeepFace logs emoji; a Windows console can't print them
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Verify a webcam face against a reference photo.")
    ap.add_argument("--reference", default=str(Path(__file__).with_name("reference.jpg")),
                    help="photo of the person to look for (default: reference.jpg next to this script)")
    ap.add_argument("--image", help="check this one image instead of opening the webcam")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--every", type=int, default=15, help="try a new check every N frames (default 15)")
    ap.add_argument("--model", default="SFace",
                    help="DeepFace model: SFace (small, fast, default), Facenet, ArcFace, VGG-Face, ...")
    ap.add_argument("--detector", default="yunet", help="face detector: yunet (default), opencv, retinaface, ...")
    a = ap.parse_args(argv)

    reference = cv2.imread(a.reference)
    if reference is None:
        print(f"Reference photo not found: {a.reference}\n"
              "Put a clear, front-facing photo of the person at that path (or pass --reference).", file=sys.stderr)
        return 1

    verify = deepface_verify(a.model, a.detector)
    if a.image:
        return run_image(verify, reference, a.image)
    return run_webcam(Verifier(verify, reference), a.camera, max(1, a.every))


if __name__ == "__main__":
    sys.exit(main())
