# Face Recognition

Live face verification from a webcam. You give it one photo of a person; it watches the camera and tells you, a few times a second, whether the face it sees is that person. It is built on [DeepFace](https://github.com/serengil/deepface) for the face model, with the video loop, threading and overlay written by me.

The window shows a green **MATCH**, a red **NO MATCH**, or an amber **NO FACE** (nobody in view), with the measured distance next to the threshold it is compared against.

## Run it

Needs Python 3.9 to 3.12 and a webcam.

```
pip install -r requirements.txt
```

Put a clear, front-facing photo of the person to look for next to `main.py` as **`reference.jpg`**. It is git-ignored on purpose, because it is a photo of a real person and should not end up in a public repo. Then:

```
python main.py                        # webcam vs reference.jpg
python main.py --reference me.jpg     # a different photo
python main.py --image test.jpg       # check one still image and print the verdict (no webcam needed)
```

Press `q` or `Esc` to quit. The first run downloads the model weights (about 40 MB for the default) into `~/.deepface/weights` and is slower to start. If a download is blocked, the error message names the file and the URL to fetch it from by hand.

Checking one image against the reference prints the verdict, for example `MATCH (distance 0.31, threshold 0.59)`. Anything under the threshold is the same person.

| Option | Default | What it does |
| --- | --- | --- |
| `--camera N` | `0` | Which camera to open. |
| `--every N` | `15` | Run a new check every N frames. |
| `--model` | `SFace` | Face model: `SFace` (small and fast), `Facenet`, `ArcFace`, `VGG-Face`, ... |
| `--detector` | `yunet` | Face detector: `yunet`, `opencv`, `retinaface`, ... |

## How it works

1. OpenCV reads frames from the camera and draws the overlay.
2. A check compares the current frame with the reference photo: the detector finds the face, the model turns each face into an embedding (a list of numbers), and `DeepFace.verify` says they are the same person when the distance between the two embeddings is under the model's threshold.
3. A check takes much longer than one video frame, so `Verifier` runs it on a background thread with a lock that allows only **one check in flight**. Frames that arrive while it is busy are skipped, so the video never freezes and threads never pile up. That was the main problem with the first version of this project, which started a new thread every 30 frames regardless.
4. When the detector finds no face, DeepFace raises `ValueError`; that is shown as NO FACE rather than a (misleading) NO MATCH.

## Tests

```
pip install pytest
pytest
```

The tests use a stand-in for DeepFace, so they run instantly and don't need a camera, a model or a photo. They cover the match, no-match and no-face outcomes, and that only one check runs at a time.

## Notes and limits

- This verifies one face against one photo; it is not a "who is this?" search across many people.
- It is a learning project, not a security system. Face verification can be fooled by photos and videos of the person, and accuracy depends on lighting, angle and the quality of the reference photo.
