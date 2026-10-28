
import base64
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# Visual model: Qwen3-VL 235B

VLM_MODEL = "qwen3-vl:235b"
WHISPER_MODEL = "large-v3"
OLLAMA_URL = "http://localhost:11434"

FRAME_INTERVAL = 3.0
MAX_IMAGE_SIZE = 1280
REQUEST_TIMEOUT = 600


def ensure_python_package(package, import_name=None):
    """Install a missing Python dependency."""
    import_name = import_name or package

    if importlib.util.find_spec(import_name) is None:
        print(f"Installing Python package: {package}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package
        ])


def ensure_dependencies():
    ensure_python_package("requests")
    ensure_python_package("opencv-python", "cv2")
    ensure_python_package("ollama")
    ensure_python_package("faster-whisper")


def ensure_ollama():
    """Verify that Ollama is installed and running."""
    import requests

    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags", timeout=5
        )
        response.raise_for_status()
        return

    except requests.RequestException:
        pass

    # Try starting the Ollama executable if it is installed.
    import shutil

    executable = shutil.which("ollama")

    if not executable and os.name == "nt":
        possible_path = (
            Path.home() / "AppData" / "Local" /
            "Programs" / "Ollama" / "ollama.exe"
        )
        if possible_path.is_file():
            executable = str(possible_path)

    if not executable:
        raise RuntimeError(
            "Ollama is not installed. Install the Ollama desktop "
            "application first, then run this script again."
        )

    print("Starting Ollama...")
    subprocess.Popen(
        [executable, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    for _ in range(30):
        time.sleep(2)
        try:
            response = requests.get(
                f"{OLLAMA_URL}/api/tags", timeout=3
            )
            response.raise_for_status()
            return
        except requests.RequestException:
            continue

    raise RuntimeError(
        "Ollama did not start. Check the Ollama application."
    )


def ensure_vlm_model():
    """Download the vision-language model if missing."""
    import requests

    response = requests.get(
        f"{OLLAMA_URL}/api/tags", timeout=15
    )
    response.raise_for_status()

    installed = {
        item["name"]
        for item in response.json().get("models", [])
    }

    if VLM_MODEL in installed:
        print(f"VLM already installed: {VLM_MODEL}")
        return

    print(f"Downloading {VLM_MODEL}. This is a very large download.")

    subprocess.run(
        ["ollama", "pull", VLM_MODEL],
        check=True
    )

    print("VLM download completed.")


def extract_frames(video_path, interval):
    """Extract frames with timestamps."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Cannot open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS.")

    duration = count / fps
    frames = []
    next_time = 0.0

    while next_time < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, next_time * 1000)
        ok, frame = cap.read()

        if ok:
            height, width = frame.shape[:2]
            scale = min(1.0, MAX_IMAGE_SIZE / max(height, width))

            if scale < 1:
                frame = cv2.resize(
                    frame,
                    (int(width * scale), int(height * scale))
                )

            success, encoded = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

            if success:
                frames.append((
                    next_time,
                    base64.b64encode(encoded).decode("utf-8")
                ))

        next_time += interval

    cap.release()
    return frames, duration


def analyze_frame(timestamp, image_base64):
    """Analyze one frame using local Ollama inference."""
    import ollama

    prompt = (
        f"This frame is from a video at {timestamp:.2f} seconds. "
        "Describe only what is visually supported by the frame. "
        "Identify people, actions, objects, scene changes and visible "
        "text. Explain uncertainty instead of guessing. "
        "Do not infer spoken dialogue or events outside this frame."
    )

    response = ollama.chat(
        model=VLM_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_base64]
        }],
        options={"temperature": 0.2}
    )

    return response["message"]["content"]


def transcribe_audio(video_path):
    """Transcribe the video's audio using Faster-Whisper."""
    from faster_whisper import WhisperModel

    print(f"Loading Whisper model: {WHISPER_MODEL}")
    print("Whisper downloads its model automatically if missing.")

    try:
        import ctranslate2
        use_gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        use_gpu = False

    if use_gpu:
        model = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="float16"
        )
    else:
        print("CUDA not detected; using CPU int8 transcription.")
        model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

    segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        vad_filter=True
    )

    transcript = []

    for segment in segments:
        transcript.append(
            f"[{segment.start:.2f}s - {segment.end:.2f}s] "
            f"{segment.text.strip()}"
        )

    return info.language, "\n".join(transcript)


def main():
    print("=== VIDEO AND AUDIO INTELLIGENCE ===")
    print(f"Visual model: {VLM_MODEL}")
    print(f"Audio model: {WHISPER_MODEL}")

    video_path = Path(
        input("Input video path: ").strip().strip('"')
    )

    output_path = Path(
        input("Output report path: ").strip().strip('"')
    )

    if not video_path.is_file():
        print("ERROR: Video file not found.")
        return

    if output_path.suffix.lower() != ".txt":
        output_path = output_path.with_suffix(".txt")

    try:
        ensure_dependencies()
        ensure_ollama()
        ensure_vlm_model()

        frames, duration = extract_frames(
            video_path, FRAME_INTERVAL
        )

        if not frames:
            raise RuntimeError("No frames could be extracted.")

        print(f"Duration: {duration:.1f} seconds")
        print(f"Frames to analyze: {len(frames)}")

        visual_results = []

        for index, (ts, image) in enumerate(frames, start=1):
            print(f"Analyzing frame {index}/{len(frames)}...")
            description = analyze_frame(ts, image)

            visual_results.append(
                f"[{ts:.2f}s]\n{description}"
            )

        print("Transcribing audio...")
        language, transcript = transcribe_audio(video_path)

        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        with output_path.open("w", encoding="utf-8") as file:
            file.write("VIDEO INTELLIGENCE REPORT\n")
            file.write("=" * 65 + "\n")
            file.write(f"Input: {video_path.resolve()}\n")
            file.write(f"Duration: {duration:.2f} seconds\n")
            file.write(f"Visual model: {VLM_MODEL}\n")
            file.write(f"Audio model: Faster-Whisper {WHISPER_MODEL}\n")
            file.write(f"Detected audio language: {language}\n\n")

            file.write("VISUAL ANALYSIS\n")
            file.write("=" * 65 + "\n\n")
            file.write("\n\n".join(visual_results))

            file.write("\n\n\nSPEECH TRANSCRIPTION\n")
            file.write("=" * 65 + "\n\n")
            file.write(transcript or "No speech detected.")

        print(f"\nSUCCESS: Report saved to {output_path.resolve()}")

    except Exception as error:
        print(f"\nERROR: {error}")


if __name__ == "__main__":
    main()
