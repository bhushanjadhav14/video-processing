from pathlib import Path

def ensure_model():
    import requests
    import subprocess

    model = "qwen3-vl:8b"
    ollama_url = "http://localhost:11434"

    try:
        response = requests.get(
            f"{ollama_url}/api/tags",
            timeout=10
        )
        response.raise_for_status()

        installed_models = [
            item["name"]
            for item in response.json().get("models", [])
        ]

        if model not in installed_models:
            print(f"Downloading {model}...")
            subprocess.run(
                ["ollama", "pull", model],
                check=True
            )
        else:
            print(f"{model} is already installed.")

    except requests.RequestException as error:
        print("Cannot connect to Ollama. Make sure Ollama is running.")
        raise error

def main():
    print("=== VIDEO TO TEXT: QWEN3-VL 8B ===")

    # Variable for user-provided input video path
    input_video_path = input(
        "Enter the input video path: "
    ).strip().strip('"')

    # Variable for user-provided output text file path
    output_text_path = input(
        "Enter the output text file path: "
    ).strip().strip('"')

    video_path = Path(input_video_path)
    output_path = Path(output_text_path)

    # Validate input video
    if not video_path.is_file():
        print("ERROR: Input video file not found.")
        return

    # Ensure output file has a .txt extension
    if output_path.suffix.lower() != ".txt":
        output_path = output_path.with_suffix(".txt")

    try:
        interval = float(
            input("Sample one frame every N seconds [5]: ").strip() or "5"
        )
        if interval <= 0:
            raise ValueError
    except ValueError:
        print("ERROR: Enter a valid positive interval.")
        return

    include_audio = (
        input("Transcribe spoken audio too? [Y/n]: ").strip().lower()
        not in ("n", "no")
    )

    try:
        ensure_model()

        print("Extracting video frames...")
        frames, duration = extract_frames(video_path, interval)

        if not frames:
            print("ERROR: No video frames extracted.")
            return

        visual_report = analyze_frames(frames)

        audio_report = ""
        if include_audio:
            audio_report = transcribe_audio(video_path)

        # Create the output directory if necessary
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the report to the user-specified output path
        with output_path.open("w", encoding="utf-8") as file:
            file.write("VIDEO ANALYSIS REPORT\n")
            file.write("=" * 60 + "\n")
            file.write(f"Input video: {video_path.resolve()}\n")
            file.write(f"Output file: {output_path.resolve()}\n")
            file.write(f"Duration: {timestamp(duration)}\n\n")

            file.write("VISUAL ANALYSIS\n")
            file.write("=" * 60 + "\n")
            file.write(visual_report)

            if include_audio:
                file.write("\n\nSPEECH TRANSCRIPTION\n")
                file.write("=" * 60 + "\n")
                file.write(audio_report)

        print(f"\nSUCCESS! Report saved to:\n{output_path.resolve()}")

    except Exception as error:
        print(f"ERROR: {error}")


if __name__ == "__main__":
    main()