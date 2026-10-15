"""Atributos visuais e de áudio de cada Short, sem modelo de linguagem.

- Imagem (nos 8 frames): brilho, contraste, saturação e colorido.
- Rostos (YuNet, do OpenCV): presença, quantidade e tamanho no quadro.
- Texto na tela (EasyOCR): presença, área ocupada e palavras lidas.
- Movimento (OpenCV, 2 frames por segundo): diferença média entre frames.
- Cortes de cena (filtro `scdet` do ffmpeg): quantidade e ritmo.
- Áudio (WAV de 16 kHz): energia média, variação e proporção de silêncio.

Métricas "no gancho" usam os 3 primeiros segundos. O resultado de cada vídeo
fica em `data/interim/visual/<video_id>.json` e não é recalculado.
"""

import json
import re
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from creative_predictor import config
from creative_predictor.features.frames import HOOK_TIMES

VISUAL_DIR = config.INTERIM_DIR / "visual"
YUNET_PATH = config.DATA_DIR / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
OCR_LANGUAGES = ["pt", "en"]
OCR_MODELS_DIR = config.DATA_DIR / "models" / "easyocr"
MOTION_FPS = 2.0
SCENE_THRESHOLD = 10.0  # escala do scdet (0–100); 10 separa cortes de movimento
SILENCE_DB = -45.0
HOOK_FRAMES = len(HOOK_TIMES)


def image_stats(image: np.ndarray) -> dict[str, float]:
    """Brilho, contraste, saturação (0–1) e colorido de um frame BGR."""
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = (image[..., i].astype(float) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    # Colorido de Hasler e Süsstrunk (2003): 0 = cinza; acima de ~80 = muito vivo.
    colorfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    return {
        "brightness": float(hsv[..., 2].mean() / 255),
        "contrast": float(gray.std() / 255),
        "saturation": float(hsv[..., 1].mean() / 255),
        "colorfulness": float(colorfulness),
    }


def ensure_yunet(path: Path = YUNET_PATH, url: str = YUNET_URL) -> Path:
    """Baixa o modelo YuNet (repositório oficial do OpenCV) se ainda não existe."""
    if not path.is_file():
        import hashlib
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != YUNET_SHA256:
            tmp.unlink()
            raise RuntimeError(f"modelo YuNet com hash inesperado: {digest}")
        tmp.replace(path)
    return path


class FaceDetector:
    """Rostos com o YuNet; devolve as caixas (x, y, largura, altura)."""

    def __init__(self, model: Path = YUNET_PATH, threshold: float = 0.7):
        import cv2

        self._detector = cv2.FaceDetectorYN.create(
            str(ensure_yunet(model)), "", (320, 320), threshold
        )

    def __call__(self, image: np.ndarray) -> list[tuple[float, float, float, float]]:
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        return [] if faces is None else [tuple(map(float, f[:4])) for f in faces]


class TextReader:
    """Texto na tela com o EasyOCR; devolve (caixa, texto) por trecho lido."""

    def __init__(self, languages: list[str] = OCR_LANGUAGES):
        import easyocr

        # Modelos baixados na 1ª vez para dentro do projeto (fora do git).
        OCR_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._reader = easyocr.Reader(
            languages,
            gpu=False,
            verbose=False,
            model_storage_directory=str(OCR_MODELS_DIR),
        )

    def __call__(self, image: np.ndarray) -> list[tuple[float, str]]:
        """Área (em px²) e texto de cada trecho com confiança mínima."""
        results = self._reader.readtext(image)
        out = []
        for box, text, confidence in results:
            if confidence < 0.4:
                continue
            xs, ys = zip(*box, strict=True)
            out.append(((max(xs) - min(xs)) * (max(ys) - min(ys)), text))
        return out


def frame_features(
    frames: list[np.ndarray],
    detect_faces: Callable[[np.ndarray], list],
    read_text: Callable[[np.ndarray], list],
) -> dict[str, float]:
    """Atributos dos 8 frames; os 3 primeiros são do gancho."""
    stats, face_counts, face_areas, text_areas, words = [], [], [], [], []
    for image in frames:
        area = image.shape[0] * image.shape[1]
        stats.append(image_stats(image))
        faces = detect_faces(image)
        face_counts.append(len(faces))
        face_areas.append(max((w * h for _, _, w, h in faces), default=0.0) / area)
        texts = read_text(image)
        text_areas.append(min(1.0, sum(a for a, _ in texts) / area))
        words.append(sum(len(t.split()) for _, t in texts))
    hook = slice(0, HOOK_FRAMES)
    out = {f"{k}_mean": float(np.mean([s[k] for s in stats])) for k in stats[0]}
    out |= {
        "face_frames_share": float(np.mean([c > 0 for c in face_counts])),
        "face_max_count": float(max(face_counts)),
        "face_area_max": float(max(face_areas)),
        "face_in_hook": float(any(face_counts[hook])),
        "text_frames_share": float(np.mean([a > 0 for a in text_areas])),
        "text_area_mean": float(np.mean(text_areas)),
        "text_words_mean": float(np.mean(words)),
        "text_in_hook": float(any(a > 0 for a in text_areas[hook])),
    }
    return out


def motion_features(video: Path, fps: float = MOTION_FPS) -> dict[str, float]:
    """Diferença média entre frames consecutivos (0–1), no vídeo e no gancho."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    native = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / fps))
    diffs, times, previous, index = [], [], None, 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if index % step == 0:
            gray = cv2.cvtColor(cv2.resize(image, (90, 160)), cv2.COLOR_BGR2GRAY)
            if previous is not None:
                diffs.append(float(np.abs(gray.astype(float) - previous).mean() / 255))
                times.append(index / native)
            previous = gray.astype(float)
        index += 1
    capture.release()
    if not diffs:
        return {"motion_mean": 0.0, "motion_std": 0.0, "motion_hook": 0.0}
    hook = [d for d, t in zip(diffs, times, strict=True) if t <= 3.0] or diffs[:1]
    return {
        "motion_mean": float(np.mean(diffs)),
        "motion_std": float(np.std(diffs)),
        "motion_hook": float(np.mean(hook)),
    }


_SCENE = re.compile(r"lavfi\.scd\.time:\s*([\d.]+)")


def scene_cuts(video: Path, threshold: float = SCENE_THRESHOLD) -> list[float]:
    """Instantes (s) dos cortes de cena detectados pelo ffmpeg."""
    done = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(video),
            "-vf",
            f"scdet=threshold={threshold}",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [float(t) for t in _SCENE.findall(done.stderr)]


def cut_features(cuts: list[float], duration: float) -> dict[str, float]:
    return {
        "cuts": float(len(cuts)),
        "cuts_per_10s": float(10 * len(cuts) / duration) if duration else 0.0,
        "cuts_in_hook": float(sum(t <= 3.0 for t in cuts)),
        "first_cut_s": float(cuts[0]) if cuts else float(duration),
    }


def audio_features(audio: Path | None, frame_s: float = 0.05) -> dict[str, float]:
    """Energia (dBFS) em janelas de 50 ms: média, variação e silêncio."""
    empty = {"audio_db_mean": -90.0, "audio_db_std": 0.0, "silence_share": 1.0}
    if audio is None:
        return empty
    with wave.open(str(audio)) as wav:
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    size = int(rate * frame_s)
    if len(samples) < size:
        return empty
    windows = samples[: len(samples) // size * size].reshape(-1, size) / 32768.0
    rms = np.sqrt((windows**2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-5))
    return {
        "audio_db_mean": float(db.mean()),
        "audio_db_std": float(db.std()),
        "silence_share": float((db < SILENCE_DB).mean()),
    }


class VisualExtractor:
    """Calcula (ou lê do cache) todos os atributos de um Short."""

    def __init__(
        self,
        detect_faces: Callable | None = None,
        read_text: Callable | None = None,
        root: Path = VISUAL_DIR,
    ):
        self._faces, self._text, self.root = detect_faces, read_text, root

    def __call__(
        self,
        video_id: str,
        video: Path,
        frames: list[Path],
        audio: Path | None,
        duration: float,
    ) -> dict:
        path = self.root / f"{video_id}.json"
        if path.is_file():
            return json.loads(path.read_text())
        import cv2

        if self._faces is None:
            self._faces = FaceDetector()
        if self._text is None:
            self._text = TextReader()
        images = [cv2.imread(str(p)) for p in frames]
        unreadable = [
            str(p) for p, image in zip(frames, images, strict=True) if image is None
        ]
        if unreadable:
            raise ValueError(f"{video_id}: frames ilegíveis: {unreadable}")
        result = {
            **frame_features(images, self._faces, self._text),
            **motion_features(video),
            **cut_features(scene_cuts(video), duration),
            **audio_features(audio),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result))
        tmp.replace(path)
        return result
