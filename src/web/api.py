"""Serve the local video workspace and run the existing CLI pipeline."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from threading import BoundedSemaphore, Lock
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware


REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name("static")
MODELS = {
    "public": ("CERBER-CV-v2", "Public YOLOv8 detector · UAV class 2", "public-smoke-cerber-v2.onnx", 2),
    "skyguard": ("SkyGuard YOLO11n", "Public YOLO11n detector · drone class 0", "skyguard-yolo11n.onnx", 0),
    "pilot": ("YOLO11n pilot", "Four-epoch detector pilot · drone class 0", "drone.onnx", 0),
}
PROFILES = {"balanced": "default.yaml", "recall": "recall.yaml"}
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
TRUTH_TYPES = {".txt", ".csv"}
VIDEO_LIMIT = 512 * 1024**2
TRUTH_LIMIT = 10 * 1024**2
MAX_PENDING = 4
ARTIFACTS = {
    "video": "annotated.mp4",
    "predictions": "predictions.txt",
    "detections": "detections.txt",
    "stats": "stats.json",
    "evaluation": "evaluation.json",
}
LOG = logging.getLogger(__name__)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _update_run(directory: Path, **changes) -> dict:
    path = directory / "run.json"
    record = (_read_json(path) or {}) | changes
    staging = directory / ".run.tmp"
    staging.write_text(json.dumps(record), encoding="utf-8")
    staging.replace(path)
    return record


def _file_path(directory: Path, key: str) -> Path:
    if key == "video" and (directory / "preview.mp4").is_file():
        return directory / "preview.mp4"
    return directory / ARTIFACTS[key]


def _detail(directory: Path, record: dict) -> dict:
    files = {}
    if record["status"] == "complete":
        files = {
            key: f"/api/runs/{record['id']}/files/{key}"
            for key in ARTIFACTS
            if _file_path(directory, key).is_file()
        }
    return {"profile": "balanced"} | record | {
        "files": files,
        "stats": _read_json(directory / "stats.json") if "stats" in files else None,
        "evaluation": _read_json(directory / "evaluation.json") if "evaluation" in files else None,
    }


def _save_file(upload: UploadFile, destination: Path, limit: int, extensions: set[str], label: str) -> None:
    if Path(upload.filename or "").suffix.lower() not in extensions:
        raise HTTPException(422, f"Choose a supported {label} file.")
    if upload.size is not None and upload.size > limit:
        raise HTTPException(413, f"The {label} exceeds the {limit // 1024**2} MiB limit.")
    count = 0
    with destination.open("wb") as stream:
        while chunk := upload.file.read(1024**2):
            count += len(chunk)
            if count > limit:
                raise HTTPException(413, f"The {label} exceeds the {limit // 1024**2} MiB limit.")
            stream.write(chunk)


def _command(args: list[str]) -> None:
    result = subprocess.run(args, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace", check=False)
    if result.returncode:
        lines = (result.stderr.strip() or result.stdout.strip()).splitlines()
        raise RuntimeError(lines[-1][:600] if lines else "The process exited before producing a result.")


def _evaluate(directory: Path, truth: Path) -> str | None:
    staging = directory / ".evaluation.json"
    args = [
        sys.executable, "-m", "scripts.evaluate",
        "--ground-truth", str(truth),
        "--predictions", str(directory / "predictions.txt"),
        "--run-stats", str(directory / "stats.json"),
        "--output", str(staging),
    ]
    if (directory / "detections.txt").is_file():
        args.extend(["--detections", str(directory / "detections.txt")])
    try:
        _command(args)
        staging.replace(directory / "evaluation.json")
        return None
    except (OSError, RuntimeError) as exc:
        return str(exc)
    finally:
        staging.unlink(missing_ok=True)


def _prepare_preview(directory: Path) -> str | None:
    staging = directory / ".preview.mp4"
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        _command([
            get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", str(directory / "annotated.mp4"),
            "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-movflags", "+faststart", str(staging),
        ])
        staging.replace(directory / "preview.mp4")
        return None
    except (ImportError, OSError, RuntimeError) as exc:
        LOG.warning("Browser preview for %s failed: %s", directory.name, exc)
        return "Browser preview conversion failed. The original annotated MP4 is available to download."
    finally:
        staging.unlink(missing_ok=True)


def _process(directory: Path, video: Path | None, model: str, truth: Path | None, profile: str) -> None:
    try:
        _update_run(directory, status="processing", message="Running detection and tracking.")
        config = REPO / "configs" / PROFILES[profile]
        args = [sys.executable, "-m", "scripts.run_video", "--config", str(config)]
        if video is None:
            demo = directory / "demo"
            _command([sys.executable, "-m", "scripts.make_demo", str(demo)])
            video = demo / "video.mp4"
            truth = directory / "ground-truth.txt"
            shutil.copyfile(demo / "gt" / "gt.txt", truth)
            args.extend(["--annotations", str(truth), "--expected-frames", "365"])
        else:
            _, _, model_path, class_id = MODELS[model]
            args.extend([
                "--model", str(REPO / "models" / model_path), "--class-id", str(class_id),
                "--detections", str(directory / "detections.txt"),
            ])
        args.extend([
            "--video", str(video), "--predictions", str(directory / "predictions.txt"),
            "--annotated", str(directory / "annotated.mp4"), "--stats", str(directory / "stats.json"),
        ])
        _command(args)
        _update_run(directory, message="Preparing the annotated video and evaluation.")
        message = _prepare_preview(directory)
        evaluation_error = _evaluate(directory, truth) if truth else None
        _update_run(directory, status="complete", message=message, evaluation_error=evaluation_error)
    except (OSError, RuntimeError) as exc:
        _update_run(directory, status="failed", message=str(exc))


def _interrupted_runs(root: Path) -> None:
    for path in root.glob("*/run.json"):
        try:
            if UUID(path.parent.name).hex != path.parent.name or path.parent.is_symlink():
                continue
            record = _read_json(path)
            if record is None:
                continue
            if record["status"] in {"queued", "processing"}:
                _update_run(path.parent, status="failed", message="The server stopped before this run finished.")
            elif record.get("evaluation_pending"):
                _update_run(path.parent, evaluation_pending=False, evaluation_error="The server stopped during evaluation.")
        except (OSError, ValueError, KeyError):
            LOG.warning("Unable to read run metadata in %s", path.parent)


def create_app(output_root: Path | None = None) -> FastAPI:
    root = (output_root or REPO / "outputs/ui").resolve()
    slots = BoundedSemaphore(MAX_PENDING)
    evaluation_lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        root.mkdir(parents=True, exist_ok=True)
        _interrupted_runs(root)
        pool = ThreadPoolExecutor(max_workers=1)
        app.state.pool = pool
        try:
            yield
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Drone Vision local workspace", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
    app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")

    @app.middleware("http")
    async def check_upload(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin is not None and origin != f"http://{request.headers.get('host')}":
                return JSONResponse({"detail": "Uploads must come from the local workspace."}, status_code=403)
            limit = TRUTH_LIMIT if request.url.path.endswith("/evaluation") else VIDEO_LIMIT + TRUTH_LIMIT
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    if int(length) > limit + 1024**2:
                        return JSONResponse({"detail": "The upload exceeds the size limit."}, status_code=413)
                except ValueError:
                    return JSONResponse({"detail": "Invalid content length."}, status_code=400)
        return await call_next(request)

    def reserve() -> None:
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "The processing queue is full. Try again when a run finishes.")

    def enqueue(fn, *args) -> None:
        future = app.state.pool.submit(fn, *args)
        future.add_done_callback(lambda _: slots.release())

    def find(run_id: UUID) -> tuple[Path, dict]:
        directory = root / run_id.hex
        record = None if directory.is_symlink() else _read_json(directory / "run.json")
        if record is None:
            raise HTTPException(404, "Run not found.")
        return directory, record

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/options")
    def options():
        return {
            "models": [
                {"id": key, "label": label, "description": description}
                for key, (label, description, filename, _) in MODELS.items()
                if (REPO / "models" / filename).is_file()
            ],
            "demo_available": True,
        }

    @app.get("/api/runs")
    def runs():
        records = []
        for path in root.glob("*/run.json"):
            if path.parent.is_symlink():
                continue
            try:
                directory, record = find(UUID(path.parent.name))
                records.append(_detail(directory, record))
            except (ValueError, HTTPException):
                continue
        return sorted(records, key=lambda record: record["created_at"], reverse=True)

    @app.get("/api/runs/{run_id}")
    def run_info(run_id: UUID):
        directory, record = find(run_id)
        return _detail(directory, record)

    @app.post("/api/runs", status_code=202)
    def create_run(
        video: Annotated[UploadFile | None, File()] = None,
        ground_truth: Annotated[UploadFile | None, File()] = None,
        model: Annotated[str, Form()] = "public",
        demo: Annotated[bool, Form()] = False,
        profile: Annotated[str, Form()] = "balanced",
    ):
        if profile not in PROFILES:
            raise HTTPException(422, "Choose the balanced or recall processing profile.")
        if demo and (video or ground_truth):
            raise HTTPException(422, "The synthetic demo supplies its own video and ground truth.")
        if not demo:
            if video is None:
                raise HTTPException(422, "Choose a video to process.")
            if model not in MODELS or not (REPO / "models" / MODELS[model][2]).is_file():
                raise HTTPException(422, "Choose an available model.")
        reserve()
        run_id = uuid4().hex
        directory = root / run_id
        run_profile = "balanced" if demo else profile
        try:
            directory.mkdir()
        except OSError:
            slots.release()
            raise
        try:
            input_video = annotations = None
            if video:
                input_video = directory / f"source{Path(video.filename or '').suffix.lower()}"
                _save_file(video, input_video, VIDEO_LIMIT, VIDEO_TYPES, "video")
                if input_video.stat().st_size == 0:
                    raise HTTPException(422, "The video file is empty.")
            if ground_truth:
                annotations = directory / "ground-truth.txt"
                _save_file(ground_truth, annotations, TRUTH_LIMIT, TRUTH_TYPES, "ground-truth")
            name = "Synthetic re-entry demo" if demo else Path((video.filename or "Video").replace("\\", "/")).name[:150]
            record = _update_run(
                directory, id=run_id, name=name, model="Oracle tracker demo" if demo else MODELS[model][0],
                profile=run_profile,
                status="queued", created_at=datetime.now(timezone.utc).isoformat(), message="Waiting for the worker.",
                evaluation_pending=False, evaluation_error=None,
            )
            enqueue(_process, directory, input_video, model, annotations, run_profile)
        except Exception:
            slots.release()
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return _detail(directory, record)

    @app.post("/api/runs/{run_id}/evaluation", status_code=202)
    def reevaluate(run_id: UUID, ground_truth: Annotated[UploadFile, File()]):
        with evaluation_lock:
            directory, record = find(run_id)
            if record["status"] != "complete" or record.get("evaluation_pending"):
                raise HTTPException(409, "Wait for the current run or evaluation to finish.")
            if _read_json(directory / "stats.json")["detector_mode"] == "oracle":
                raise HTTPException(422, "The synthetic demo includes its own evaluation.")
            reserve()
            annotations = directory / ".pending-truth.txt"
            try:
                _save_file(ground_truth, annotations, TRUTH_LIMIT, TRUTH_TYPES, "ground-truth")
                record = _update_run(directory, evaluation_pending=True, evaluation_error=None)

                def evaluate_again() -> None:
                    error = None
                    try:
                        error = _evaluate(directory, annotations)
                        if error is None:
                            annotations.replace(directory / "ground-truth.txt")
                    except OSError as exc:
                        error = str(exc)
                    finally:
                        annotations.unlink(missing_ok=True)
                    _update_run(directory, evaluation_pending=False, evaluation_error=error)

                enqueue(evaluate_again)
            except Exception:
                slots.release()
                annotations.unlink(missing_ok=True)
                _update_run(directory, evaluation_pending=False)
                raise
        return _detail(directory, record)

    @app.api_route("/api/runs/{run_id}/files/{kind}", methods=["GET", "HEAD"])
    def artifact(run_id: UUID, kind: str):
        directory, record = find(run_id)
        if kind not in ARTIFACTS or record["status"] != "complete":
            raise HTTPException(404, "Artifact not found.")
        path = _file_path(directory, kind)
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "Artifact not found.")
        return FileResponse(path, filename=ARTIFACTS[kind],
                            content_disposition_type="inline" if kind == "video" else "attachment")

    return app
