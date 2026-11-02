# gui/app.py
# ---------------------------------------------------------------------------
# Local Flask GUI for the EN->MR S2ST baseline pipeline. Runs entirely on
# localhost (see configs/hardware.yaml -> gui.server_name/server_port), no
# external hosting assumptions.
#
# A pipeline run happens in a background thread so the initiating HTTP
# request can return immediately with a job_id, while the browser opens a
# separate Server-Sent Events connection to watch per-stage progress (ASR
# running -> MT running -> TTS running) via EventSource.
#
# GPU discipline: this app never preloads models at startup (S2STPipeline
# only touches the GPU inside run()), and gui/state.py's run_lock allows
# only one pipeline run in flight at a time — baseline/pipeline.py already
# enforces sequential stage loading *within* a run; this extends the same
# discipline across concurrent browser requests on a 6GB card.
# ---------------------------------------------------------------------------

from __future__ import annotations

import csv
import json
import logging
import threading
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file, send_from_directory

from baseline.pipeline import PROJECT_ROOT, S2STPipeline, load_hardware_config
from gui import state
from gui.components import UploadError, result_to_payload, save_upload, sse_event

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

_config = load_hardware_config("configs/hardware.yaml")
_upload_dir = Path(_config["paths"]["cache_dir"]) / "gui_uploads"
_pipeline = S2STPipeline(config_path="configs/hardware.yaml")

# Report page: serves the paper/debugging figures (scripts/generate_report_
# figures.py, scripts/generate_tts_validation_samples.py) from inside the
# GUI. Whitelisted to these two project-root directories only — never serve
# arbitrary paths from user input.
_REPORT_ASSET_DIRS = {
    "figures": PROJECT_ROOT / "figures",
    "samples": PROJECT_ROOT / "samples" / "tts_validation",
    "research": PROJECT_ROOT / "samples" / "research_comparison",
    "full_eval": PROJECT_ROOT / "samples" / "full_evaluation",
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/report")
def report():
    results_csv = _REPORT_ASSET_DIRS["samples"] / "results.csv"
    validation_rows = []
    if results_csv.exists():
        with open(results_csv, newline="", encoding="utf-8") as f:
            validation_rows = list(csv.DictReader(f))

    figures_dir = _REPORT_ASSET_DIRS["figures"]
    available_figures = {p.name for p in figures_dir.glob("*.png")} if figures_dir.exists() else set()

    research_stats_path = _REPORT_ASSET_DIRS["research"] / "comparison_stats.json"
    research_stats = None
    if research_stats_path.exists():
        with open(research_stats_path, "r", encoding="utf-8") as f:
            research_stats = json.load(f)
    research_audio_available = (_REPORT_ASSET_DIRS["research"] / "baseline.wav").exists()

    research_10samples_csv = _REPORT_ASSET_DIRS["research"] / "10samples_results.csv"
    research_10samples_rows = []
    if research_10samples_csv.exists():
        with open(research_10samples_csv, newline="", encoding="utf-8") as f:
            research_10samples_rows = list(csv.DictReader(f))

    full_eval_summary_path = _REPORT_ASSET_DIRS["full_eval"] / "summary.json"
    full_eval_summary = None
    if full_eval_summary_path.exists():
        with open(full_eval_summary_path, "r", encoding="utf-8") as f:
            full_eval_summary = json.load(f)

    full_eval_csv = _REPORT_ASSET_DIRS["full_eval"] / "results.csv"
    full_eval_rows = []
    if full_eval_csv.exists():
        with open(full_eval_csv, newline="", encoding="utf-8") as f:
            full_eval_rows = list(csv.DictReader(f))

    return render_template(
        "report.html",
        validation_rows=validation_rows,
        available_figures=available_figures,
        research_stats=research_stats,
        research_audio_available=research_audio_available,
        research_10samples_rows=research_10samples_rows,
        full_eval_summary=full_eval_summary,
        full_eval_rows=full_eval_rows,
    )


@app.route("/report/asset/<dir_key>/<path:filename>")
def report_asset(dir_key: str, filename: str):
    base_dir = _REPORT_ASSET_DIRS.get(dir_key)
    if base_dir is None or not base_dir.exists():
        abort(404)
    return send_from_directory(base_dir, filename)


@app.route("/api/state")
def api_state():
    """Polled by the page on load so a refresh mid-run still shows the right stage."""
    return jsonify(state.get_current_stage())


@app.route("/api/run", methods=["POST"])
def api_run():
    if not state.run_lock.acquire(blocking=False):
        return jsonify({"error": "Pipeline is busy processing another request. Try again shortly."}), 409

    try:
        audio_path = save_upload(request.files.get("audio"), _upload_dir)
        ref_audio_path = save_upload(request.files.get("ref_audio"), _upload_dir)
        ref_text = request.form.get("ref_text") or None
        mode = request.form.get("mode", "baseline")
    except UploadError as exc:
        state.run_lock.release()
        return jsonify({"error": str(exc)}), 400

    job = state.create_job()

    def status_callback(stage: str, status: str, message: str) -> None:
        state.set_current_stage(stage, status)
        job.events.put(sse_event("stage", {"stage": stage, "status": status, "message": message}))

    def worker() -> None:
        try:
            result = _pipeline.run(
                source_audio_path=audio_path,
                reference_speaker_audio_path=ref_audio_path,
                reference_text=ref_text,
                mode=mode,
                status_callback=status_callback,
            )
            job.result = result
            job.events.put(sse_event("result", result_to_payload(result)))
        except Exception as exc:  # last-resort guard so the SSE stream always terminates
            logger.exception("Unhandled pipeline error")
            job.events.put(sse_event("result", {"error": str(exc)}))
        finally:
            state.set_current_stage("idle", "")
            job.events.put(None)  # sentinel: tells the stream generator to stop
            state.run_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job.job_id})


@app.route("/api/run/<job_id>/stream")
def api_run_stream(job_id: str):
    job = state.get_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404

    def generate():
        while True:
            item = job.events.get()  # blocks until the worker thread pushes something
            if item is None:
                break
            yield item

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/audio/<job_id>")
def api_audio(job_id: str):
    job = state.get_job(job_id)
    if job is None or job.result is None or not job.result.tts_audio_path:
        return jsonify({"error": "No audio available for this job"}), 404
    return send_file(job.result.tts_audio_path, mimetype="audio/wav")


@app.route("/api/audio/<job_id>/baseline")
def api_audio_baseline(job_id: str):
    """Pre-enhancement TTS audio, only present in baseline+research mode."""
    job = state.get_job(job_id)
    if job is None or job.result is None or not job.result.baseline_tts_audio_path:
        return jsonify({"error": "No baseline audio available for this job"}), 404
    return send_file(job.result.baseline_tts_audio_path, mimetype="audio/wav")


if __name__ == "__main__":
    gui_config = _config.get("gui", {})
    app.run(
        host=gui_config.get("server_name", "127.0.0.1"),
        port=gui_config.get("server_port", 7860),
        debug=False,
        threaded=True,
    )
