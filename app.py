import os
from base64 import b64decode, b64encode
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

from datetime import datetime

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

import predict

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key")
SESSION_HISTORY_KEY = "prediction_history"
SESSION_HISTORY_LIMIT = 5


def _purge_uploads() -> None:
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            file_path.unlink(missing_ok=True)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _persist_image(
    upload: Optional[FileStorage],
    capture_payload: Optional[str],
    *,
    persist: bool = True,
) -> Tuple[Optional[Path], Optional[str], Optional[bytes]]:
    if capture_payload:
        try:
            header, encoded = capture_payload.split(",", 1)
        except ValueError as err:
            raise ValueError("Malformed capture payload.") from err

        mime = header.split(";")[0].split(":")[-1]
        subtype = mime.split("/")[-1]
        extension = "jpg" if subtype == "jpeg" else subtype

        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError("Captured image format not supported.")

        binary = b64decode(encoded)
        if not persist:
            return None, None, binary

        filename = f"{uuid4().hex}_capture.{extension}"
        stored_path = UPLOAD_DIR / filename
        stored_path.write_bytes(binary)
        return stored_path, filename, None

    if upload and upload.filename:
        if not _allowed_file(upload.filename):
            raise ValueError("Unsupported file type. Upload PNG, JPG, JPEG, or BMP images only.")

        if not persist:
            upload.stream.seek(0)
            return None, None, upload.read()

        filename = f"{uuid4().hex}_{secure_filename(upload.filename)}"
        stored_path = UPLOAD_DIR / filename
        upload.save(stored_path)
        return stored_path, filename, None

    raise ValueError("Please upload a photo or capture a snapshot before submitting.")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    inline_preview = None
    history = list(session.get(SESSION_HISTORY_KEY, []))
    privacy_opt_out = False

    if request.method == "POST":
        upload = request.files.get("image")
        capture_payload = request.form.get("capture_data", "").strip()
        privacy_opt_out = request.form.get("ephemeral") == "on"
        persist_images = not privacy_opt_out

        try:
            stored_path, filename, raw_bytes = _persist_image(
                upload,
                capture_payload,
                persist=persist_images,
            )
        except ValueError as err:
            flash(str(err))
        else:
            try:
                if raw_bytes is not None:
                    age, gender, mood = predict.get_age_gender_mood_from_bytes(raw_bytes)
                elif stored_path is not None:
                    age, gender, mood = predict.get_age_gender_mood(str(stored_path))
                else:
                    raise RuntimeError("Unable to locate image data.")
                prediction = {
                    "age": age,
                    "gender": gender,
                    "mood": mood,
                }
                if persist_images and stored_path and filename:
                    image_url = url_for("uploaded_file", filename=filename)
                elif capture_payload:
                    inline_preview = capture_payload
                elif raw_bytes is not None:
                    mimetype = upload.mimetype if upload else "image/jpeg"
                    encoded = b64encode(raw_bytes).decode("ascii")
                    inline_preview = f"data:{mimetype};base64,{encoded}"

                entry = {
                    "age": age,
                    "gender": gender,
                    "mood": mood,
                    "image_url": image_url if persist_images else None,
                    "captured_at": datetime.utcnow().strftime("%H:%M:%S UTC"),
                    "id": uuid4().hex,
                }
                history.insert(0, entry)
                session[SESSION_HISTORY_KEY] = history[:SESSION_HISTORY_LIMIT]
                session.modified = True
            except Exception as exc:  # pragma: no cover - surfaced to UI
                flash(f"Could not generate prediction: {exc}")
                if stored_path:
                    stored_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        prediction=prediction,
        image_url=image_url,
        inline_preview=inline_preview,
        history=history[:SESSION_HISTORY_LIMIT],
        privacy_opt_out=privacy_opt_out,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/privacy/clear", methods=["POST"])
def clear_session():
    session.pop(SESSION_HISTORY_KEY, None)
    _purge_uploads()
    flash("Session data cleared and uploads removed.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
