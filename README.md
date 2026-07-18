# DarkIR Deployment

Low-light image and video restoration powered by **[DarkIR](https://github.com/Fundacion-Cidaut/DarkIR)** (CVPR 2025), wrapped as a FastAPI service with authentication, Postgres, and an Nginx gateway.

You can use this repo in two ways:

1. **Full project** — auth + database + inference API (Docker + local Python)
2. **DarkIR only** — research/inference scripts inside `DarkIR/` without the web stack

---

## Project layout

```
.
├── main.py                 # Inference API (images & videos)
├── docker-compose.yml      # Postgres, pgAdmin, auth-service, Nginx
├── requirements.txt        # Full-stack Python deps (includes DarkIR + FastAPI)
├── Authentication/         # JWT auth service
├── Database/               # Postgres / pgAdmin Dockerfiles & env
├── nginx/                  # API gateway
├── Job/                    # Job models & DB helpers
├── DarkIR/                 # DarkIR model, weights download, CLI inference
├── Source/                 # Uploaded images (created on server start)
└── Target/                 # Restored outputs (created on server start)
```

---

## Requirements

| Component | Suggested |
|-----------|-----------|
| Python | 3.10+ |
| CUDA (optional but recommended) | 12.x with matching PyTorch |
| Docker + Docker Compose | for Postgres, auth, gateway |
| GPU RAM | enough for DarkIR weights (CUDA preferred) |

Model weights are downloaded automatically from Hugging Face (`Cidaut/DarkIR`) on first run if missing under `DarkIR/models/`.

---

# Option A — Use the whole project

### 1. Clone and create a virtualenv

```bash
cd "DarkIR Deployment"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r Authentication/requirements_auth.txt
```

> `requirements.txt` includes PyTorch and DarkIR deps. Auth extras live in `Authentication/requirements_auth.txt`.

### 2. Configure environment files

Secrets are **not** committed. Copy the examples:

```bash
cp Database/.env.example Database/.env
cp Authentication/.env.example Authentication/.env
```

Edit them before production use:

**`Database/.env`**

```env
POSTGRES_USER=admin
POSTGRES_PASSWORD=secret
POSTGRES_DB=mydb
PGADMIN_DEFAULT_EMAIL=admin@admin.com
PGADMIN_DEFAULT_PASSWORD=admin
```

**`Authentication/.env`**

```env
SECRET_KEY=change-me-to-a-long-random-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

`Authentication/config.py` loads both files. Local defaults connect to Postgres on `localhost:5433`. Docker Compose overrides host/port to `postgres:5432` for the auth container.

### 3. Start infrastructure (Docker)

```bash
docker compose up -d --build
```

| Service | URL / port |
|---------|------------|
| Nginx gateway | http://localhost:80 |
| Auth (via gateway) | http://localhost/auth/... |
| Postgres | `localhost:5433` |
| pgAdmin | http://localhost:5050 |

### 4. Start the inference API

From the **project root** (so imports and folders resolve correctly):

```bash
source .venv/bin/activate
python main.py
```

This:

- creates `Source/` and `Target/` if missing
- downloads DarkIR weights if needed
- serves FastAPI on **http://0.0.0.0:8000**

OpenAPI docs: http://localhost:8000/docs

### 5. Typical API flow

1. **Sign up / log in** (auth service, through gateway or directly if exposed):

   - `POST /auth/signup`
   - `POST /auth/login` → receive `access_token`

2. **Run inference** (inference server on port 8000), with `Authorization: Bearer <access_token>`:

   | Method | Path | Description |
   |--------|------|-------------|
   | `POST` | `/inference/image` | Upload a low-light image (`.jpg`, `.png`, `.jpeg`) |
   | `POST` | `/inference/video` | Upload a video (`.mp4`, `.avi`, `.mov`, …) |
   | `GET` | `/job/{job_id}` | Poll job; returns file when completed |
   | `GET` | `/jobs` | List jobs for the current user |

Jobs run in background threads. Uploaded images go to `Source/`; outputs go to `Target/`. Videos currently land under `uploads/` and `Target/`.

Example (after login):

```bash
TOKEN="<access_token>"

curl -X POST "http://localhost:8000/inference/image" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/dark.jpg"

curl -X GET "http://localhost:8000/job/1" \
  -H "Authorization: Bearer $TOKEN" \
  --output restored.jpg
```

### 6. Stop Docker stack

```bash
docker compose down
```

Data in the `database_pgdata` volume is kept unless you remove volumes explicitly.

---

# Option B — Use only the `DarkIR/` directory

Use this if you want model inference or evaluation **without** auth, Postgres, or Docker.

Upstream paper & model: [DarkIR (CVPR 2025)](https://arxiv.org/abs/2412.13443) · [DarkIR/README.md](DarkIR/README.md) · [Hugging Face](https://huggingface.co/Cidaut/DarkIR)

> In this deployment fork, `inference.py` / `inference_video.py` are **programmatic APIs** (no `-i` CLI). Call them from Python as shown below. Upstream-style CLI docs still appear in `DarkIR/README.md` for reference.

### 1. Install

From the **project root** (package imports use `DarkIR.`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or only DarkIR deps:
# pip install -r DarkIR/requirements.txt
```

### 2. Weights

On first load, weights are fetched from Hugging Face (`Cidaut/DarkIR`) into `DarkIR/models/` if missing. You can also place checkpoints there manually ([OneDrive mirror](https://cidautes-my.sharepoint.com/:f:/g/personal/alvgar_cidaut_es/Epntbl4SucFNpeIT_jyYZ-cB9BamMbacbyq_svrkMCpShA?e=XB9YBB)).

### 3. Image inference (Python)

Run from the project root with the venv active:

```python
from DarkIR.download_model import load_ready_model, shutdown_ready_model
from DarkIR.inference import run_low_light_img_inference

model = load_ready_model()
run_low_light_img_inference(
    "path/to/dark.jpg",
    "path/to/restored.jpg",
    loaded_model=model,
)
shutdown_ready_model()
```

Omit `loaded_model` to load weights for a one-off call (slower).

### 4. Video inference (Python)

```python
from DarkIR.download_model import load_ready_model, shutdown_ready_model
from DarkIR.inference_video import run_low_light_video_inference

model = load_ready_model()
run_low_light_video_inference(
    "path/to/input.mp4",
    "path/to/output.mp4",
    loaded_model=model,
)
shutdown_ready_model()
```

### 5. Evaluation (optional)

```bash
cd DarkIR
# Put datasets under DarkIR/data/datasets/
python testing.py -p ./options/test/LOLBlur.yml
python testing_unpaired.py -p ./options/test/RealBlur_Night.yml
```

### 6. Gradio demo (optional)

```bash
cd DarkIR
python app.py
```

More datasets, metrics, and citation details: [`DarkIR/README.md`](DarkIR/README.md).

---

## Environment & secrets notes

- Real `.env` files are gitignored; only `.env.example` is tracked.
- Docker builds ignore `.env` via `.dockerignore`; Compose injects env at **runtime** with `env_file`.
- Change `SECRET_KEY` and DB passwords before any shared or production deployment.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Auth cannot reach Postgres locally | Ensure `docker compose up` is running and port `5433` is free; config defaults to `localhost:5433`. |
| CUDA / torch errors | Install a PyTorch build that matches your CUDA driver, or run on CPU (slower). |
| Weights missing | First run needs network access to Hugging Face; check `DarkIR/models/`. |
| Import errors for `DarkIR` / `Authentication` | Run `python main.py` from the **project root**, with the venv activated. |
| `Source` / `Target` missing | They are created automatically when `main.py` starts. |

---

## Citation

If you use DarkIR in research, please cite the original paper:

```bibtex
@InProceedings{Feijoo_2025_CVPR,
    author    = {Feijoo, Daniel and Benito, Juan C. and Garcia, Alvaro and Conde, Marcos V.},
    title     = {DarkIR: Robust Low-Light Image Restoration},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {10879-10889}
}
```
