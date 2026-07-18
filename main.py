import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from DarkIR.download_model import (
    LoadedModel,
    load_ready_model,
    shutdown_ready_model,
)
from DarkIR.inference import run_low_light_img_inference
from DarkIR.inference_video import run_low_light_video_inference
from Job.database import create_job, get_job_by_id, get_jobs_by_user_id
from Job.models import Job,JobType, JobStatus, JobResponse
from Authentication.security import get_current_user
from Authentication.database import get_db
from Authentication.models import User
from sqlalchemy.orm import Session
_loaded_model: LoadedModel | None = None
import threading

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loaded_model
    os.makedirs("Source", exist_ok=True)
    os.makedirs("Target", exist_ok=True)
    print("Loading DarkIR model...")
    _loaded_model = load_ready_model()
    print("DarkIR model loaded.")
    yield
    print("Shutting down DarkIR model...")
    shutdown_ready_model()
    _loaded_model = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
JOP_IMG=1
JOP_VIDEO=1




@app.post("/inference/video")
async def inference_video_upload(file: UploadFile = File(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if _loaded_model is None:
        raise RuntimeError("Inference model is not loaded")

    os.makedirs("uploads", exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".mp4"
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]:
        raise HTTPException(status_code=400, detail="Invalid file extension you should choose video only")
    in_name = f"{os.path.splitext(file.filename or 'video')[0]}_{JOP_VIDEO}{ext}"
    JOP_VIDEO+=1
    in_path = os.path.join("uploads", in_name)

    contents = await file.read()
    with open(in_path, "wb") as f:
        f.write(contents)
    out_path = os.path.join("Target", in_name.replace(".mp4", "_output.mp4"))
    job = Job(user_id=user.id, type=JobType.VIDEO, status=JobStatus.PENDING, source_path=in_path, target_path=out_path)
    create_job(db, job)
    threading.Thread(target=run_low_light_video_inference, args=(in_path,out_path, _loaded_model)).start()
    return JobResponse(id=job.id, status=job.status)



@app.post("/inference/image")
async def inference_image_upload(file: UploadFile = File(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if _loaded_model is None:
        raise RuntimeError("Inference model is not loaded")

    os.makedirs("Source", exist_ok=True)
    os.makedirs("Target", exist_ok=True)

    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    if ext not in [".jpg", ".png", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file extension you should choose image only")
    in_name = f"{os.path.splitext(file.filename or 'image')[0]}_{JOP_IMG}{ext}"
    JOP_IMG+=1
    in_path = os.path.join("Source", in_name)
    out_path = os.path.join("Target", in_name.replace(".jpg", "_output.jpg"))

    contents = await file.read()
    with open(in_path, "wb") as f:
        f.write(contents)
    job = Job(user_id=user.id, type=JobType.IMAGE, status=JobStatus.PENDING, source_path=in_path, target_path=out_path)
    create_job(db, job)
    threading.Thread(target=run_low_light_img_inference, args=(in_path, out_path, _loaded_model)).start()
    return JobResponse(id=job.id, status=job.status)

@app.get("/job/{job_id}")
async def get_job(job_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = get_job_by_id(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != user.id or job.is_seen:
        raise HTTPException(status_code=403, detail="You are not authorized to access this job")
    if job.status == JobStatus.COMPLETED and  not job.is_seen:
        job.is_seen = True
        db.commit()
        return FileResponse(job.target_path, media_type="video/mp4" if job.type == JobType.VIDEO else "image/jpeg", filename=os.path.basename(job.target_path))
    if job.status == JobStatus.FAILED:
        job.is_seen = True
        db.commit()
        return JobResponse(id=job.id, status=job.status)
    if job.status == JobStatus.PENDING: 
        return JobResponse(id=job.id, status=job.status, description=job.description)

@app.get("/jobs")
async def get_jobs(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    jobs = get_jobs_by_user_id(db, user.id)
    return [JobResponse(id=job.id, status=job.status) for job in jobs]
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
