import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fetch_by_coords import fetch_satellite_image

app = FastAPI()
BASE = Path(__file__).parent

class CoordRequest(BaseModel):
    lat: float
    lon: float
    box_size_km: float = 3.0

@app.post("/upload")
async def upload(file: UploadFile):
    img_path = BASE / "current.jpg"
    with open(img_path, "wb") as f:
        f.write(await file.read())

    subprocess.run(["python3", "pair_a_depth_model.py", "--input", "current.jpg", "--output", "current_depth.npy"], cwd=BASE, check=True)
    subprocess.run(["python3", "pair_b_mesh_export.py", "--depth", "current_depth.npy", "--image", "current.jpg", "--output", "current.glb", "--z_scale", "60", "--downsample", "4"], cwd=BASE, check=True)
    return {"status": "done", "model": "/current.glb"}

@app.post("/fetch-coords")
async def fetch_coords(req: CoordRequest):
    img_path = str(BASE / "current.jpg")
    print(f"Fetching Sentinel-2 for Lat: {req.lat}, Lon: {req.lon} ({req.box_size_km} km box)...")
    fetch_satellite_image(lat=req.lat, lon=req.lon, box_size_km=req.box_size_km, output_path=img_path)

    print("Running depth estimation...")
    subprocess.run(["python3", "pair_a_depth_model.py", "--input", "current.jpg", "--output", "current_depth.npy"], cwd=BASE, check=True)

    print("Exporting 3D terrain mesh...")
    subprocess.run(["python3", "pair_b_mesh_export.py", "--depth", "current_depth.npy", "--image", "current.jpg", "--output", "current.glb", "--z_scale", "60", "--downsample", "4"], cwd=BASE, check=True)

    return {"status": "done", "model": "/current.glb"}

app.mount("/", StaticFiles(directory=BASE, html=True), name="static")
