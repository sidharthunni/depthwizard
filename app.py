import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fetch_by_coords import fetch_satellite_image, fetch_highres_building_image

app = FastAPI()
BASE = Path(__file__).parent


class CoordRequest(BaseModel):
    lat: float
    lon: float
    box_size_km: float = 3.0
    z_scale: float = 45.0
    source: str = "auto"  # 'auto' | 'highres' | 'copernicus'


@app.post("/upload")
async def upload(file: UploadFile):
    img_path = BASE / "current.jpg"
    with open(img_path, "wb") as f:
        f.write(await file.read())

    subprocess.run([
        "python3", "pair_a_depth_model.py",
        "--input", "current.jpg",
        "--output", "current_depth.npy"
    ], cwd=BASE, check=True)

    subprocess.run([
        "python3", "pair_b_mesh_export.py",
        "--depth", "current_depth.npy",
        "--image", "current.jpg",
        "--output", "current.glb",
        "--z_scale", "35",
        "--downsample", "4",
        "--box_size_km", "2.0",
        "--lat", "11.5277",
        "--lon", "76.1950"
    ], cwd=BASE, check=True)

    return {"status": "done", "model": "/current.glb"}


@app.post("/fetch-coords")
async def fetch_coords(req: CoordRequest):
    img_path = str(BASE / "current.jpg")

    # Smart Multi-Source Routing:
    # If box <= 0.8km or source is explicitly 'highres', use sub-meter aerial imagery (ESRI / World Imagery)
    # Otherwise use regional Copernicus Sentinel-2 L2A multispectral data
    use_highres = (req.source == "highres") or (req.source == "auto" and req.box_size_km <= 0.8)

    if use_highres:
        print(f"Ingesting sub-meter high-res building aerial for Lat: {req.lat}, Lon: {req.lon} ({req.box_size_km} km footprint)...")
        fetch_highres_building_image(lat=req.lat, lon=req.lon, box_size_km=req.box_size_km, output_path=img_path)
    else:
        print(f"Ingesting Copernicus Sentinel-2 for Lat: {req.lat}, Lon: {req.lon} ({req.box_size_km} km footprint)...")
        fetch_satellite_image(lat=req.lat, lon=req.lon, box_size_km=req.box_size_km, output_path=img_path)

    print("Running depth estimation...")
    subprocess.run([
        "python3", "pair_a_depth_model.py",
        "--input", "current.jpg",
        "--output", "current_depth.npy"
    ], cwd=BASE, check=True)

    print("Exporting calibrated 3D terrain mesh...")
    subprocess.run([
        "python3", "pair_b_mesh_export.py",
        "--depth", "current_depth.npy",
        "--image", "current.jpg",
        "--output", "current.glb",
        "--z_scale", str(req.z_scale),
        "--downsample", "4",
        "--box_size_km", str(req.box_size_km),
        "--lat", str(req.lat),
        "--lon", str(req.lon)
    ], cwd=BASE, check=True)

    return {"status": "done", "model": "/current.glb", "source_used": "highres" if use_highres else "sentinel2"}


app.mount("/", StaticFiles(directory=BASE, html=True), name="static")
