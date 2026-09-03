import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fetch_by_coords import fetch_highres_building_image, fetch_satellite_image

app = FastAPI()
BASE = Path(__file__).parent


class CoordRequest(BaseModel):
    lat: float
    lon: float
    box_size_km: Optional[float] = 3.0
    z_scale: Optional[float] = 45.0
    source: Optional[str] = "auto"  # 'auto' | 'highres' | 'copernicus'


@app.get("/geocode")
def geocode_place(q: str):
    """
    Robust dual-engine geocoding (Nominatim + Photon OSM) for buildings, landmarks, and disaster zones.
    """
    q = q.strip()
    if not q:
        return {"status": "error", "message": "Query string is empty"}

    # 1. Try OSM Nominatim
    try:
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={urllib.parse.quote(q)}"
        req = urllib.request.Request(url, headers={"User-Agent": "DepthWizard/2.0 (SIH26175)"})
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
            if data and len(data) > 0:
                item = data[0]
                return {
                    "status": "ok",
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "name": item.get("display_name", q),
                    "type": item.get("type", "landmark")
                }
    except Exception as e:
        print("Nominatim error:", e)

    # 2. Try Photon (Komoot OSM geocoder)
    try:
        url = f"https://photon.komoot.io/api/?q={urllib.parse.quote(q)}&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "DepthWizard/2.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            pdata = json.loads(r.read())
            if pdata.get("features"):
                feat = pdata["features"][0]
                coords = feat["geometry"]["coordinates"]
                props = feat.get("properties", {})
                name = props.get("name", q)
                city = props.get("city", props.get("state", props.get("country", "")))
                full_name = f"{name}, {city}" if city else name
                return {
                    "status": "ok",
                    "lat": float(coords[1]),
                    "lon": float(coords[0]),
                    "name": full_name,
                    "type": props.get("osm_value", "place")
                }
    except Exception as e:
        print("Photon error:", e)

    return {"status": "not_found", "message": f"Could not locate '{q}'. Try entering coordinates directly."}


def fetch_osm_buildings(lat: float, lon: float, box_size_km: float = 1.0, output_path=None):
    """
    Query OpenStreetMap Overpass API for real 2D vector building footprints within the bounding box.
    Enables LOD-1/LOD-2 3D building extrusion with vertical walls and height attributes.
    """
    half_deg = (box_size_km / 111.0) / 2
    min_lat, max_lat = lat - half_deg, lat + half_deg
    min_lon, max_lon = lon - half_deg, lon + half_deg

    q = f'[out:json][timeout:8];way["building"]({min_lat},{min_lon},{max_lat},{max_lon});out geom;'
    url = "https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(q)
    req = urllib.request.Request(url, headers={"User-Agent": "DepthWizard/2.0 (SIH26175)"})

    buildings = []
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
            for el in data.get("elements", []):
                geom = el.get("geometry", [])
                if len(geom) >= 3:
                    coords = [[pt["lon"], pt["lat"]] for pt in geom]
                    tags = el.get("tags", {})
                    name = tags.get("name", tags.get("building", "Building"))
                    levels = tags.get("building:levels", None)
                    b_type = tags.get("amenity", tags.get("building", "residential"))
                    buildings.append({
                        "id": el.get("id"),
                        "name": name,
                        "type": b_type,
                        "levels": levels,
                        "polygon": coords
                    })
    except Exception as e:
        print("OSM building extraction error (will proceed with fallback):", e)

    if output_path:
        try:
            with open(output_path, "w") as f:
                json.dump({"status": "ok", "count": len(buildings), "buildings": buildings}, f)
            print(f"Extracted {len(buildings)} OSM building footprints to {output_path}")
        except Exception as e:
            print("Error writing buildings json:", e)

    return buildings


@app.get("/fetch-buildings")
def get_buildings(lat: float, lon: float, box_size_km: float = 1.0):
    buildings = fetch_osm_buildings(lat, lon, box_size_km, str(BASE / "current_buildings.json"))
    return {"status": "ok", "count": len(buildings), "buildings": buildings}


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

    lat = float(req.lat)
    lon = float(req.lon)
    box_size_km = float(req.box_size_km) if req.box_size_km is not None else 3.0
    z_scale = float(req.z_scale) if req.z_scale is not None else 45.0
    source = req.source or "auto"

    use_highres = (source == "highres") or (source == "auto" and box_size_km <= 0.8)

    if use_highres:
        print(f"Ingesting sub-meter high-res building aerial for Lat: {lat}, Lon: {lon} ({box_size_km} km footprint)...")
        fetch_highres_building_image(lat=lat, lon=lon, box_size_km=box_size_km, output_path=img_path)
    else:
        print(f"Ingesting Copernicus Sentinel-2 for Lat: {lat}, Lon: {lon} ({box_size_km} km footprint)...")
        fetch_satellite_image(lat=lat, lon=lon, box_size_km=box_size_km, output_path=img_path)

    # Concurrently extract vector building footprints
    fetch_osm_buildings(lat, lon, box_size_km, str(BASE / "current_buildings.json"))

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
        "--z_scale", str(z_scale),
        "--downsample", "4",
        "--box_size_km", str(box_size_km),
        "--lat", str(lat),
        "--lon", str(lon)
    ], cwd=BASE, check=True)

    return {"status": "done", "model": "/current.glb", "source_used": "highres" if use_highres else "sentinel2"}


app.mount("/", StaticFiles(directory=BASE, html=True), name="static")
