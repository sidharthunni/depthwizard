import urllib.request
from sentinelhub import (
    SHConfig, SentinelHubRequest, DataCollection,
    MimeType, CRS, BBox, bbox_to_dimensions
)
from PIL import Image
import numpy as np

config = SHConfig('cdse')

# Fix: define a NEW collection pointed at CDSE instead of rebinding the existing one
CDSE_SENTINEL2_L2A = DataCollection.SENTINEL2_L2A.define_from(
    "CDSE_SENTINEL2_L2A", service_url=config.sh_base_url
)


def fetch_satellite_image(lat, lon, box_size_km=2.0, resolution=10, output_path='fetched_image.png'):
    """Fetch regional multispectral satellite imagery from ESA Copernicus Sentinel-2."""
    half_deg = (box_size_km / 111.0) / 2
    bbox = BBox(
        bbox=[lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg],
        crs=CRS.WGS84
    )
    size = bbox_to_dimensions(bbox, resolution=resolution)

    evalscript = """
    //VERSION=3
    function setup() {
        return { input: ["B02","B03","B04"], output: { bands: 3 } };
    }
    function evaluatePixel(sample) {
        return [sample.B04*2.5, sample.B03*2.5, sample.B02*2.5];
    }
    """

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=CDSE_SENTINEL2_L2A,
            time_interval=("2026-01-01", "2026-09-01"),
            mosaicking_order="leastCC"
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config
    )

    image = request.get_data()[0]

    img = Image.fromarray(np.clip(image, 0, 255).astype("uint8"))
    img.save(output_path)
    print(f"Saved Sentinel-2 image to {output_path}, size: {img.size}")
    return output_path


def fetch_highres_building_image(lat, lon, box_size_km=0.5, output_path="current.jpg"):
    """
    Fetch sub-meter (30-50cm) high-resolution aerial imagery via public ESRI World Imagery.
    Zero authentication / API keys needed. Ideal for single buildings, rooftop helipads, and barriers.
    """
    half_deg = (box_size_km / 111.0) / 2
    min_lon = lon - half_deg
    max_lon = lon + half_deg
    min_lat = lat - half_deg
    max_lat = lat + half_deg

    url = (
        f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export?"
        f"bbox={min_lon},{min_lat},{max_lon},{max_lat}&bboxSR=4326&imageSR=4326"
        f"&size=750,750&format=png&f=image"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "DepthWizard/2.0 (SIH26175)"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()
        with open(output_path, "wb") as f:
            f.write(data)
    print(f"Saved sub-meter high-res building aerial image to {output_path} ({len(data)} bytes)")
    return output_path


if __name__ == "__main__":
    fetch_satellite_image(lat=8.5241, lon=76.9366, box_size_km=3.0, output_path="test_fetch.png")
