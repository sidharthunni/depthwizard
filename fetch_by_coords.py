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
    print(f"Saved image to {output_path}, size: {img.size}")
    return output_path


if __name__ == "__main__":
    fetch_satellite_image(lat=8.5241, lon=76.9366, box_size_km=3.0, output_path="test_fetch.png")
