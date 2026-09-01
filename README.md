# DepthWizard (SIH26175)
Single-view height estimation and 3D flythrough.

Pipeline: RGB image → monocular depth model → textured 3D mesh (.glb) → browser viewer (Three.js)

## Run it
1. `python3 pair_a_depth_model.py --input sample.jpg --output depth_output.npy`
2. `python3 pair_b_mesh_export.py --depth depth_output.npy --image sample.jpg --output terrain.glb`
3. `python3 -m http.server 8000` then open `http://localhost:8000/index.html`
