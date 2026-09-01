import argparse
import numpy as np
import trimesh
from PIL import Image


def build_mesh(depth_path, image_path, output_path, z_scale=50.0, downsample=4):
    depth = np.load(depth_path)
    image = Image.open(image_path).convert("RGB")

    # Resize image to match depth array exactly (safety, in case of mismatch)
    image = image.resize((depth.shape[1], depth.shape[0]))
    img_arr = np.array(image)

    # Downsample for a lighter mesh (full-res = way too many triangles)
    depth_ds = depth[::downsample, ::downsample]
    img_ds = img_arr[::downsample, ::downsample]

    h, w = depth_ds.shape
    print(f"Building mesh at {w}x{h} resolution...")

    # Normalize depth so height variation is visually reasonable
    d = depth_ds.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    z = d * z_scale

    # Build grid of vertices: x, y from pixel coords, z from depth
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    vertices = np.stack([xs.flatten(), -ys.flatten(), z.flatten()], axis=1).astype(np.float32)

    # Build faces (two triangles per grid cell)
    faces = []
    for row in range(h - 1):
        for col in range(w - 1):
            i = row * w + col
            faces.append([i, i + w, i + 1])
            faces.append([i + 1, i + w, i + w + 1])
    faces = np.array(faces)

    # UV coordinates for texture mapping
    uvs = np.stack([xs.flatten() / (w - 1), 1 - ys.flatten() / (h - 1)], axis=1)

    # Build texture
    texture_image = Image.fromarray(img_ds)
    material = trimesh.visual.texture.TextureVisuals(uv=uvs, image=texture_image)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=material, process=False)
    mesh.export(output_path)
    print(f"Saved mesh to {output_path}")
    print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="terrain.glb")
    parser.add_argument("--z_scale", type=float, default=50.0)
    parser.add_argument("--downsample", type=int, default=4)
    args = parser.parse_args()

    build_mesh(args.depth, args.image, args.output, args.z_scale, args.downsample)


if __name__ == "__main__":
    main()
