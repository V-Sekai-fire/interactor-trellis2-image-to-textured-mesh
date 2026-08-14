"""TRELLIS.2 image to textured mesh, PBR. RFD 0038.

An HTTP server in a plain Docker image (RFD 0036 -- not a Cog). This is the base
image for four other catalog entries (RFD 0038); a packaging mistake here costs
five models, not one.

Upstream is microsoft/TRELLIS.2, MIT, weights at microsoft/TRELLIS.2-4B on
Hugging Face. The sparse-structure flow runs first, the SLat (texturing) flow
second, both stay in one process because they share the DINOv2 image encoder.

The `_run_upstream` call below is the real, documented Python API from
TRELLIS.2's own README (`Trellis2ImageTo3DPipeline.from_pretrained(...).run(...)`
+ `o_voxel.postprocess.to_glb(...)`) -- checked against the actual upstream repo,
not invented, since TRELLIS.2 ships no `inference.py` CLI (only a Gradio app).
"""

import base64
import os
import tempfile
import urllib.request
from pathlib import Path

SRC = os.environ.get("TRELLIS2_SRC", "/src/TRELLIS.2")
WEIGHTS = os.environ.get("TRELLIS2_WEIGHTS", "microsoft/TRELLIS.2-4B")

# API_MAX_MESH_VERTICES, src/library/aiModelsCatalog.js. A larger mesh fails
# the next stage.
MAX_MESH_VERTICES = 210000

STUB = os.environ.get("WEFTSPUN_STUB") == "1"

_READY = {"loaded": False}
_PIPELINE = {"pipeline": None}


class InputError(ValueError):
    """The request is wrong. This is the caller's fault, and not ours."""


def _fetch(image: str, work: Path) -> Path:
    target = work / "input.png"
    if image.startswith(("http://", "https://")):
        urllib.request.urlretrieve(image, target)
        return target
    if image.startswith("data:"):
        image = image.split(",", 1)[1]
    target.write_bytes(base64.b64decode(image))
    return target


def _validate(job_input: dict) -> dict:
    if not job_input.get("image"):
        raise InputError("image is required: a URL, a data URI, or base64")

    decimation = int(job_input.get("decimation_target", MAX_MESH_VERTICES))
    if not 1000 <= decimation <= MAX_MESH_VERTICES:
        raise InputError(f"decimation_target must be between 1000 and {MAX_MESH_VERTICES}")

    texture_resolution = int(job_input.get("texture_resolution", 1024))
    if texture_resolution not in (512, 1024, 2048, 4096):
        raise InputError("texture_resolution must be 512, 1024, 2048, or 4096")

    return {
        "image": job_input["image"],
        "texture_resolution": texture_resolution,
        "decimation_target": decimation,
        "seed": int(job_input.get("seed", -1)),
    }


def _run_upstream(image_path: Path, glb: Path, args: dict) -> None:
    """The real TRELLIS.2 Python API, per its own README:

        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        pipeline.cuda()
        mesh = pipeline.run(image)[0]
        mesh.simplify(16777216)  # nvdiffrast limit
        glb = o_voxel.postprocess.to_glb(vertices=..., faces=..., ...)
        glb.export(str(path), extension_webp=True)

    Loaded once (`_PIPELINE`), not per request -- the model loads once at start,
    same reasoning as every other image in this catalog.
    """
    import o_voxel
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if _PIPELINE["pipeline"] is None:
        _PIPELINE["pipeline"] = Trellis2ImageTo3DPipeline.from_pretrained(WEIGHTS)
        _PIPELINE["pipeline"].cuda()
    pipeline = _PIPELINE["pipeline"]

    image = Image.open(image_path)
    if args["seed"] >= 0:
        import torch

        torch.manual_seed(args["seed"])
    mesh = pipeline.run(image)[0]
    mesh.simplify(16777216)  # nvdiffrast's own limit, per upstream's README

    result = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args["decimation_target"],
        texture_size=args["texture_resolution"],
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    result.export(str(glb), extension_webp=True)


def _to_usd(glb: Path, work: Path) -> Path:
    """Base USD layer, RFD 0053. Records the GLB as an asset path, not a
    `references` arc -- plain usd-core has no glTF file-format plugin
    (RFD 0040's finding, carried forward here)."""
    from pxr import Sdf, Usd, UsdGeom

    layer = work / "layer.usda"
    stage = Usd.Stage.CreateNew(str(layer))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    geometry = stage.DefinePrim("/Asset/Geometry")
    geometry.CreateAttribute("weftspun:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(glb.name)
    )
    geometry.CreateAttribute("weftspun:sourceFormat", Sdf.ValueTypeNames.Token).Set("gltf")
    geometry.CreateAttribute("weftspun:stage", Sdf.ValueTypeNames.Token).Set(
        "trellis2_image_to_textured_mesh"
    )
    stage.GetRootLayer().Save()
    return layer


def _encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def predict(job_input: dict) -> dict:
    args = _validate(job_input)
    work = Path(tempfile.mkdtemp())
    image_path = _fetch(args["image"], work)
    glb = work / "output.glb"

    if STUB:
        glb.write_bytes(bytes([0x67, 0x6C, 0x54, 0x46, 0x02]) + b"stub")
    else:
        _run_upstream(image_path, glb, args)

    layer = _to_usd(glb, work)

    return {
        "glb": _encode(glb),
        "layer": _encode(layer),
        "seed": args["seed"],
        "stub": STUB,
    }


def load() -> None:
    if STUB:
        _READY["loaded"] = True
        return
    if not Path(SRC).is_dir():
        raise RuntimeError("the upstream source is absent: " + SRC)
    _READY["loaded"] = True


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="trellis2_image_to_textured_mesh", version="0.1.0")

    class PredictRequest(BaseModel):
        image: str
        texture_resolution: int = 1024
        decimation_target: int = MAX_MESH_VERTICES
        seed: int = -1

    @app.get("/health")
    def health():
        return {"status": "ok", "ready": _READY["loaded"], "stub": STUB}

    @app.post("/predict")
    def run(request: PredictRequest):
        try:
            return predict(request.model_dump())
        except InputError as error:
            return JSONResponse(status_code=400, content={"error": str(error)})

    return app


if __name__ == "__main__":
    import uvicorn

    load()
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
