# interactor-trellis2-image-to-textured-mesh

Model image for `trellis2_image_to_textured_mesh` (image to 3D, PBR), per
[weftspun's RFD 0036](https://github.com/weftspun/request-for-discussion/tree/main/0036-packaging-convention)
packaging convention. Facts below come from
[RFD 0038](https://github.com/weftspun/request-for-discussion/tree/main/0038-trellis2-image-to-textured-mesh),
already written and reviewed by this org — this repo converts that RFD's `cog.yaml` +
`predict.py` into the Dockerfile/server.py shape RFD 0036 decided on ("convert one when the
model is next worked on, not in a sweep").

## Why this one first

Per RFD 0038: **this is the base image for four other catalog entries** —
`trellis2_image_mesh_painting`, `voxhammer_text_mesh_editing`, `voxhammer_image_mesh_editing`,
and (abandoned) `weftspun_image_to_world`. A packaging mistake here costs five models, not one.
It's also the direct dependency of two of this session's other new repos
(`interactor-trellis2-image-mesh-painting`, `interactor-voxhammer-{text,image}-mesh-editing`),
so it ships first.

## Model

| Property   | Value                                                                                                                                         |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream   | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) ([microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) on HF) |
| License    | MIT                                                                                                                                           |
| Parameters | 4.0 B, estimated (RFD 0026)                                                                                                                   |
| bf16       | 8.0 GB                                                                                                                                        |
| Q4_K_M     | 2.20 GB (not shipped — RFD 0038 pins bf16 as the ship format)                                                                                 |

License independently checked, not just carried from the RFD: TRELLIS.2's own repo and model
card state MIT, matching RFD 0038's table.

## Interface

`POST /predict`:

| Input                | Type            | Default  | Note                                                                                                       |
| -------------------- | --------------- | -------- | ---------------------------------------------------------------------------------------------------------- |
| `image`              | Path/URL/base64 | required |                                                                                                            |
| `texture_resolution` | int             | 1024     |                                                                                                            |
| `decimation_target`  | int             | 210000   | Hard cap — `API_MAX_MESH_VERTICES` in `src/library/aiModelsCatalog.js`; a larger mesh fails the next stage |
| `seed`               | int             | -1       |                                                                                                            |

Returns `{glb, layer, seed, stub}` — `layer` is the base USD layer per RFD 0053, `glb` the
transmission file recorded as an asset path (plain `usd-core` has no glTF file-format plugin,
so the GLB is referenced, not opened, by the layer — RFD 0040's finding, carried forward here).

## Two stages in one container

Per RFD 0038: the sparse-structure flow runs first, the SLat (texturing) flow runs second, and
both stay in one image because they share the DINOv2 image encoder — splitting them would load
that encoder twice.

## Two-stage image (RFD 0036)

```sh
docker build --target contract -t interactor-trellis2-image-to-textured-mesh:contract .
docker run --rm -p 8000:8000 interactor-trellis2-image-to-textured-mesh:contract
curl -X POST localhost:8000/predict -d @test_input.json -H 'Content-Type: application/json'
```

`worker` stage: CUDA 12.4.1, clones `microsoft/TRELLIS.2` at a pinned commit, downloads
`microsoft/TRELLIS.2-4B` from Hugging Face at build time (RFD 0038's own `cog.yaml` pointed
weight URLs at `weights.invalid` — RFC 2606's reserved placeholder domain, never a real pin;
the real source is the HF repo above, found independently).

## Status

**Scaffolded from the RFD, not yet built or run.** Contract-stage shape (routes, request/reply
schema, USD authoring) is real, taken from RFD 0038's interface table. `_run_upstream()`'s
exact call into TRELLIS.2's own entry point is **not yet verified against the real repo** —
unlike `interactor-pixal3d-image-to-textured-mesh`, where RFD 0040 already documented the
precise `inference.py` CLI. Confirm TRELLIS.2's actual entry point and flags before trusting the
worker stage.
