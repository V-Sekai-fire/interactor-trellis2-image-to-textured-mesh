# interactor-trellis2-image-to-textured-mesh

A container that turns one image into a textured 3D mesh over HTTP and returns it with a USD layer that references it.

## What it is for

It packages the TRELLIS.2 image-to-3D model once, as the base image the mesh-painting and mesh-editing models build on. RFD 1038 owns the model and its interface, and RFD 1036 owns the packaging convention.

## Build and run

    docker build .

The final stage is the GPU worker. The `contract` stage serves the same interface with stub replies, needing no GPU and no weights.

## Licence

MIT. See [LICENSE](LICENSE). The model it packages is MIT-licensed upstream.
