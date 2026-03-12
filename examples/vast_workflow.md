# VAST Lite Workflow

This repository currently supports a practical VAST sidecar workflow with VAST Lite 1.5.0 on Windows.

## Operational loop

1. Open your EM stack in VAST Lite.
2. Navigate to the slice that contains the target cell.
3. Export the current slice as an image file.
4. Record the pixel coordinates of a click near the center of the target cell.
5. Run the `predict` command to generate a mask, or `predict-vast-import` to generate a VAST-importable segmentation image.
6. Inspect the overlay image.
7. If you used `predict-vast-import`, import the generated RGB segmentation image into VAST.
8. Review the result in VAST and manually correct if needed.
9. Export the corrected mask.
10. Run `add-feedback`.
11. Periodically run `finetune`.

## About direct VAST integration

This repository does not yet ship a native VAST live plugin.

Reason:

- The reliable integration point available right now is exported images plus VAST-compatible segmentation-image import.
- A sidecar workflow is the fastest path to a working tool you can test on real EM data.

The official VAST Lite 1.5.0 manual documents a Remote Control API Server and direct segmentation-writing functions, so a true live bridge is possible. The missing piece in this repo is the supplementary API client package that defines the client-side command protocol.

Once that package is available locally, the next step is to automate click capture and mask transfer through the API rather than through exported images and manual import.
