# VAST Lite Workflow

This repository uses a sidecar workflow with VAST Lite 1.5.0 on Windows.

## Operational loop

1. Open your EM stack in VAST Lite.
2. Navigate to the slice that contains the target cell.
3. Export the current slice as an image file.
4. Record the pixel coordinates of a click near the center of the target cell.
5. Run the `predict` command to generate a mask.
6. Inspect the overlay image.
7. Review the result in VAST and manually correct if needed.
8. Export the corrected mask.
9. Run `add-feedback`.
10. Periodically run `finetune`.

## About direct VAST integration

This first version does not ship a native VAST plugin.

Reason:

- The reliable integration point available without reverse-engineering VAST is exported images and masks.
- A sidecar workflow is the fastest path to a working tool you can test on real EM data.

Once you validate model quality, the next step is to automate click capture and mask transfer with Windows UI automation or a helper app.
