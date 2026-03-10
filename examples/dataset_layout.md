# Dataset Layout

Minimal dataset structure:

```text
data/
|-- train/
|   |-- images/
|   `-- masks/
|-- val/
|   |-- images/
|   `-- masks/
`-- feedback/
    |-- images/
    `-- masks/
```

Guidelines:

- Keep image and mask names aligned.
- Use one binary mask per target object.
- If an original EM image has many cells, either crop individual training examples or duplicate the image and pair it with different single-cell masks.
- Start with 2D single-slice training. Add adjacent slices later if needed.
