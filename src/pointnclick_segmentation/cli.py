from __future__ import annotations

import argparse

from pointnclick_segmentation.config import TrainConfig
from pointnclick_segmentation.prepare_exports import prepare_exports_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PointnClick EM segmentation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from scratch")
    _add_training_args(train_parser)

    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a model with feedback data")
    _add_training_args(finetune_parser)
    finetune_parser.add_argument("--checkpoint", required=True, help="Existing checkpoint to resume from")

    predict_parser = subparsers.add_parser("predict", help="Predict a mask from one click")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--x", required=True, type=int)
    predict_parser.add_argument("--y", required=True, type=int)
    predict_parser.add_argument("--output-mask", required=True)
    predict_parser.add_argument("--output-overlay")
    predict_parser.add_argument("--image-size", type=int)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument("--device", default="cuda")

    vast_predict_parser = subparsers.add_parser(
        "predict-vast-import",
        help="Predict a mask and write a VAST-importable RGB segmentation image",
    )
    vast_predict_parser.add_argument("--checkpoint", required=True)
    vast_predict_parser.add_argument("--image", required=True)
    vast_predict_parser.add_argument("--x", required=True, type=int)
    vast_predict_parser.add_argument("--y", required=True, type=int)
    vast_predict_parser.add_argument("--segment-id", required=True, type=int)
    vast_predict_parser.add_argument("--z-index", required=True, type=int)
    vast_predict_parser.add_argument("--output-dir", required=True)
    vast_predict_parser.add_argument("--output-stem")
    vast_predict_parser.add_argument("--image-size", type=int)
    vast_predict_parser.add_argument("--threshold", type=float, default=0.5)
    vast_predict_parser.add_argument("--device", default="cuda")

    feedback_parser = subparsers.add_parser("add-feedback", help="Add a corrected sample for future fine-tuning")
    feedback_parser.add_argument("--image", required=True)
    feedback_parser.add_argument("--mask", required=True)
    feedback_parser.add_argument("--feedback-dir", required=True)
    feedback_parser.add_argument("--sample-id", required=True)

    prepare_parser = subparsers.add_parser("prepare-exports", help="Build train/val data from exports\\EM and exports\\Boutons")
    prepare_parser.add_argument("--exports-dir", default="exports")
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--val-boutons", help="Comma-separated bouton ids for validation, for example 16,17,18,19")
    prepare_parser.add_argument("--no-resize-masks", action="store_true", help="Fail if a bouton mask size does not match the EM export")

    return parser


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feedback-dir")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"train", "finetune"}:
        from pointnclick_segmentation.train import train_model

        config = TrainConfig(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            output_dir=args.output_dir,
            feedback_dir=args.feedback_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            resume_checkpoint=getattr(args, "checkpoint", None),
        )
        result = train_model(config)
        print(f"Training complete. Best val IoU: {result['best_val_iou']:.4f}")
        print(f"Artifacts saved in: {result['output_dir']}")
        return

    if args.command == "predict":
        from pointnclick_segmentation.infer import predict_mask

        predict_mask(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            x=args.x,
            y=args.y,
            output_mask=args.output_mask,
            output_overlay=args.output_overlay,
            image_size=args.image_size,
            threshold=args.threshold,
            device_name=args.device,
        )
        print(f"Saved predicted mask to: {args.output_mask}")
        if args.output_overlay:
            print(f"Saved overlay to: {args.output_overlay}")
        return

    if args.command == "predict-vast-import":
        from pointnclick_segmentation.vast import predict_vast_import_image

        result = predict_vast_import_image(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            x=args.x,
            y=args.y,
            segment_id=args.segment_id,
            z_index=args.z_index,
            output_dir=args.output_dir,
            output_stem=args.output_stem,
            image_size=args.image_size,
            threshold=args.threshold,
            device_name=args.device,
        )
        print(f"Saved VAST import image to: {result['vast_import_path']}")
        print(f"Saved preview to: {result['vast_preview_path']}")
        print(f"Saved metadata to: {result['metadata_path']}")
        return

    if args.command == "add-feedback":
        from pointnclick_segmentation.feedback import add_feedback_sample

        result = add_feedback_sample(
            image_path=args.image,
            mask_path=args.mask,
            feedback_dir=args.feedback_dir,
            sample_id=args.sample_id,
        )
        print(f"Copied image to: {result['image']}")
        print(f"Copied mask to: {result['mask']}")
        return

    if args.command == "prepare-exports":
        val_boutons = None
        if args.val_boutons:
            val_boutons = [int(part.strip()) for part in args.val_boutons.split(",") if part.strip()]
        result = prepare_exports_dataset(
            exports_dir=args.exports_dir,
            output_dir=args.output_dir,
            val_boutons=val_boutons,
            resize_masks_to_em=not args.no_resize_masks,
        )
        print(f"Prepared dataset in: {args.output_dir}")
        print(f"Training samples: {result['num_train']}")
        print(f"Validation samples: {result['num_val']}")
        print(f"Validation boutons: {result['val_boutons']}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
