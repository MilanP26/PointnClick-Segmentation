from __future__ import annotations

import argparse

from pointnclick_segmentation.config import TrainConfig
from pointnclick_segmentation.prepare_exports import prepare_exports_dataset
from pointnclick_segmentation.prepare_worm import prepare_worm_dataset


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

    vast_state_parser = subparsers.add_parser("vast-state", help="Read current state from a running VAST Remote Control API server")
    vast_state_parser.add_argument("--host", default="127.0.0.1")
    vast_state_parser.add_argument("--port", type=int, default=22081)

    vast_live_parser = subparsers.add_parser(
        "vast-live",
        help="Watch VAST clicks and write predicted masks directly into the selected segmentation layer",
    )
    vast_live_parser.add_argument("--checkpoint", required=True)
    vast_live_parser.add_argument("--host", default="127.0.0.1")
    vast_live_parser.add_argument("--port", type=int, default=22081)
    vast_live_parser.add_argument("--poll-interval", type=float, default=0.1)
    vast_live_parser.add_argument("--crop-size", type=int, default=512)
    vast_live_parser.add_argument("--threshold", type=float, default=0.5)
    vast_live_parser.add_argument("--image-size", type=int)
    vast_live_parser.add_argument("--device", default="cpu")
    vast_live_parser.add_argument("--output-dir", default="outputs\\vast_live")
    vast_live_parser.add_argument("--allowed-uimode", type=int)
    vast_live_parser.add_argument("--auto-key", default="P", help="Hold this key while clicking to trigger auto-segmentation")
    vast_live_parser.add_argument("--feedback-key", default="I", help="Hold this key while clicking corrected masks to capture feedback")
    vast_live_parser.add_argument("--feedback-dir", default="data\\feedback_vast")
    vast_live_parser.add_argument("--disable-online-learning", action="store_true")
    vast_live_parser.add_argument("--online-output-dir", default="runs\\live_feedback")
    vast_live_parser.add_argument("--online-epochs", type=int, default=1)
    vast_live_parser.add_argument("--online-learning-rate", type=float, default=1e-4)

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

    prepare_worm_parser = subparsers.add_parser(
        "prepare-worm",
        help="Build train/val/test data from a dense worm dataset stored as em/mask folders or em.zip/mask.zip",
    )
    prepare_worm_parser.add_argument("--data-dir", default="data\\Training Round 2")
    prepare_worm_parser.add_argument("--output-dir", required=True)
    prepare_worm_parser.add_argument("--val-fraction", type=float, default=0.2)
    prepare_worm_parser.add_argument("--test-fraction", type=float, default=0.1)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on a held-out dataset")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--data-dir", required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=4)
    evaluate_parser.add_argument("--num-workers", type=int, default=0)
    evaluate_parser.add_argument("--device", default="cuda")

    return parser


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feedback-dir")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, help="Training crop size before optional resize. Defaults to image-size.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--selection-metric", choices=["vi", "iou"], default="vi")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=10)


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
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            resume_checkpoint=getattr(args, "checkpoint", None),
            base_channels=args.base_channels,
            selection_metric=args.selection_metric,
            early_stopping_patience=args.early_stopping_patience,
            min_epochs=args.min_epochs,
        )
        result = train_model(config)
        best_metrics = result["best_metrics"] or {}
        print(
            "Training complete. "
            f"Best epoch: {result['best_epoch']}. "
            f"Best val IoU: {best_metrics.get('iou', 0.0):.4f}. "
            f"Best val VI: {best_metrics.get('vi', 0.0):.4f}"
        )
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

    if args.command == "vast-state":
        from pointnclick_segmentation.vast_client import VastClient

        try:
            with VastClient(host=args.host, port=args.port) as client:
                info = client.get_info()
                client.set_api_layers_enabled(True)
                selected_layer, selected_em_layer, selected_seg_layer = client.get_selected_layer_info()
                selected_segment = None
                try:
                    selected_segment = client.get_selected_segment_nr()
                except Exception as exc:
                    if not (hasattr(exc, "error_code") and getattr(exc, "error_code") == 21):
                        raise
                state = client.get_current_ui_state()
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to the VAST Remote Control API server. "
                "In VAST Lite, enable Window > Remote Control API Server and confirm the host/port. "
                f"Underlying error: {exc!r}"
            ) from exc
        print(f"Dataset size (x,y,z): {info['uints'][0]}, {info['uints'][1]}, {info['uints'][2]}")
        print(f"Selected layer: {selected_layer}")
        print(f"Selected EM layer: {selected_em_layer}")
        print(f"Selected segmentation layer: {selected_seg_layer}")
        print(f"Selected segment: {selected_segment if selected_segment is not None else 'Unavailable in current VAST UI state'}")
        print(f"UI mode: {state['uimode']}")
        print(f"Mouse voxel: ({state['mousecoordsx']}, {state['mousecoordsy']}, {state['mousecoordsz']})")
        print(f"Last left click window coords: ({state['lastleftclickx']}, {state['lastleftclicky']})")
        return

    if args.command == "vast-live":
        from pointnclick_segmentation.vast_live import VastLiveConfig, run_vast_live_bridge

        config = VastLiveConfig(
            checkpoint_path=args.checkpoint,
            host=args.host,
            port=args.port,
            poll_interval_s=args.poll_interval,
            crop_size=args.crop_size,
            threshold=args.threshold,
            image_size=args.image_size,
            device_name=args.device,
            output_dir=args.output_dir,
            allowed_uimode=args.allowed_uimode,
            auto_segment_key=args.auto_key,
            feedback_capture_key=args.feedback_key,
            feedback_dir=args.feedback_dir,
            online_learning=not args.disable_online_learning,
            online_learning_output_dir=args.online_output_dir,
            online_learning_epochs=args.online_epochs,
            online_learning_rate=args.online_learning_rate,
        )
        try:
            run_vast_live_bridge(config)
        except Exception as exc:
            raise RuntimeError(
                "VAST live bridge failed. Make sure VAST Lite is running, "
                "Window > Remote Control API Server is enabled, and the selected segment/layers are valid."
            ) from exc
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

    if args.command == "prepare-worm":
        result = prepare_worm_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )
        print(f"Prepared worm dataset in: {args.output_dir}")
        print(f"Training slices: {result['num_train']}")
        print(f"Validation slices: {result['num_val']}")
        print(f"Test slices: {result['num_test']}")
        return

    if args.command == "evaluate":
        from pointnclick_segmentation.train import evaluate_model

        result = evaluate_model(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
        )
        print(f"Loss: {result['loss']:.4f}")
        print(f"IoU: {result['iou']:.4f}")
        print(f"Dice: {result['dice']:.4f}")
        print(f"VI: {result['vi']:.4f}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
