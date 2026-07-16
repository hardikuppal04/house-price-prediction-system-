"""Unified command-line interface for the production workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from house_price.config import load_config
from house_price.data import download_raw, make_temporal_split, verify_ames_schema
from house_price.drift import drift_report
from house_price.production import load_production_model, predict_one, train_production_model
from house_price.production_evaluation import evaluate_final_holdout
from house_price.provenance import sha256_file, write_json
from house_price.schema import PredictionRequest, curated_frame, schema_document


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _manifest(cfg, raw: pd.DataFrame) -> dict:
    raw_path = cfg.paths.data_raw / cfg.dataset.raw_filename
    development, holdout = make_temporal_split(raw)
    return {
        "openml_id": cfg.dataset.openml_id,
        "openml_version": cfg.dataset.openml_version,
        "raw_sha256": sha256_file(raw_path),
        "rows": len(raw),
        "columns": list(raw.columns),
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "holdout_year": 2010,
        "feature_schema": schema_document(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="house-price")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("validate-data")
    sub.add_parser("train")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--confirm-final", action="store_true")
    evaluate.add_argument("--force", action="store_true")
    predict = sub.add_parser("predict")
    predict.add_argument("--input", type=Path, required=True, help="JSON request file")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    drift = sub.add_parser("drift-report")
    drift.add_argument("--reference", type=Path, required=True)
    drift.add_argument("--current", type=Path, required=True)
    drift.add_argument("--output", type=Path, default=Path("reports/drift_report.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config()
    cfg.paths.ensure()
    if args.command == "download":
        raw = download_raw(cfg)
        print(f"Downloaded {len(raw)} rows")
    elif args.command == "validate-data":
        raw = download_raw(cfg)
        verify_ames_schema(raw)
        curated_frame(raw, target=cfg.dataset.target)
        manifest = _manifest(cfg, raw)
        write_json(cfg.paths.data_raw / "dataset_manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
    elif args.command == "train":
        raw = download_raw(cfg)
        print(json.dumps(train_production_model(cfg, raw), indent=2))
    elif args.command == "evaluate":
        if not args.confirm_final:
            raise SystemExit("Refusing to inspect 2010 holdout without --confirm-final")
        raw = download_raw(cfg)
        print(json.dumps(evaluate_final_holdout(cfg, raw, force=args.force), indent=2))
    elif args.command == "predict":
        payload = PredictionRequest.model_validate_json(args.input.read_text(encoding="utf-8"))
        model, metadata = load_production_model(cfg.paths.models)
        result = predict_one(model, metadata, payload.model_dump())
        print(json.dumps(result.__dict__, indent=2))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("house_price.service:app", host=args.host, port=args.port)
    elif args.command == "drift-report":
        report = drift_report(_read_frame(args.reference), _read_frame(args.current))
        write_json(args.output, report)
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
