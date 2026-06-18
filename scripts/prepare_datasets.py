#!/usr/bin/env python3
"""Download source datasets and rebuild the YOLO unified dataset.

The unified dataset is single-class: class 0 = lixo.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTS_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
SEED = int(os.getenv("SEED", "42"))
TRAIN_RATIO = float(os.getenv("UNIFIED_TRAIN_RATIO", "0.85"))


@dataclass(frozen=True)
class RoboflowDataset:
    key: str
    workspace: str
    project: str
    version: int
    role: str

    @property
    def raw_dir(self) -> Path:
        if self.role == "negative":
            return ROOT / "data" / "raw_sidewalk"
        if self.role == "test":
            return ROOT / "data" / "raw_imagens_teste"
        return ROOT / "data" / f"raw_{self.project}"


def read_env(path: Path = ROOT / ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    os.environ.update({k: v for k, v in env.items() if k not in os.environ})
    return env


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS_IMG)


def clean_dir(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def yolo_label_for_image(img: Path) -> Path:
    parts = list(img.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")


def normalize_yolo_lines(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    out: list[str] = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue

        if len(coords) == 4:
            x, y, w, h = coords
        else:
            xs = coords[0::2]
            ys = coords[1::2]
            if not xs or not ys:
                continue
            xmin, xmax = max(0.0, min(xs)), min(1.0, max(xs))
            ymin, ymax = max(0.0, min(ys)), min(1.0, max(ys))
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        w = min(1.0, max(0.0, w))
        h = min(1.0, max(0.0, h))
        if w <= 0 or h <= 0:
            continue
        out.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return out


def roboflow_original_stem(path: Path) -> str:
    """Return a stable source-image key, ignoring Roboflow export hashes.

    Roboflow exports often create files like:
    image_name_jpg.rf.012345abcdef.jpg

    The suffix after ".rf." is an export/augmentation hash. Using only the
    portion before ".rf." keeps augmented variants of the same source image in
    the same train/val split.
    """
    stem = path.stem
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    stem = re.sub(r"_(jpg|jpeg|png|bmp|webp|heic)$", "", stem, flags=re.IGNORECASE)
    return stem


def group_key_for_record(source: str, image: Path) -> str:
    return f"{source}:{roboflow_original_stem(image)}"


def roboflow_configs(env: dict[str, str]) -> list[RoboflowDataset]:
    keys = [x.strip() for x in env.get("ROBOFLOW_POSITIVE_DATASETS", "").split(",") if x.strip()]
    datasets: list[RoboflowDataset] = []
    for key in keys:
        ws = env.get(f"ROBOFLOW_WORKSPACE_{key}")
        project = env.get(f"ROBOFLOW_PROJECT_{key}")
        version = env.get(f"ROBOFLOW_VERSION_{key}")
        if not ws or not project or not version:
            raise RuntimeError(f"Configuracao Roboflow incompleta para {key}.")
        datasets.append(RoboflowDataset(key, ws, project, int(version), "positive"))

    for key, role in [("SIDEWALK", "negative"), ("TESTE", "test")]:
        ws = env.get(f"ROBOFLOW_WORKSPACE_{key}")
        project = env.get(f"ROBOFLOW_PROJECT_{key}")
        version = env.get(f"ROBOFLOW_VERSION_{key}")
        if ws and project and version:
            datasets.append(RoboflowDataset(key, ws, project, int(version), role))
    return datasets


def download_roboflow(datasets: list[RoboflowDataset], overwrite: bool = False) -> None:
    pending: list[RoboflowDataset] = []
    for ds in datasets:
        dest = ds.raw_dir
        if not overwrite and list_images(dest):
            print(f"[ok] {ds.project}: ja existem {len(list_images(dest))} imagens em {dest.relative_to(ROOT)}")
        else:
            pending.append(ds)

    if not pending:
        return

    try:
        from roboflow import Roboflow
    except Exception as exc:
        raise RuntimeError("Pacote roboflow indisponivel. Instale requirements.txt no kernel/ambiente.") from exc

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY ausente no .env.")

    rf = Roboflow(api_key=api_key)
    for ds in pending:
        dest = ds.raw_dir

        print(f"[download] {ds.workspace}/{ds.project}:{ds.version} -> {dest.relative_to(ROOT)}")
        dest.mkdir(parents=True, exist_ok=True)
        rf.workspace(ds.workspace).project(ds.project).version(ds.version).download(
            "yolov8", location=str(dest), overwrite=overwrite
        )
        if not list_images(dest):
            direct_download_roboflow(ds, api_key, dest)


def direct_download_roboflow(ds: RoboflowDataset, api_key: str, dest: Path) -> None:
    import requests

    print(f"[download direto] {ds.workspace}/{ds.project}:{ds.version} -> {dest.relative_to(ROOT)}")
    export_url = f"https://api.roboflow.com/{ds.workspace}/{ds.project}/{ds.version}/yolov8"
    response = requests.get(export_url, params={"api_key": api_key}, timeout=90)
    response.raise_for_status()
    data = response.json()
    link = data.get("export", {}).get("link")
    if not link:
        raise RuntimeError(f"Roboflow nao retornou link de exportacao para {ds.project}.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / f"{ds.project}.zip"
        with requests.get(link, stream=True, timeout=180) as download:
            download.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in download.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    print(f"[ok] {ds.project}: {len(list_images(dest))} imagens baixadas")


def ensure_taco(download_images: bool) -> None:
    env = read_env()
    repo = ROOT / env.get("TACO_DEST", "data/raw_taco") / "TACO"
    repo_url = env.get("TACO_REPO_URL", "https://github.com/pedropro/TACO.git")
    if not repo.exists():
        print(f"[clone] TACO -> {repo.relative_to(ROOT)}")
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repo_url, str(repo)], check=True)
    if download_images:
        print("[download] TACO images")
        subprocess.run([sys.executable, "download.py"], cwd=str(repo), check=False)
    print(f"[ok] TACO: {len(list_images(repo / 'data'))} imagens locais")


def taco_records() -> list[dict]:
    env = read_env()
    taco_root = ROOT / env.get("TACO_DEST", "data/raw_taco") / "TACO" / "data"
    ann_path = taco_root / "annotations.json"
    if not ann_path.exists():
        raise RuntimeError(f"Anotacoes TACO ausentes: {ann_path}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_img[int(ann["image_id"])].append(ann)

    records: list[dict] = []
    for img in data.get("images", []):
        img_path = taco_root / img["file_name"]
        if not img_path.exists():
            continue
        width = float(img["width"])
        height = float(img["height"])
        lines: list[str] = []
        for ann in anns_by_img.get(int(img["id"]), []):
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 0 or h <= 0:
                continue
            xc = (x + w / 2) / width
            yc = (y + h / 2) / height
            lines.append(f"0 {xc:.6f} {yc:.6f} {w / width:.6f} {h / height:.6f}")
        if lines:
            records.append(
                {
                    "source": "taco",
                    "image": img_path,
                    "labels": lines,
                    "positive": True,
                    "group_key": group_key_for_record("taco", img_path),
                }
            )

    target = int(env.get("TACO_N_ALVO", "1500"))
    if len(records) > target:
        rng = random.Random(SEED)
        records = rng.sample(records, target)
    return records


def roboflow_positive_records(datasets: list[RoboflowDataset]) -> list[dict]:
    records: list[dict] = []
    for ds in datasets:
        if ds.role != "positive":
            continue
        before = len(records)
        for img in list_images(ds.raw_dir):
            if "/images/" not in img.as_posix():
                continue
            lines = normalize_yolo_lines(yolo_label_for_image(img))
            if lines:
                records.append(
                    {
                        "source": ds.project,
                        "image": img,
                        "labels": lines,
                        "positive": True,
                        "group_key": group_key_for_record(ds.project, img),
                    }
                )
        if len(records) == before:
            raise RuntimeError(
                f"Nenhuma imagem positiva valida encontrada em {ds.raw_dir.relative_to(ROOT)}. "
                "Execute primeiro: python scripts/prepare_datasets.py --download"
            )
    return records


def sidewalk_negative_records(datasets: list[RoboflowDataset]) -> list[dict]:
    ds = next((d for d in datasets if d.role == "negative"), None)
    if ds is None:
        return []
    folder = ds.raw_dir
    return [
        {
            "source": ds.project,
            "image": img,
            "labels": [],
            "positive": False,
            "group_key": group_key_for_record(ds.project, img),
        }
        for img in list_images(folder)
        if "/images/" in img.as_posix()
    ]


def split_records_grouped(records: list[dict]) -> dict[str, list[dict]]:
    by_source_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        group_key = rec.get("group_key") or group_key_for_record(rec["source"], rec["image"])
        rec["group_key"] = group_key
        by_source_group[rec["source"]][group_key].append(rec)

    rng = random.Random(SEED)
    split = {"train": [], "val": []}
    for source, groups_map in sorted(by_source_group.items()):
        groups = list(groups_map.values())
        rng.shuffle(groups)
        n_train_groups = max(1, int(round(len(groups) * TRAIN_RATIO))) if len(groups) > 1 else len(groups)
        train_groups = groups[:n_train_groups]
        val_groups = groups[n_train_groups:]
        train_items = [rec for group in train_groups for rec in group]
        val_items = [rec for group in val_groups for rec in group]
        split["train"].extend(train_items)
        split["val"].extend(val_items)
        print(
            f"[split agrupado] {source}: "
            f"grupos_train={len(train_groups)} grupos_val={len(val_groups)} "
            f"imgs_train={len(train_items)} imgs_val={len(val_items)}"
        )
    return split


def rebuild_test_set() -> None:
    env = read_env()
    datasets = roboflow_configs(env)
    ds = next((d for d in datasets if d.role == "test"), None)
    if ds is None:
        raise RuntimeError("Dataset Roboflow de teste nao configurado no .env.")

    raw_images = [img for img in list_images(ds.raw_dir) if "/images/" in img.as_posix()]
    if not raw_images:
        raise RuntimeError(
            f"Nenhuma imagem de teste encontrada em {ds.raw_dir.relative_to(ROOT)}. "
            "Execute primeiro: python scripts/prepare_datasets.py --download"
        )

    test_root = ROOT / "data" / "teste"
    clean_dir(test_root / "images")
    clean_dir(test_root / "labels")

    rows: list[dict] = []
    for img in raw_images:
        img_dst = test_root / "images" / img.name
        lbl_dst = test_root / "labels" / f"{img.stem}.txt"
        lines = normalize_yolo_lines(yolo_label_for_image(img))
        shutil.copy2(img, img_dst)
        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        rows.append({"image_id": img_dst.name, "tem_lixo": int(bool(lines)), "n_bboxes": len(lines)})

    rows = sorted(rows, key=lambda r: r["image_id"])
    gt_path = test_root / "ground_truth.csv"
    with gt_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "tem_lixo", "n_bboxes"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "images": len(rows),
        "positive_images": int(sum(r["tem_lixo"] for r in rows)),
        "negative_images": int(sum(1 - r["tem_lixo"] for r in rows)),
        "boxes": int(sum(r["n_bboxes"] for r in rows)),
        "source": ds.project,
        "version": ds.version,
    }
    (test_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] data/teste reconstruido: {summary}")


def rebuild_unified() -> None:
    env = read_env()
    datasets = roboflow_configs(env)
    records = taco_records()
    records.extend(roboflow_positive_records(datasets))
    records.extend(sidewalk_negative_records(datasets))
    if not records:
        raise RuntimeError("Nenhum registro encontrado para reconstruir data/unified.")

    unified = ROOT / "data" / "unified"
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        clean_dir(unified / sub)
    for cache_file in (unified / "labels").glob("*.cache"):
        cache_file.unlink()

    split = split_records_grouped(records)
    manifest_rows: list[dict] = []
    for split_name, items in split.items():
        counters: dict[str, int] = defaultdict(int)
        for rec in items:
            source = rec["source"]
            counters[source] += 1
            stem = f"{source}_{counters[source]:06d}"
            img_dst = unified / "images" / split_name / f"{stem}{rec['image'].suffix.lower()}"
            lbl_dst = unified / "labels" / split_name / f"{stem}.txt"
            shutil.copy2(rec["image"], img_dst)
            lbl_dst.write_text("\n".join(rec["labels"]) + ("\n" if rec["labels"] else ""), encoding="utf-8")
            manifest_rows.append(
                {
                    "split": split_name,
                    "source": source,
                    "image": img_dst.relative_to(ROOT).as_posix(),
                    "label": lbl_dst.relative_to(ROOT).as_posix(),
                    "original_image": rec["image"].relative_to(ROOT).as_posix(),
                    "group_key": rec.get("group_key") or group_key_for_record(source, rec["image"]),
                    "positive": int(rec["positive"]),
                    "boxes": len(rec["labels"]),
                }
            )

    data_yaml = unified / "data.yaml"
    data_yaml.write_text(
        "path: data/unified\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: lixo\n",
        encoding="utf-8",
    )

    manifest = unified / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in manifest_rows:
        key = f"{row['split']}:{row['source']}"
        summary[key]["images"] += 1
        summary[key]["positive_images"] += int(row["positive"])
        summary[key]["boxes"] += int(row["boxes"])
    (unified / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    train_groups = {row["group_key"] for row in manifest_rows if row["split"] == "train"}
    val_groups = {row["group_key"] for row in manifest_rows if row["split"] == "val"}
    split_audit = {
        "strategy": "grouped_by_source_original_stem",
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "overlap_groups": len(train_groups & val_groups),
        "overlap_group_keys": sorted(train_groups & val_groups),
    }
    (unified / "split_audit.json").write_text(json.dumps(split_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[ok] data/unified reconstruido")
    print(data_yaml.read_text(encoding="utf-8"))
    for key, values in sorted(summary.items()):
        print(key, dict(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Baixa datasets Roboflow e garante TACO.")
    parser.add_argument("--rebuild", action="store_true", help="Reconstrói data/unified.")
    parser.add_argument("--rebuild-test", action="store_true", help="Reconstrói data/teste a partir do Roboflow bruto.")
    parser.add_argument("--all", action="store_true", help="Executa download e rebuild.")
    parser.add_argument("--overwrite", action="store_true", help="Força novo download Roboflow.")
    parser.add_argument("--download-taco-images", action="store_true", help="Executa download.py do TACO.")
    args = parser.parse_args()

    env = read_env()
    if args.download or args.all:
        datasets = roboflow_configs(env)
        download_roboflow(datasets, overwrite=args.overwrite)
        ensure_taco(download_images=args.download_taco_images)
    if args.rebuild_test or args.all:
        rebuild_test_set()
    if args.rebuild or args.all:
        rebuild_unified()


if __name__ == "__main__":
    main()
