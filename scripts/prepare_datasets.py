#!/usr/bin/env python3
"""Download source datasets and rebuild the YOLO unified dataset.

The unified dataset is single-class: class 0 = saco_de_lixo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTS_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
SEED = int(os.getenv("SEED", "42"))
TRAIN_RATIO = float(os.getenv("UNIFIED_TRAIN_RATIO", "0.85"))
BACKGROUND_RATIO = float(os.getenv("UNIFIED_BACKGROUND_RATIO", "0.10"))
BACKGROUND_MAX_IMAGES = int(os.getenv("UNIFIED_BACKGROUND_MAX_IMAGES", "271"))


@dataclass(frozen=True)
class RoboflowDataset:
    key: str
    workspace: str
    project: str
    version: int
    role: str
    positive_classes: tuple[str, ...] = ()
    background_classes: tuple[str, ...] = ()
    expected_positive_images: int | None = None

    @property
    def raw_dir(self) -> Path:
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


def normalize_class_name(name: str) -> str:
    return re.sub(r"[\s_-]+", " ", str(name).strip().casefold())


def dataset_class_names(folder: Path) -> dict[int, str]:
    yaml_paths = sorted(folder.rglob("data.yaml")) + sorted(folder.rglob("data.yml"))
    if not yaml_paths:
        raise RuntimeError(f"data.yaml ausente no dataset {folder.relative_to(ROOT)}.")
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML e necessario para ler as classes exportadas pelo Roboflow.") from exc

    payload = yaml.safe_load(yaml_paths[0].read_text(encoding="utf-8")) or {}
    names = payload.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    raise RuntimeError(f"Campo names invalido em {yaml_paths[0].relative_to(ROOT)}.")


def class_ids_for_names(folder: Path, requested: tuple[str, ...]) -> set[int]:
    if not requested:
        return set()
    names = dataset_class_names(folder)
    wanted = {normalize_class_name(name) for name in requested}
    selected = {idx for idx, name in names.items() if normalize_class_name(name) in wanted}
    missing = wanted - {normalize_class_name(names[idx]) for idx in selected}
    if missing:
        available = ", ".join(f"{idx}:{name}" for idx, name in sorted(names.items()))
        raise RuntimeError(
            f"Classes {sorted(missing)} ausentes em {folder.relative_to(ROOT)}. Disponiveis: {available}"
        )
    return selected


def normalized_yolo_objects(label_path: Path) -> list[tuple[int, str]]:
    if not label_path.exists():
        return []
    out: list[tuple[int, str]] = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
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
        out.append((class_id, f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"))
    return out


def normalize_yolo_lines(label_path: Path, allowed_class_ids: set[int] | None = None) -> list[str]:
    objects = normalized_yolo_objects(label_path)
    if allowed_class_ids is None:
        return [line for _, line in objects]
    return [line for class_id, line in objects if class_id in allowed_class_ids]


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
    def value(name: str, default: str) -> str:
        return os.getenv(name, env.get(name, default))

    return [
        RoboflowDataset(
            key="GARBAGE_8UZHA",
            workspace=value("ROBOFLOW_WORKSPACE_GARBAGE_8UZHA", "project-r1emy"),
            project=value("ROBOFLOW_PROJECT_GARBAGE_8UZHA", "garbage-8uzha"),
            version=int(value("ROBOFLOW_VERSION_GARBAGE_8UZHA", "4")),
            role="train",
            positive_classes=("black_bag", "white_bag"),
            expected_positive_images=3601,
        ),
        RoboflowDataset(
            key="GARBAGE_MVZG3",
            workspace=value("ROBOFLOW_WORKSPACE_GARBAGE_MVZG3", "bill-lhxqf"),
            project=value("ROBOFLOW_PROJECT_GARBAGE_MVZG3", "garbage-mvzg3"),
            version=int(value("ROBOFLOW_VERSION_GARBAGE_MVZG3", "1")),
            role="train",
            positive_classes=("bag - v4 2023-05-12 11-25pm",),
            expected_positive_images=865,
        ),
        RoboflowDataset(
            key="SIDEWALK",
            workspace=value("ROBOFLOW_WORKSPACE_SIDEWALK", "sidewalk"),
            project=value("ROBOFLOW_PROJECT_SIDEWALK", "sidewalk-segmentation"),
            version=int(value("ROBOFLOW_VERSION_SIDEWALK", "4")),
            role="train",
        ),
        RoboflowDataset(
            key="TESTE",
            workspace=value("ROBOFLOW_WORKSPACE_TESTE", "jaime-teixeira"),
            project=value("ROBOFLOW_PROJECT_TESTE", "urban-waste-brazil"),
            version=int(value("ROBOFLOW_VERSION_TESTE", "1")),
            role="test",
        ),
    ]


def download_roboflow(datasets: list[RoboflowDataset], overwrite: bool = False) -> None:
    pending: list[RoboflowDataset] = []
    for ds in datasets:
        dest = ds.raw_dir
        marker = dest / ".download_complete.json"
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
        except (OSError, json.JSONDecodeError):
            marker_data = {}
        expected = {"workspace": ds.workspace, "project": ds.project, "version": ds.version}
        if not overwrite and list_images(dest) and all(marker_data.get(k) == v for k, v in expected.items()):
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
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            rf.workspace(ds.workspace).project(ds.project).version(ds.version).download(
                "yolov8", location=str(dest), overwrite=True
            )
        except Exception as exc:
            print(f"[aviso] SDK Roboflow falhou para {ds.project}: {exc}; tentando download direto.")
        if not list_images(dest):
            direct_download_roboflow(ds, api_key, dest)
        images = list_images(dest)
        if not images:
            raise RuntimeError(f"Download de {ds.project} concluido sem imagens.")
        marker_payload = {
            "workspace": ds.workspace, "project": ds.project, "version": ds.version,
            "images": len(images),
        }
        (dest / ".download_complete.json").write_text(
            json.dumps(marker_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


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


def roboflow_training_records(datasets: list[RoboflowDataset]) -> tuple[list[dict], list[dict]]:
    positives: list[dict] = []
    background_candidates: list[dict] = []
    for ds in datasets:
        if ds.role != "train":
            continue
        before = len(positives)
        positive_ids = class_ids_for_names(ds.raw_dir, ds.positive_classes)
        background_ids = class_ids_for_names(ds.raw_dir, ds.background_classes)
        for img in list_images(ds.raw_dir):
            if "/images/" not in img.as_posix():
                continue
            objects = normalized_yolo_objects(yolo_label_for_image(img))
            positive_lines = [line for class_id, line in objects if class_id in positive_ids]
            if ds.background_classes:
                has_background = any(class_id in background_ids for class_id, _ in objects)
            elif not ds.positive_classes:
                # Se não tem classes positivas nem restrição de fundo, aceita qualquer imagem com alguma anotação
                has_background = len(objects) > 0
            else:
                has_background = False

            if positive_lines:
                positives.append(
                    {
                        "source": ds.project,
                        "image": img,
                        "labels": positive_lines,
                        "positive": True,
                        "group_key": group_key_for_record(ds.project, img),
                        "selected_classes": list(ds.positive_classes),
                    }
                )
            elif has_background:
                background_candidates.append(
                    {
                        "source": f"{ds.project}_background",
                        "image": img,
                        "labels": [],
                        "positive": False,
                        "group_key": group_key_for_record(ds.project, img),
                        "selected_classes": list(ds.background_classes),
                    }
                )
        found = len(positives) - before
        if found == 0 and ds.positive_classes:
            raise RuntimeError(
                f"Nenhuma imagem positiva valida encontrada em {ds.raw_dir.relative_to(ROOT)}. "
                "Execute primeiro: python scripts/prepare_datasets.py --download"
            )
        expected = ds.expected_positive_images
        expectation = f"; referencia informada={expected}" if expected is not None else ""
        print(f"[classes] {ds.project}: positivas={found}{expectation}; classes={list(ds.positive_classes)}")
    return positives, background_candidates


def sample_background_records(positives: list[dict], candidates: list[dict]) -> list[dict]:
    target = min(math.ceil(len(positives) * BACKGROUND_RATIO), BACKGROUND_MAX_IMAGES)
    unique: dict[str, dict] = {}
    for rec in candidates:
        unique.setdefault(rec["group_key"], rec)
    pool = list(unique.values())
    random.Random(SEED).shuffle(pool)
    selected = pool[: min(target, len(pool))]
    print(
        f"[background] candidatos_Roadway={len(pool)} selecionados={len(selected)} "
        f"alvo={target} razao_sobre_positivas={BACKGROUND_RATIO:.2%} limite={BACKGROUND_MAX_IMAGES}"
    )
    if len(selected) < target:
        print(f"[aviso] apenas {len(selected)} imagens Roadway elegiveis para um alvo de {target}.")
    return selected


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
    build_root = ROOT / "data" / ".teste_building"
    clean_dir(build_root)
    (build_root / "images").mkdir(parents=True, exist_ok=True)
    (build_root / "labels").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for img in raw_images:
        img_dst = build_root / "images" / img.name
        lbl_dst = build_root / "labels" / f"{img.stem}.txt"
        lines = normalize_yolo_lines(yolo_label_for_image(img))
        shutil.copy2(img, img_dst)
        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        rows.append({"image_id": img_dst.name, "tem_lixo": int(bool(lines)), "n_bboxes": len(lines)})

    rows = sorted(rows, key=lambda r: r["image_id"])
    gt_path = build_root / "ground_truth.csv"
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
    (build_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if test_root.exists():
        shutil.rmtree(test_root)
    build_root.rename(test_root)
    print(f"[ok] data/teste reconstruido: {summary}")


def rebuild_unified() -> None:
    env = read_env()
    datasets = roboflow_configs(env)
    positives, background_candidates = roboflow_training_records(datasets)
    negatives = sample_background_records(positives, background_candidates)
    records = positives + negatives
    if not records:
        raise RuntimeError("Nenhum registro encontrado para reconstruir data/unified.")

    final_unified = ROOT / "data" / "unified"
    unified = ROOT / "data" / ".unified_building"
    clean_dir(unified)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (unified / sub).mkdir(parents=True, exist_ok=True)

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
                    "image": (Path("data/unified") / img_dst.relative_to(unified)).as_posix(),
                    "label": (Path("data/unified") / lbl_dst.relative_to(unified)).as_posix(),
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
        "  0: saco_de_lixo\n",
        encoding="utf-8",
    )

    manifest = unified / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_by_split_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in manifest_rows:
        key = f"{row['split']}:{row['source']}"
        summary_by_split_source[key]["images"] += 1
        summary_by_split_source[key]["positive_images"] += int(row["positive"])
        summary_by_split_source[key]["negative_images"] += int(not row["positive"])
        summary_by_split_source[key]["boxes"] += int(row["boxes"])
    summary = {
        "target_class": "saco_de_lixo",
        "positive_classes": {
            ds.project: list(ds.positive_classes) for ds in datasets if ds.role == "train"
        },
        "background": {
            "source_project": "garbage-mvzg3",
            "source_class": "Roadway",
            "ratio_over_positive_images": BACKGROUND_RATIO,
            "max_images": BACKGROUND_MAX_IMAGES,
            "available_candidates": len(background_candidates),
            "selected_images": len(negatives),
        },
        "totals": {
            "images": len(records),
            "positive_images": len(positives),
            "negative_images": len(negatives),
            "boxes": sum(len(rec["labels"]) for rec in positives),
        },
        "by_split_source": {key: dict(values) for key, values in sorted(summary_by_split_source.items())},
    }
    (unified / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    train_groups = {row["group_key"] for row in manifest_rows if row["split"] == "train"}
    val_groups = {row["group_key"] for row in manifest_rows if row["split"] == "val"}
    split_audit = {
        "strategy": "grouped_by_source_original_stem",
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "background_ratio_over_positive_images": BACKGROUND_RATIO,
        "background_max_images": BACKGROUND_MAX_IMAGES,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "overlap_groups": len(train_groups & val_groups),
        "overlap_group_keys": sorted(train_groups & val_groups),
    }
    (unified / "split_audit.json").write_text(json.dumps(split_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    if final_unified.exists():
        shutil.rmtree(final_unified)
    unified.rename(final_unified)
    print("[ok] data/unified reconstruido")
    print((final_unified / "data.yaml").read_text(encoding="utf-8"))
    for key, values in sorted(summary_by_split_source.items()):
        print(key, dict(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Baixa os datasets configurados no Roboflow.")
    parser.add_argument("--download-train", action="store_true", help="Baixa somente os datasets de treino.")
    parser.add_argument("--download-test", action="store_true", help="Baixa somente o dataset de teste.")
    parser.add_argument("--rebuild", action="store_true", help="Reconstrói data/unified.")
    parser.add_argument("--rebuild-test", action="store_true", help="Reconstrói data/teste a partir do Roboflow bruto.")
    parser.add_argument("--all", action="store_true", help="Executa download e rebuild.")
    parser.add_argument("--overwrite", action="store_true", help="Força novo download Roboflow.")
    args = parser.parse_args()

    env = read_env()
    if args.download or args.download_train or args.download_test or args.all:
        datasets = roboflow_configs(env)
        if not (args.download or args.all):
            roles = set()
            if args.download_train:
                roles.add("train")
            if args.download_test:
                roles.add("test")
            datasets = [ds for ds in datasets if ds.role in roles]
        download_roboflow(datasets, overwrite=args.overwrite)
    if args.rebuild_test or args.all:
        rebuild_test_set()
    if args.rebuild or args.all:
        rebuild_unified()


if __name__ == "__main__":
    main()
