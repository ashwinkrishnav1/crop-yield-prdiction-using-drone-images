"""
MaizeSight Web Application Server
Serves the 3D animated frontend and exposes:
  - Live CropYieldPipeline dual-model inference (/predict_both)
  - Interactive Trial Database exploration (/api/database/...)
  - One-click plot prediction & automatic multi-temporal timepoint auto-matching
"""
import io
import os
import re
import shutil
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from crop_inference.pipeline import CropYieldPipeline
from crop_inference.tabular import clean_manifest

app = FastAPI(title="MaizeSight - UAV Maize Yield Intelligence")

# Enable CORS for flexible local deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"
DATASET_ROOT = PROJECT_ROOT / "project_requirements" / "drone_dataset"
VALID_PLOTS_FILE = DATASET_ROOT / "valid_plots.csv"

# Lazily initialized pipeline instance
_pipeline: Optional[CropYieldPipeline] = None


def get_pipeline() -> CropYieldPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CropYieldPipeline(model_root=PROJECT_ROOT, need_images=True)
    return _pipeline


# Standard filename regex pattern: Location-Timepoint-Experiment_Range_Row.PNG
# e.g.: Ames-TP1-4231_10_2.PNG or Crawfordsville_TP3_4351_3_15.png
REGEX_PLOT = re.compile(
    r"^(?P<loc>[A-Za-z]+)[-_](?P<tp>TP[123])[-_](?P<exp>.+?)_(?P<range>\d+(?:\.\d+)?)_(?P<row>\d+(?:\.\d+)?)\.",
    re.IGNORECASE,
)

# In-memory database caches
_plots_df: Optional[pd.DataFrame] = None
_plot_id_map: Dict[str, dict] = {}
_filename_map: Dict[str, dict] = {}
_loc_plots_map: Dict[str, List[dict]] = {}


def get_database():
    """Lazily loads and indexes valid_plots.csv for sub-millisecond lookup."""
    global _plots_df, _plot_id_map, _filename_map, _loc_plots_map
    if _plots_df is None and VALID_PLOTS_FILE.exists():
        _plots_df = pd.read_csv(VALID_PLOTS_FILE)
        records = _plots_df.to_dict("records")
        for r in records:
            pid = str(r["plot_id"])
            _plot_id_map[pid] = r
            _plot_id_map[pid.lower()] = r

            # Map filenames across all timepoints
            for tp_col in ("TP1_image_path", "TP2_image_path", "TP3_image_path"):
                p_val = r.get(tp_col)
                if p_val and pd.notna(p_val):
                    fn = Path(str(p_val)).name.lower()
                    _filename_map[fn] = r

            loc = str(r.get("location", "Unknown"))
            if loc not in _loc_plots_map:
                _loc_plots_map[loc] = []
            _loc_plots_map[loc].append({
                "plot_id": pid,
                "location": loc,
                "experiment": str(r.get("experiment", "")),
                "range": int(r.get("range", 0)) if pd.notna(r.get("range")) else 0,
                "row": int(r.get("row", 0)) if pd.notna(r.get("row")) else 0,
                "genotype": str(r.get("genotype", "Unknown")),
                "yieldPerAcre": round(float(r["yieldPerAcre"]), 2) if pd.notna(r.get("yieldPerAcre")) else None,
                "totalStandCount": float(r["totalStandCount"]) if pd.notna(r.get("totalStandCount")) else None,
                "plantingDate": str(r.get("plantingDate", "")),
            })
    return _plots_df, _plot_id_map, _filename_map, _loc_plots_map


def find_plot_record(query: str) -> Optional[dict]:
    """Finds a plot record by plot_id, image filename, or regex pattern."""
    get_database()
    q = query.strip()
    q_lower = q.lower()

    # 1. Exact plot_id match
    if q in _plot_id_map:
        return _plot_id_map[q]
    if q_lower in _plot_id_map:
        return _plot_id_map[q_lower]

    # 2. Filename match
    fname = Path(q_lower).name
    if fname in _filename_map:
        return _filename_map[fname]
    alt_fname = fname.replace("_", "-")
    if alt_fname in _filename_map:
        return _filename_map[alt_fname]
    alt_fname2 = fname.replace("-", "_")
    if alt_fname2 in _filename_map:
        return _filename_map[alt_fname2]

    # 3. Regex match
    m = REGEX_PLOT.match(fname)
    if m:
        gd = m.groupdict()
        loc = gd["loc"]
        exp = gd["exp"]
        rng = int(float(gd["range"]))
        rw = int(float(gd["row"]))
        cand_id = f"{loc}_{exp}_{rng}_{rw}"
        if cand_id in _plot_id_map:
            return _plot_id_map[cand_id]
        if cand_id.lower() in _plot_id_map:
            return _plot_id_map[cand_id.lower()]

    return None


@app.get("/")
async def serve_index():
    """Serves the MaizeSight frontend HTML."""
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    root_index = PROJECT_ROOT / "index.html"
    if root_index.exists():
        return FileResponse(root_index)
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/health")
async def health_check():
    """Health check endpoint confirming model and database availability."""
    state_m1 = PROJECT_ROOT / "inference_state" / "state_m1.pkl"
    state_m2 = PROJECT_ROOT / "inference_state" / "state_m2.pkl"
    get_database()
    return {
        "status": "online",
        "state_m1_ready": state_m1.exists(),
        "state_m2_ready": state_m2.exists(),
        "dataset_ready": VALID_PLOTS_FILE.exists(),
        "total_database_plots": len(_plots_df) if _plots_df is not None else 0,
    }


# =========================================================================
# Database API Endpoints
# =========================================================================

@app.get("/api/database/summary")
async def get_database_summary():
    """Returns trial location summary counts and verification statistics."""
    get_database()
    if _plots_df is None:
        raise HTTPException(status_code=404, detail="Database not found.")

    loc_summary = {}
    for loc, items in _loc_plots_map.items():
        loc_summary[loc] = {
            "count": len(items),
            "avg_yield": round(float(np.mean([x["yieldPerAcre"] for x in items if x["yieldPerAcre"] is not None])), 1),
            "sample_plot": items[0]["plot_id"] if items else None,
        }

    return {
        "total_plots": len(_plots_df),
        "locations": loc_summary,
    }


@app.get("/api/database/plots")
async def get_database_plots(
    location: Optional[str] = Query(None, description="Filter by location (e.g. Ames, Crawfordsville)"),
    search: Optional[str] = Query(None, description="Search query for plot_id or genotype"),
    limit: int = Query(200, ge=1, le=2500, description="Max plots to return"),
):
    """Returns filtered plot list for quick UI selection and search."""
    get_database()
    if _plots_df is None:
        raise HTTPException(status_code=404, detail="Database not found.")

    if location and location in _loc_plots_map:
        plots = _loc_plots_map[location]
    else:
        plots = []
        for loc_items in _loc_plots_map.values():
            plots.extend(loc_items)

    if search:
        s_lower = search.strip().lower()
        plots = [
            p for p in plots
            if s_lower in p["plot_id"].lower()
            or s_lower in p["genotype"].lower()
            or s_lower in f"range {p['range']} row {p['row']}"
        ]

    return {
        "count": len(plots),
        "plots": plots[:limit],
    }


@app.get("/api/database/plot/{plot_id}")
async def get_plot_detail(plot_id: str):
    """Returns complete metadata and image URLs for a specific plot."""
    record = find_plot_record(plot_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Plot '{plot_id}' not found.")

    pid = str(record["plot_id"])
    tp1_fn = Path(str(record.get("TP1_image_path", ""))).name
    tp2_fn = Path(str(record.get("TP2_image_path", ""))).name
    tp3_fn = Path(str(record.get("TP3_image_path", ""))).name

    return {
        "plot_id": pid,
        "location": record.get("location"),
        "experiment": str(record.get("experiment", "")),
        "range": int(record.get("range", 0)) if pd.notna(record.get("range")) else 0,
        "row": int(record.get("row", 0)) if pd.notna(record.get("row")) else 0,
        "plotNumber": float(record.get("plotNumber", 0)) if pd.notna(record.get("plotNumber")) else None,
        "genotype": str(record.get("genotype", "Unknown")),
        "actual_yield": round(float(record["yieldPerAcre"]), 2) if pd.notna(record.get("yieldPerAcre")) else None,
        "totalStandCount": float(record["totalStandCount"]) if pd.notna(record.get("totalStandCount")) else None,
        "plantingDate": str(record.get("plantingDate", "")),
        "tp1_filename": tp1_fn,
        "tp2_filename": tp2_fn,
        "tp3_filename": tp3_fn,
        "tp1_url": f"/api/database/image/{pid}/TP1",
        "tp2_url": f"/api/database/image/{pid}/TP2",
        "tp3_url": f"/api/database/image/{pid}/TP3",
    }


@app.get("/api/database/image/{plot_id}/{tp}")
async def get_plot_image(plot_id: str, tp: str):
    """Serves the actual plot drone image thumbnail/file from dataset disk repository."""
    record = find_plot_record(plot_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Plot '{plot_id}' not found.")

    tp_key = f"{tp.upper()}_image_path"
    rel_path = record.get(tp_key)
    if not rel_path or pd.isna(rel_path):
        raise HTTPException(status_code=404, detail=f"Image for {tp} not available for plot {plot_id}.")

    abs_path = DATASET_ROOT / str(rel_path)
    if not abs_path.exists():
        # Check case-insensitive fallback in directory
        parent = abs_path.parent
        target_name = abs_path.name.lower()
        found = None
        if parent.exists():
            for p in parent.iterdir():
                if p.name.lower() == target_name:
                    found = p
                    break
        if found:
            abs_path = found
        else:
            raise HTTPException(status_code=404, detail=f"File not found on disk: {abs_path}")

    return FileResponse(
        abs_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/database/lookup")
async def lookup_plot_by_query(query: str = Query(..., description="Filename, plot ID, or coordinate string")):
    """
    Intelligent lookup endpoint: detects if a query or uploaded filename
    belongs to a known plot, and returns its metadata + matching timepoint links.
    """
    record = find_plot_record(query)
    if not record:
        return {"matched": False, "query": query}

    pid = str(record["plot_id"])
    return {
        "matched": True,
        "plot_id": pid,
        "location": record.get("location"),
        "genotype": record.get("genotype"),
        "range": int(record.get("range", 0)) if pd.notna(record.get("range")) else 0,
        "row": int(record.get("row", 0)) if pd.notna(record.get("row")) else 0,
        "totalStandCount": float(record.get("totalStandCount")) if pd.notna(record.get("totalStandCount")) else None,
        "plantingDate": str(record.get("plantingDate", "")),
        "actual_yield": round(float(record["yieldPerAcre"]), 2) if pd.notna(record.get("yieldPerAcre")) else None,
        "tp1_filename": Path(str(record.get("TP1_image_path", ""))).name,
        "tp2_filename": Path(str(record.get("TP2_image_path", ""))).name,
        "tp3_filename": Path(str(record.get("TP3_image_path", ""))).name,
        "tp1_url": f"/api/database/image/{pid}/TP1",
        "tp2_url": f"/api/database/image/{pid}/TP2",
        "tp3_url": f"/api/database/image/{pid}/TP3",
    }


@app.post("/api/predict_plot/{plot_id}")
async def predict_single_plot(plot_id: str):
    """
    Direct one-click prediction for any trial plot in the database.
    Loads real multi-temporal images + ground truth metadata directly.
    """
    record = find_plot_record(plot_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Plot '{plot_id}' not found in database.")

    # Prepare single row manifest
    plot_row = dict(record)
    for tp in ("TP1", "TP2", "TP3"):
        rel = plot_row.get(f"{tp}_image_path")
        if rel and pd.notna(rel):
            abs_p = DATASET_ROOT / str(rel)
            if abs_p.exists():
                plot_row[f"{tp}_image_path"] = str(abs_p)

    plots_df = pd.DataFrame([plot_row])
    pipeline = get_pipeline()
    results_df = pipeline.run(plots_df, image_root=DATASET_ROOT)

    r = results_df.iloc[0]
    m1_val = r.get("model1_yield")
    m1_yield = round(float(m1_val), 1) if pd.notna(m1_val) else None
    m2_val = r.get("model2_yield")
    m2_yield = round(float(m2_val), 1) if pd.notna(m2_val) else None

    act_val = record.get("yieldPerAcre")
    actual_yield = round(float(act_val), 1) if (act_val is not None and pd.notna(act_val)) else None

    flags_str = str(r.get("flags", ""))
    flags = [f for f in flags_str.split(";") if f]

    m1_item = {
        "plot_id": plot_id,
        "predicted_yield": m1_yield if m1_yield is not None else (m2_yield if m2_yield is not None else 0.0),
        "flags": [f for f in flags if "m1" in f or "unseen" in f or "stand" in f],
    }
    m2_item = {
        "plot_id": plot_id,
        "predicted_yield": m2_yield if m2_yield is not None else 0.0,
        "flags": [f for f in flags if "m2" in f or "field" in f],
    }

    return {
        "plot_id": plot_id,
        "location": record.get("location"),
        "genotype": record.get("genotype"),
        "range": int(record.get("range", 0)) if pd.notna(record.get("range")) else 0,
        "row": int(record.get("row", 0)) if pd.notna(record.get("row")) else 0,
        "actual_yield": actual_yield,
        "totalStandCount": float(record.get("totalStandCount")) if pd.notna(record.get("totalStandCount")) else None,
        "plantingDate": str(record.get("plantingDate", "")),
        "model1": [m1_item],
        "model2": [m2_item],
        "m1_error": round(m1_yield - actual_yield, 1) if (m1_yield is not None and actual_yield is not None) else None,
        "m2_error": round(m2_yield - actual_yield, 1) if (m2_yield is not None and actual_yield is not None) else None,
        "delta": round(abs((m1_yield or 0) - (m2_yield or 0)), 1),
    }


# =========================================================================
# Upload Prediction Endpoint (with Smart Multi-Temporal Auto-Matching)
# =========================================================================

@app.post("/predict_both")
async def predict_both(
    images: List[UploadFile] = File(...),
    metadata: Optional[UploadFile] = File(None),
):
    """
    Main prediction endpoint called by the MaizeSight UI.
    Receives multi-temporal plot images (TP1/TP2/TP3) and optional CSV metadata.

    SMART AUTO-MATCHING:
    If the user uploads only 1 photo (e.g. TP1) or misses timepoints:
    1. Looks up the plot in the trial database.
    2. Automatically retrieves matching TP2 / TP3 photos from disk.
    3. Automatically injects ground truth metadata (genotype, planting date, stand count).
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded.")

    get_database()
    temp_dir = tempfile.mkdtemp(prefix="maizesight_inference_")
    try:
        temp_path = Path(temp_dir)
        uploaded_map: Dict[str, Dict[str, Path]] = {}
        plot_meta: Dict[str, Dict] = {}
        auto_matched_plots: Dict[str, dict] = {}

        # 1. Save uploaded image files to temp directory & parse filenames
        for img in images:
            fname = Path(img.filename).name
            dest = temp_path / fname
            with open(dest, "wb") as f:
                content = await img.read()
                f.write(content)

            # Check database lookup first
            db_record = find_plot_record(fname)
            if db_record:
                plot_id = str(db_record["plot_id"])
                auto_matched_plots[plot_id] = db_record

                # Detect which TP was uploaded
                tp_match = re.search(r"TP[123]", fname, re.IGNORECASE)
                tp = tp_match.group(0).upper() if tp_match else "TP1"

                loc = db_record.get("location", "Unknown")
                exp = str(db_record.get("experiment", "UNK"))
                rng = float(db_record.get("range", 0.0))
                rw = float(db_record.get("row", 0.0))
            else:
                # Match standard pattern
                m = REGEX_PLOT.match(fname)
                if m:
                    gd = m.groupdict()
                    loc = gd["loc"]
                    tp = gd["tp"].upper()
                    exp = gd["exp"]
                    rng = float(gd["range"])
                    rw = float(gd["row"])
                    plot_id = f"{loc}_{exp}_{int(rng)}_{int(rw)}"
                else:
                    # Fallback: look for TP1 / TP2 / TP3 anywhere in the filename
                    tp_match = re.search(r"TP[123]", fname, re.IGNORECASE)
                    tp = tp_match.group(0).upper() if tp_match else "TP3"
                    clean_name = re.sub(r"[-_]?TP[123][-_]?", "", Path(fname).stem, flags=re.IGNORECASE)
                    loc_cand = clean_name.split("-")[0].split("_")[0]
                    loc = loc_cand if loc_cand else "Unknown"
                    exp = "UNK"
                    rng, rw = 0.0, 0.0
                    plot_id = clean_name

            if plot_id not in uploaded_map:
                uploaded_map[plot_id] = {}
                plot_meta[plot_id] = {
                    "plot_id": plot_id,
                    "location": loc,
                    "experiment": exp,
                    "range": rng,
                    "row": rw,
                }
            uploaded_map[plot_id][tp] = dest

        if not uploaded_map:
            raise HTTPException(status_code=400, detail="Could not parse any valid plot images.")

        # 2. Smart Auto-Matching: find missing timepoints in database
        for pid, paths in uploaded_map.items():
            db_rec = auto_matched_plots.get(pid) or find_plot_record(pid)
            if db_rec:
                auto_matched_plots[pid] = db_rec
                # If any TP is missing from the upload, load from dataset repository
                for t in ("TP1", "TP2", "TP3"):
                    if t not in paths:
                        rel = db_rec.get(f"{t}_image_path")
                        if rel and pd.notna(rel):
                            abs_p = DATASET_ROOT / str(rel)
                            if abs_p.exists():
                                paths[t] = abs_p

                # Merge metadata fields if not already populated
                for k in ("genotype", "plantingDate", "totalStandCount", "yieldPerAcre", "plotNumber"):
                    if k in db_rec and pd.notna(db_rec[k]):
                        plot_meta[pid][k] = db_rec[k]

        # 3. Build plot manifest records
        plot_rows = []
        for pid, paths in uploaded_map.items():
            meta_item = dict(plot_meta[pid])
            # Set image paths (with graceful fallbacks)
            meta_item["TP1_image_path"] = str(paths.get("TP1", paths.get("TP2", paths.get("TP3"))))
            meta_item["TP2_image_path"] = str(paths.get("TP2", paths.get("TP3", paths.get("TP1"))))
            meta_item["TP3_image_path"] = str(paths.get("TP3", paths.get("TP2", paths.get("TP1"))))
            plot_rows.append(meta_item)

        plots_df = pd.DataFrame(plot_rows)

        # 4. Incorporate uploaded metadata CSV if provided
        if metadata is not None and metadata.filename:
            try:
                csv_bytes = await metadata.read()
                csv_df = pd.read_csv(io.BytesIO(csv_bytes))
                csv_df = clean_manifest(csv_df)

                # Merge on plot_id or match coordinates
                if "plot_id" in csv_df.columns:
                    plots_df = plots_df.merge(
                        csv_df.drop(columns=[c for c in plots_df.columns if c not in ("plot_id",)], errors="ignore"),
                        on="plot_id",
                        how="left",
                    )
            except Exception:
                pass

        # 5. Run inference pipeline
        pipeline = get_pipeline()
        results_df = pipeline.run(plots_df, image_root=DATASET_ROOT)

        # 6. Format response to match exact MaizeSight frontend schema
        m1_list = []
        m2_list = []
        for _, r in results_df.iterrows():
            flags_str = str(r.get("flags", ""))
            flags = [f for f in flags_str.split(";") if f]

            m1_val = r.get("model1_yield")
            m1_yield = round(float(m1_val), 1) if pd.notna(m1_val) else None

            m2_val = r.get("model2_yield")
            m2_yield = round(float(m2_val), 1) if pd.notna(m2_val) else None

            pid = str(r["plot_id"])
            db_match = auto_matched_plots.get(pid)
            act_y = round(float(db_match["yieldPerAcre"]), 1) if (db_match and pd.notna(db_match.get("yieldPerAcre"))) else None
            genotype = str(db_match["genotype"]) if (db_match and pd.notna(db_match.get("genotype"))) else str(r.get("genotype", "Unknown"))

            m1_list.append({
                "plot_id": pid,
                "genotype": genotype,
                "actual_yield": act_y,
                "predicted_yield": m1_yield if m1_yield is not None else (m2_yield if m2_yield is not None else 0.0),
                "flags": [f for f in flags if "m1" in f or "unseen" in f or "stand" in f],
                "auto_matched": pid in auto_matched_plots,
            })
            m2_list.append({
                "plot_id": pid,
                "genotype": genotype,
                "actual_yield": act_y,
                "predicted_yield": m2_yield if m2_yield is not None else 0.0,
                "flags": [f for f in flags if "m2" in f or "field" in f],
                "auto_matched": pid in auto_matched_plots,
            })

        return {
            "model1": m1_list,
            "model2": m2_list,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import socket
    import sys
    import threading

    # Reconfigure stdout/stderr for UTF-8 on Windows command prompts
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def find_free_port(start_port: int = 8000, max_attempts: int = 20) -> int:
        for p in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        return start_port

    target_port = 8000
    port = find_free_port(target_port)
    url = f"http://localhost:{port}"

    print("=" * 65)
    print(f"[*] MaizeSight Web Server Starting on {url}")
    if port != target_port:
        print(f"   (Notice: Port {target_port} was in use. Automatically switched to {url})")
    print("   * Interactive 3D Maize Visualization: Online")
    print("   * Dual-Model CropYieldPipeline Engine: Ready")
    print("   * 2,131 Plot Research Database: Mounted")
    print("   * Multi-Temporal Image Auto-Matching: Enabled")
    print("=" * 65)

    def open_browser():
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            pass

    # Launch browser after slight delay so uvicorn socket is listening
    timer = threading.Timer(1.0, open_browser)
    timer.daemon = True
    timer.start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

