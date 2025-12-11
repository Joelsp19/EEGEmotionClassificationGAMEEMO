import os, zipfile, tarfile, gzip
import os
from pathlib import Path
import numpy as np
import cv2
import fitz  # PyMuPDF
import glob
import pandas as pd
from typing import Optional, Dict, List
import pdfplumber
import re
import pandas as pd
import zipfile, pathlib

def extract_data():
    dest = pathlib.Path("/content/gammemo")  # your folder (double-check the path name)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(p, "r") as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"Corrupt file inside zip: {bad}")
        z.extractall(dest)
    print("Extracted to", dest)

def render_pdf_to_gray(pdf_path: Path, page_index: int = 0, dpi: int = 300) -> np.ndarray:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Path does not exist: {pdf_path}")
    doc = fitz.open(str(pdf_path))
    if page_index >= len(doc):
        raise IndexError(f"PDF has {len(doc)} pages, requested {page_index}.")
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def load_image_gray(path: Path, page_index: int = 0, dpi: int = 300) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return render_pdf_to_gray(p, page_index=page_index, dpi=dpi)
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Could not read image: {p}")
    return img

def _align_ecc(T: np.ndarray, F: np.ndarray) -> np.ndarray:
    t = cv2.normalize(T.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    f = cv2.normalize(F.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    t_blur = cv2.GaussianBlur(t, (5,5), 0); f_blur = cv2.GaussianBlur(f, (5,5), 0)
    warp = np.eye(2,3,dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 5000, 1e-7)
    _, warp = cv2.findTransformECC(t_blur, f_blur, warp, cv2.MOTION_AFFINE, crit)
    return cv2.warpAffine(F, warp, (T.shape[1], T.shape[0]),
                          flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)

def _align_orb(T: np.ndarray, F: np.ndarray) -> np.ndarray:
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(T, None)
    kp2, des2 = orb.detectAndCompute(F, None)
    if des1 is None or des2 is None: raise RuntimeError("ORB: no descriptors")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:200]
    if len(matches) < 10: raise RuntimeError("ORB: not enough matches")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    M,_ = cv2.estimateAffine2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None: raise RuntimeError("ORB: affine failed")
    return cv2.warpAffine(F, M, (T.shape[1], T.shape[0]), flags=cv2.INTER_LINEAR)

def align_images(T: np.ndarray, F: np.ndarray) -> np.ndarray:
    try: return _align_ecc(T,F)
    except Exception: return _align_orb(T,F)

def compute_diff_binary(T: np.ndarray, A: np.ndarray) -> np.ndarray:
    t = cv2.GaussianBlur(T,(5,5),0); a = cv2.GaussianBlur(A,(5,5),0)
    diff = cv2.absdiff(t,a)
    # suppress thin template edges
    edges = cv2.Canny(t,70,170)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    diff[edges>0] = 0
    th = cv2.threshold(diff,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return th

def stroke_filter(mask_bin: np.ndarray, min_stroke_px: int = 9) -> np.ndarray:
    k = max(3,int(round(min_stroke_px)))
    if k%2==0: k+=1
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    er = cv2.erode(mask_bin,K,1)
    th = cv2.dilate(er,K,1)
    return th

def fixed_roi_from_fractions(shape, roi_frac):
    H,W = shape[:2]
    x0 = int(W*roi_frac[0]); y0 = int(H*roi_frac[1])
    x1 = int(W*roi_frac[2]); y1 = int(H*roi_frac[3])
    x0 = max(0,min(x0,W-1)); x1 = max(x0+1,min(x1,W))
    y0 = max(0,min(y0,H-1)); y1 = max(y0+1,min(y1,H))
    return (x0,y0,x1,y1)

def ring_like_score_adaptive(thick_mask, cx, cy, MA, ma, angle_deg,
                             min_band_px=3, rel_band=0.05):
    """
    Returns a score in [0,1] of how 'ring-like' the contour area is around (cx,cy).
    Band thickness is adaptive: max(min_band_px, rel_band * min(MA,ma)).
    Score = (ink pixels in ring band) / (ring band area).
    """
    h, w = thick_mask.shape[:2]
    cx, cy = int(cx), int(cy)
    # choose band thickness proportional to ellipse size
    band = max(int(round(rel_band * min(MA, ma))), int(min_band_px))

    # outer/inner axes
    ax_out = (max(int(MA/2)+1, 1), max(int(ma/2)+1, 1))
    ax_in  = (max(ax_out[0]-band, 1), max(ax_out[1]-band, 1))

    # draw masks (page coords)
    outer = np.zeros((h,w), np.uint8)
    inner = np.zeros((h,w), np.uint8)
    cv2.ellipse(outer, (cx,cy), ax_out, angle_deg, 0, 360, 255, -1)
    cv2.ellipse(inner, (cx,cy), ax_in,  angle_deg, 0, 360, 255, -1)

    ring_band = ((outer > 0) & (inner == 0))
    band_area = int(ring_band.sum())
    if band_area == 0:
        return 0.0

    ink_in_band = int((ring_band & (thick_mask > 0)).sum())
    return ink_in_band / float(band_area)

def find_oval_centers_in_roi(thick_mask: np.ndarray, roi: tuple,
                             min_area=1200, max_area=90000,
                             aspect_range=(1.0, 4.0), ring_tol_px=5):
    """
    Finds hand-drawn ovals (as contours) inside ROI and returns centers in page coords.
    Uses ellipse fit + ring-ness check to avoid filled blobs.
    """
    x0,y0,x1,y1 = roi
    roi_mask = thick_mask[y0:y1, x0:x1].copy()
    edges = cv2.Canny(roi_mask, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    centers = []   # (cx_page, cy_page)
    ellipses = []  # (center, axes, angle) in page coords

    for cnt in contours:
        if len(cnt) < 40:
            continue
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)
            if not(min_area <= area <= max_area):
                continue
            ellipse = cv2.fitEllipse(cnt)
            (cx,cy),(MA,ma),ang = ellipse
            A,B = max(MA,ma), min(MA,ma)
            if B <= 0:
                continue
            aspect = A/B
            if not(aspect_range[0] <= aspect <= aspect_range[1]):
                continue

            # ADAPTIVE ring-ness
            cxp, cyp = int(cx) + x0, int(cy) + y0  # page coords
            ring_score = ring_like_score_adaptive(
                thick_mask, cxp, cyp, MA, ma, ang,
                min_band_px=3,          # try 3–6 depending on DPI
                rel_band=0.05           # 5% of minor axis; try 0.04–0.07 if needed
            )
            # accept if enough of the band has ink
            if ring_score < 0.10:       # was hard-coded 0.08 against outer; this is more tolerant
                continue

            # center via contour moments (robust to irregular ovals)
            center_fit = (int(round(cx)) + x0, int(round(cy)) + y0)
            centers.append(center_fit)
            ellipses.append(((cxp,cyp),(MA,ma),ang))

    return centers, ellipses

def bin_by_center(centers, roi, img_shape):
    """
    Map center points to 2×9 fixed bins inside ROI.
    Returns winners per row (1..9 or None) based on the point falling in a bin.
    If multiple centers land in the same row, choose the one with the largest y-distance
    from the row midline (i.e., more central vertically) — simple disambiguation.
    """
    x0,y0,x1,y1 = roi
    H = y1 - y0; H2 = H//2
    x_edges = np.linspace(x0, x1, 10, dtype=int)

    # Split centers by row
    top_pts, bot_pts = [], []
    for (cx,cy) in centers:
        if y0 <= cy < y0+H2:   top_pts.append((cx,cy))
        elif y0+H2 <= cy < y1: bot_pts.append((cx,cy))

    def pick_bin(pts, row_top):
        if not pts: return None
        # if multiple, choose the one closest to the row vertical mid
        row_mid_y = row_top + H2//2
        pts_sorted = sorted(pts, key=lambda p: abs(p[1]-row_mid_y))
        cx,cy = pts_sorted[0]
        # which bin?
        for i in range(9):
            if x_edges[i] <= cx < x_edges[i+1]:
                return i+1
        # clamp if on the rightmost boundary
        return 9

    valence_bin = pick_bin(top_pts, y0)
    arousal_bin = pick_bin(bot_pts, y0+H2)
    return valence_bin, arousal_bin

def process_sam_center_bins(template_image,
                            filled_path,
                            outdir="./sam_out",
                            page_index=0, dpi=200,
                            min_stroke_px=9,
                            roi_frac=(0.145, 0.455, 0.855, 0.735),
                            min_area=1200, max_area=90000,
                            aspect_range=(1.0, 4.0), ring_tol_px=1):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    # 1) Load + align
    T = template_image
    F = load_image_gray(filled_path, page_index=page_index, dpi=dpi)
    if T.shape != F.shape:
        F = cv2.resize(F, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
    A = align_images(T, F)

    # 2) Diff + stroke filter
    diff_bin = compute_diff_binary(T, A)
    thick = stroke_filter(diff_bin, min_stroke_px=min_stroke_px)

    # 3) Fixed ROI & circle centers
    roi = fixed_roi_from_fractions(A.shape, roi_frac)
    centers, ellipses = find_oval_centers_in_roi(
        thick, roi, min_area=min_area, max_area=max_area,
        aspect_range=aspect_range, ring_tol_px=ring_tol_px
    )

    # 4) Map center(s) to bins (one per row)
    v_bin, a_bin = bin_by_center(centers, roi, A.shape)

    # 5) Debug overlays
    vis = cv2.cvtColor(A, cv2.COLOR_GRAY2BGR)
    x0,y0,x1,y1 = roi
    H = y1-y0; H2 = H//2
    x_edges = np.linspace(x0, x1, 10, dtype=int)
    # ROI + grid
    cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,0),2)
    cv2.line(vis,(x0,y0+H2),(x1,y0+H2),(0,255,0),1)
    for xe in x_edges: cv2.line(vis,(xe,y0),(xe,y1),(0,255,0),1)
    # centers
    for (cx,cy) in centers:
        cv2.circle(vis,(int(cx),int(cy)),4,(0,0,255),-1)
    # labels
    if v_bin is not None:
        cx = (x_edges[v_bin-1]+x_edges[v_bin])//2; cy = y0+H2//2
        cv2.putText(vis,f"V{v_bin}",(cx-14,cy-6),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
    if a_bin is not None:
        cx = (x_edges[a_bin-1]+x_edges[a_bin])//2; cy = y0+H2+H2//2
        cv2.putText(vis,f"A{a_bin}",(cx-14,cy-6),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

   
    cv2.imwrite(str(out/f"06_overlay_centers_bins_{filled_path.name}.png"), vis)

    return {
        "valence_bin": v_bin, "arousal_bin": a_bin,
        "centers": centers, "roi_frac": roi_frac, "roi_px": (x0,y0,x1,y1),
        "files": {
            "overlay":       out/f"06_overlay_centers_bins_{filled_path.name}.png",
        }
    }

def _iter_pattern(pattern: str):
    pattern = os.path.expanduser(os.path.expandvars(pattern))
    if os.path.isabs(pattern):
        for p in glob.glob(pattern, recursive=True):
            yield Path(p)
    else:
        yield from Path().glob(pattern)

def process_glob(glob_pattern: str, template_img, out_dir: str, index):
    out_dir = Path(out_dir)
    rows: List[Dict[str, Optional[int]]] = []
    for pdf in sorted(_iter_pattern(glob_pattern)):
        if pdf.suffix.lower() != ".pdf":
            continue
        rec = process_sam_center_bins(
            template_image=template_img,
            filled_path=pdf,
            outdir=out_dir,
            page_index=0, dpi=200,
            min_stroke_px=5,
            # Tune once so the green box tightly wraps both SAM rows:
            roi_frac=(0.145, 0.505, 0.855, 0.785),
            # If a real oval is thin/small, relax these slightly:
            min_area=1000, max_area=120000, aspect_range=(1.0, 4.2), ring_tol_px=10
        )
        # only include the name of the pdf, valence_bin and the arousal_bin
        res = {
          "Path": pdf.name,
          "Subject": f"S{str(index).zfill(2)}",
          "Valence": rec["valence_bin"],
          "Arousal": rec["arousal_bin"]
        }
        rows.append(res)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sam_scores.csv"
    df.to_csv(csv_path, index=False)
    print(df)
    print("CSV:", csv_path)
    return rows

def get_SAM_scores_clean():
    total_rows = []
    template_img = render_pdf_to_gray("SAM_master_exact.pdf")

    for i in range(1,29):
        index_string = str(i).zfill(2)
        total_rows.extend(process_glob(f"gammemo/GAMEEMO/(S{index_string})/SAM Ratings/*.pdf", template_img,f"/content/gammemo/GAMEEMO/(S{index_string})/SAM Ratings/", i))
        print(i)

    df = pd.DataFrame(total_rows)
    df.to_csv("survey_clean_SAM_scores.csv", index=False)

def process_pdf_text(glob, index):
  records = []
  for path in _iter_pattern(glob):
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()

            # Example regex capture groups
            gender = re.search(r"Gender:\s*(\w+)", text)
            age = re.search(r"Age:\s*(\d+)", text)
            subject = re.search(r"S\d+", text)
            anchor = re.search(r"questions?\s+to\s+1-10\.", text, flags=re.I)
            end = re.search(r"\bSELF-?ASSESSMENT\s+MANIKIN\b", text[anchor.end():], flags=re.I)
            block = text[anchor.end():]
            scores = re.findall(r"\b(?:10|[1-9])\b", block)

            record = {
                "Path": path.name,
                "Gender": gender.group(1) if gender else None,
                "Age": int(age.group(1)) if age else None,
                "Subject": subject.group(0) if subject else None,
                "Satisfaction": scores[0] if len(scores) > 0 else None,
                "Boring": scores[1] if len(scores) > 1 else None,
                "Horrible": scores[2] if len(scores) > 2 else None,
                "Calm": scores[3] if len(scores) > 3 else None,
                "Funny": scores[4] if len(scores) > 4 else None,
            }
            records.append( record)
  return records

def get_survey_clean():
    total_records = []
    for i in range(1,29):
        index_string = str(i).zfill(2)
        total_records.extend(process_pdf_text(f"gammemo/GAMEEMO/(S{index_string})/SAM Ratings/*.pdf", i))

    df = pd.DataFrame(total_records)
    df.to_csv("survey_clean.csv", index=False)

def merge_csv():
    csv1 = "SAM_scores_cleaned.csv"
    csv2 = "survey_clean.csv"

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    merged_df = pd.merge(df1, df2, on=['Subject', 'Path'], how='outer')
    merged_df.to_csv('final_SAM_scores.csv')

if __name__ == "__main__":
    extract_data()
    get_SAM_scores_clean() # gets the valence and arousal bins  (manequin)
    get_survey_clean() # gets the survey responses
    merge_csv() # merge to get final csv