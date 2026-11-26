"""Photo features: what a listing's pictures say about condition and finish.

For each sale, the first PHOTOS_PER_SALE public photos (PropTx order, the
preferred photo first) are downloaded at medium size, embedded with CLIP,
and thrown away. Only numbers are kept, per sale:

* the mean CLIP image embedding (512 numbers), and
* zero-shot condition scores: for each prompt pair ("a newly renovated modern
  kitchen" vs "an old dated kitchen"), how much more the photos look like the
  first than the second, averaged and maxed over the photos.

Images are never written to disk. Listing photos belong to the brokerages and
photographers; this keeps only derived features, on the server.

    python -m src.photos --out data/raw/photos       # resumable; skips done sales
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger("photos")

PHOTOS_PER_SALE = 8
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
RESO_URL = "https://query.ampre.ca/odata/"
USER_AGENT = "ottawa-home-value/0.1 (+https://github.com/mertmetin/ottawa-home-value)"

# (positive, negative): score = similarity(positive) - similarity(negative)
PROMPTS = {
    "kitchen_updated": ("a photo of a newly renovated modern kitchen", "a photo of an old dated kitchen"),
    "bath_updated": ("a photo of a modern renovated bathroom", "a photo of an old dated bathroom"),
    "living_updated": ("a photo of a bright updated living room with hardwood floors",
                       "a photo of a worn living room with old carpet"),
    "luxury": ("a photo of a luxury home interior with high-end finishes", "a photo of a modest plain home interior"),
    "needs_work": ("a photo of a house that needs repairs", "a photo of a well maintained house"),
    "basement_finished": ("a photo of a finished basement rec room", "a photo of an unfinished basement"),
    "yard_large": ("a photo of a large private backyard", "a photo of a small backyard"),
    "pool": ("a photo of an in-ground swimming pool", "a photo of a backyard without a pool"),
    "curb_appeal": ("a photo of an attractive house exterior with landscaping", "a photo of a plain house exterior"),
}
ZS_NAMES = [f"zs_{k}_{agg}" for k in PROMPTS for agg in ("mean", "max")]


# --------------------------------------------------------------------------
# PropTx media
# --------------------------------------------------------------------------

def _get(resource: str, params: dict, token: str) -> list[dict]:
    # PropTx wants %20 for spaces; '+' turns a filter into a type error.
    url = RESO_URL + resource + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token, "User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["value"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"PropTx {resource}: HTTP {e.code} {e.read()[:200]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt * 2)
    return []


def photo_urls(keys: list[str], token: str, per_sale: int = PHOTOS_PER_SALE) -> dict[str, list[str]]:
    """Public, active, medium-size photo URLs per listing, preferred first, then by order."""
    rows = _get("Media", {
        "$filter": "ResourceRecordKey in (" + ",".join("'" + k.replace("'", "''") + "'" for k in keys) + ")"
                   " and MediaCategory eq 'Photo' and ImageSizeDescription eq 'Medium'",
        "$select": "ResourceRecordKey,MediaURL,Order,PreferredPhotoYN,Permission,MediaStatus",
        "$top": "5000",
    }, token)
    by: dict[str, list[tuple]] = {}
    for r in rows:
        perm = r.get("Permission") or ["Public"]
        if r.get("MediaStatus") != "Active" or "Public" not in perm or not r.get("MediaURL"):
            continue
        by.setdefault(r["ResourceRecordKey"], []).append(
            (not r.get("PreferredPhotoYN"), r.get("Order") if r.get("Order") is not None else 1e9, r["MediaURL"]))
    return {k: [u for *_, u in sorted(v)[:per_sale]] for k, v in by.items()}


def _download(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except Exception:  # a missing photo is not worth failing a batch over
        return None


# --------------------------------------------------------------------------
# CLIP
# --------------------------------------------------------------------------

class Embedder:
    def __init__(self, device: str = "cpu"):
        import open_clip
        import torch

        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=device)
        self.model.eval()
        tok = open_clip.get_tokenizer(MODEL)
        texts = [t for pair in PROMPTS.values() for t in pair]
        with torch.no_grad():
            t = self.model.encode_text(tok(texts))
        self.text = (t / t.norm(dim=-1, keepdim=True)).numpy()

    def images(self, blobs: list[bytes]) -> np.ndarray:
        from PIL import Image

        ims = []
        for b in blobs:
            try:
                ims.append(self.preprocess(Image.open(io.BytesIO(b)).convert("RGB")))
            except Exception:
                continue
        if not ims:
            return np.zeros((0, self.text.shape[1]), np.float32)
        with self.torch.no_grad():
            e = self.model.encode_image(self.torch.stack(ims))
        return (e / e.norm(dim=-1, keepdim=True)).numpy()

    def zero_shot(self, emb: np.ndarray) -> np.ndarray:
        """Per prompt pair: mean and max over photos of sim(pos) - sim(neg)."""
        sims = emb @ self.text.T * 100  # CLIP logit scale
        diff = sims[:, 0::2] - sims[:, 1::2]
        return np.column_stack([diff.mean(0), diff.max(0)]).reshape(-1)


# --------------------------------------------------------------------------
# Job
# --------------------------------------------------------------------------

def done_keys(out: Path) -> set[str]:
    keys: set[str] = set()
    for f in sorted(out.glob("chunk_*.npz")):
        keys.update(np.load(f, allow_pickle=False)["keys"].tolist())
    return keys


def load(out: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(keys, n_photos, mean embedding, zero-shot scores) for every finished sale."""
    parts = [np.load(f, allow_pickle=False) for f in sorted(Path(out).glob("chunk_*.npz"))]
    if not parts:
        return np.array([]), np.array([]), np.zeros((0, 512)), np.zeros((0, len(ZS_NAMES)))
    return tuple(np.concatenate([p[k] for p in parts]) for k in ("keys", "n", "emb", "zs"))


def run(keys: list[str], out: Path, token: str, chunk: int = 300, threads: int = 8) -> None:
    out.mkdir(parents=True, exist_ok=True)
    todo = [k for k in keys if k not in done_keys(out)]
    log.info("%d sales, %d already done, %d to go", len(keys), len(keys) - len(todo), len(todo))
    emb = Embedder()
    start, n_done = time.time(), 0
    next_chunk = len(list(out.glob("chunk_*.npz")))
    with ThreadPoolExecutor(threads) as pool:
        for c in range(0, len(todo), chunk):
            part = todo[c:c + chunk]
            urls: dict[str, list[str]] = {}
            for i in range(0, len(part), 15):
                urls.update(photo_urls(part[i:i + 15], token))
            K, N, E, Z = [], [], [], []
            for k in part:
                blobs = [b for b in pool.map(_download, urls.get(k, [])) if b]
                e = emb.images(blobs)
                K.append(k)
                N.append(len(e))
                if len(e):
                    E.append(e.mean(0))
                    Z.append(emb.zero_shot(e))
                else:
                    E.append(np.full(emb.text.shape[1], np.nan, np.float32))
                    Z.append(np.full(len(ZS_NAMES), np.nan, np.float32))
            np.savez(out / f"chunk_{next_chunk:05d}.npz", keys=np.array(K), n=np.array(N),
                     emb=np.array(E, np.float16), zs=np.array(Z, np.float32))
            next_chunk += 1
            n_done += len(part)
            rate = n_done / (time.time() - start)
            log.info("%d/%d sales (%.1f/s, ~%.0f min left), %d with photos in this chunk",
                     n_done, len(todo), rate, (len(todo) - n_done) / max(rate, 1e-9) / 60, sum(x > 0 for x in N))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/photos")
    ap.add_argument("--limit", type=int, default=0, help="only the first N sales (for a trial run)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import psycopg

    with psycopg.connect(os.environ["DATABASE_URL"]) as c:
        keys = [r[0] for r in c.execute(
            "SELECT listing_key FROM sold_listings WHERE county = 'Ottawa' ORDER BY contract_date DESC NULLS LAST")]
    if a.limit:
        keys = keys[:a.limit]
    run(keys, Path(a.out), os.environ["PROPTX_VOW_TOKEN"])


if __name__ == "__main__":
    main()
