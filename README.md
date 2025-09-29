

**HOW TO USE PYTHON TO SEGREGATE ITEMS USING PYTHON**
# retail_csv_categorizer.py
import os, re, sys
import pandas as pd
import numpy as np

# ---- Try semantic embeddings (preferred) ----
USE_EMBEDDINGS = True
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    USE_EMBEDDINGS = False

# ---------- Categories (Google-style retail taxonomy) ----------
CATEGORIES = [
    "Home & Garden > Decor > Home Décor",
    "Home & Garden > Kitchen & Dining",
    "Apparel & Accessories > Clothing & Accessories",
    "Toys & Games",
    "Office Supplies > Stationery & Gift Wrap",
    "Home & Garden > Furniture & Storage",
    "Seasonal & Festive",
    "Home & Garden > Garden & Outdoor",
    "Bath & Laundry",
    "Electrical & Lighting",
    "Pet Supplies",
    "Baby & Toddler",
    "Beauty & Personal Care",
    "Food & Confectionery",
]

# ---------- Overrides for precision ----------
OVERRIDES = [
    (r"\bHOT\s*WATER\s*BOTTLE\b",                    "Bath & Laundry"),
    (r"\bDOOR\s*MAT|DOORMAT\b",                      "Home & Garden > Decor > Home Décor"),
    (r"\bLANTERN(S)?\b",                             "Electrical & Lighting"),
    (r"\bCANDLE\s*HOLDER|T-?LIGHT\s*HOLDER\b",       "Home & Garden > Decor > Home Décor"),
    (r"\bCLOCK(S)?\b",                               "Home & Garden > Furniture & Storage"),
    (r"\bBUNTING|PARTY|BALLOON|CONFETTI\b",          "Seasonal & Festive"),
    (r"\bJIGSAW|PUZZLE|DOLL|TOY|PLAYHOUSE|BLOCK(S)?\b","Toys & Games"),
    (r"\bMUG|TEAPOT|TEA\s*TOWEL|PLATE|BOWL|COASTER|TRAY|JAR|GLASS\b","Home & Garden > Kitchen & Dining"),
    (r"\bSCARF|HANDBAG|PURSE|BAG|BRACELET|NECKLACE|UMBRELLA\b","Apparel & Accessories > Clothing & Accessories"),
    (r"\bPLANTER|POTTING|WATERING\s*CAN|BIRD\s*HOUSE|GARDEN\b","Home & Garden > Garden & Outdoor"),
    (r"\bCHRISTMAS|XMAS|EASTER|VALENTINE|HALLOWEEN|ADVENT|NOEL\b","Seasonal & Festive"),
    (r"\bCARD|NOTEBOOK|DIARY|WRAP|RIBBON|STICKER|PEN|PENCIL|POSTCARD|ENVELOPE\b","Office Supplies > Stationery & Gift Wrap"),
    (r"\bCABINET|RACK|SHELF|DRAWER|STOOL|TABLE|CHAIR|STORAGE\s*BOX\b","Home & Garden > Furniture & Storage"),
    (r"\bBATTERY|LED\b",                             "Electrical & Lighting"),
    (r"\bDOG|CAT|PET\b",                             "Pet Supplies"),
    (r"\bBABY|NEWBORN|PRAM\b",                       "Baby & Toddler"),
    (r"\bLIP\s*BALM|COSMETIC|MAKE[-\s]?UP|NAIL\b",   "Beauty & Personal Care"),
    (r"\bCHOC|CHOCOLATE|CANDY|SWEET|FUDGE|MARSHMALLOW|COFFEE|TEA\b","Food & Confectionery"),
]

# ---------- Embedding prototypes ----------
PROTOTYPES = {
    "Home & Garden > Decor > Home Décor": ["home decor ornament","candle holder","lantern decor","doormat","photo frame"],
    "Home & Garden > Kitchen & Dining":   ["ceramic mug","teapot","dinner plate","bowl","glassware","cutlery"],
    "Apparel & Accessories > Clothing & Accessories": ["scarf","handbag","purse","fashion bracelet","necklace","umbrella"],
    "Toys & Games": ["jigsaw puzzle","doll","toy blocks","playhouse","kids toy"],
    "Office Supplies > Stationery & Gift Wrap": ["greeting card","notebook","gift wrap","ribbon","sticker","diary"],
    "Home & Garden > Furniture & Storage": ["coat rack","storage box","cabinet","shelf","stool","clock"],
    "Seasonal & Festive": ["christmas ornament","xmas decoration","advent calendar","halloween decor","valentine bunting"],
    "Home & Garden > Garden & Outdoor": ["flower planter","watering can","garden ornament","bird house","outdoor decor"],
    "Bath & Laundry": ["bath towel","soap dish","bathroom accessory","toilet roll holder","hot water bottle"],
    "Electrical & Lighting": ["led string lights","battery lights","electric lamp","lantern light"],
    "Pet Supplies": ["dog bowl","cat bowl","pet accessory"],
    "Baby & Toddler": ["baby toy","newborn gift","pram accessory"],
    "Beauty & Personal Care": ["lip balm","cosmetic bag","nail kit","makeup set","hand cream"],
    "Food & Confectionery": ["chocolate bar","candy sweets","fudge gift","tea tin","coffee beans"],
}

# ---------- Utility ----------
def clean_text(s): return re.sub(r"\s+"," ",str(s)).strip()

def apply_overrides(desc):
    U = desc.upper()
    for pat, cat in OVERRIDES:
        if re.search(pat, U):
            return cat
    return None

def embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_prototypes(model):
    labels, phrases = list(PROTOTYPES.keys()), []
    for k in PROTOTYPES: phrases.append("; ".join(PROTOTYPES[k]))
    return labels, model.encode(phrases, normalize_embeddings=True)

def predict_semantic(model, labels, proto_embs, texts):
    txt_embs = model.encode(texts, normalize_embeddings=True)
    sims = cosine_similarity(txt_embs, proto_embs)
    best_idx = sims.argmax(axis=1)
    return [labels[i] for i in best_idx], sims.max(axis=1)

def categorize_series(desc_series):
    out, pend_idx, pend_texts = [], [], []
    for i,d in enumerate(desc_series.fillna("").astype(str)):
        d_clean = clean_text(d)
        ov = apply_overrides(d_clean)
        if ov: out.append(ov)
        else: out.append(None); pend_idx.append(i); pend_texts.append(d_clean)
    if USE_EMBEDDINGS and pend_texts:
        model = embed_model()
        labels, proto_embs = build_prototypes(model)
        cats, scores = predict_semantic(model, labels, proto_embs, pend_texts)
        for j, idx in enumerate(pend_idx):
            if scores[j] >= 0.38: out[idx] = cats[j]
    for i in range(len(out)):
        if out[i] is None: out[i] = "Home & Garden > Decor > Home Décor"
    return pd.Series(out, index=desc_series.index)

# ---------- Main ----------
if __name__ == "__main__":
    csv_path = "Retail.csv"   # your CSV file
    df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)
    if "Description" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "description":
                df.rename(columns={c:"Description"}, inplace=True)
                break
    df["Category"] = categorize_series(df["Description"])
    out_csv = "Retail_categorized.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print("✅ Saved:", out_csv)
