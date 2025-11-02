# 📍 WHERE TO PUT YOUR DATA - SIMPLE GUIDE

**Date: October 31, 2025**

---

## 🎯 YOUR TWO DATA FOLDERS

### 1️⃣ **Sentinel-2 L1C Data**
**PUT HERE:** 
```
C:\Users\harsh\PS-10\PS10_Input_Data\Sentinel2_L1C\
```

**What to put:**
- All your Sentinel-2 L1C satellite images
- Format: `.tif` (GeoTIFF) files
- Can be individual bands or multi-band images
- Original SAFE folders are also OK

**Example files:**
```
Sentinel2_L1C/
├── location1_20250101_t1.tif   (first/older image)
├── location1_20250215_t2.tif   (second/newer image)
├── location2_20250101_t1.tif
├── location2_20250215_t2.tif
└── ...
```

---

### 2️⃣ **LISS4 MX70 L2 Data**
**PUT HERE:**
```
C:\Users\harsh\PS-10\PS10_Input_Data\LISS4_L2\
```

**What to put:**
- All your LISS4 MX70 L2 satellite images
- Format: `.tif` (GeoTIFF) or `.jp2` (JPEG2000) files
- Multi-spectral bands

**Example files:**
```
LISS4_L2/
├── location1_20250101_t1.tif   (first/older image)
├── location1_20250215_t2.tif   (second/newer image)
├── location2_20250101_t1.tif
├── location2_20250215_t2.tif
└── ...
```

---

## 📋 IMPORTANT: Image Pairs

### You need PAIRS of images:
- **t1** = First/Older image (Time 1)
- **t2** = Second/Newer image (Time 2)

### Naming convention:
```
anything_t1.tif  ←→  anything_t2.tif
```

**Examples:**
- ✅ `mumbai_jan_t1.tif` and `mumbai_jan_t2.tif`
- ✅ `site1_before_t1.tif` and `site1_before_t2.tif`
- ✅ `S2_delhi_20250101_t1.tif` and `S2_delhi_20250101_t2.tif`

---

## 🚀 AFTER YOU PUT THE DATA

### Step 1: Check folders contain your files
```powershell
# In PowerShell, run:
ls C:\Users\harsh\PS-10\PS10_Input_Data\Sentinel2_L1C\
ls C:\Users\harsh\PS-10\PS10_Input_Data\LISS4_L2\
```

### Step 2: Run the processing
```powershell
cd C:\Users\harsh\PS-10
python master_ps10_windows.py --run PS10_Input_Data
```

### Step 3: Get your results
The system will create:
```
C:\Users\harsh\PS-10\PS10_31-Oct-2025_XBosonAI.zip  ← SUBMIT THIS!
```

---

## 📝 FILE FORMAT REQUIREMENTS

### Sentinel-2 L1C:
- ✅ GeoTIFF (.tif)
- ✅ JPEG2000 (.jp2)
- ✅ Original SAFE folders
- Must have georeference info (CRS/projection)
- Typical bands: Blue, Green, Red, NIR, SWIR, etc.

### LISS4 MX70 L2:
- ✅ GeoTIFF (.tif)
- ✅ JPEG2000 (.jp2)
- Must have georeference info (CRS/projection)
- Typical bands: Pan, SWIR, NIR, Red, Green

---

## 🎨 VISUAL GUIDE

```
YOUR DATA GOES HERE:
====================

PS-10/
└── PS10_Input_Data/
    │
    ├── Sentinel2_L1C/        👈 PUT SENTINEL-2 FILES HERE
    │   ├── file1_t1.tif
    │   ├── file1_t2.tif
    │   ├── file2_t1.tif
    │   └── file2_t2.tif
    │
    ├── LISS4_L2/             👈 PUT LISS4 FILES HERE
    │   ├── file1_t1.tif
    │   ├── file1_t2.tif
    │   ├── file2_t1.tif
    │   └── file2_t2.tif
    │
    └── Metadata/             👈 (Optional) Put notes here
        └── readme.txt
```

---

## ⚡ QUICK START CHECKLIST

- [ ] Downloaded Sentinel-2 L1C data
- [ ] Downloaded LISS4 MX70 L2 data
- [ ] Files are in `.tif` or `.jp2` format
- [ ] Files have `_t1` and `_t2` in names (or will rename)
- [ ] Copied Sentinel-2 to: `PS10_Input_Data\Sentinel2_L1C\`
- [ ] Copied LISS4 to: `PS10_Input_Data\LISS4_L2\`
- [ ] Ready to run: `python master_ps10_windows.py --run PS10_Input_Data`

---

## 🆘 IF YOU HAVE QUESTIONS

**Ask me:**
- "How do I rename my files?"
- "My files don't have _t1 and _t2, what should I do?"
- "Can I process just Sentinel-2 first?"
- "The system says no data found, help!"

---

## 📞 READY TO START?

**Simply tell me:**
1. ✅ "I've put the Sentinel-2 data in the folder"
2. ✅ "I've put the LISS4 data in the folder"
3. ✅ "Ready to run the full processing"

**And I'll guide you through the execution step-by-step!**

---

*System ready. Waiting for your data...*
