# 📊 PS-10 FORMAT FIX - VISUAL GUIDE

## 🔴 THE PROBLEM VISUALIZED

### What Happened in Weeks 3 & 4:

```
┌─────────────────────────────────────────┐
│ Your Submission (Week 3 & 4)            │
│                                         │
│  PS10_DD-MMM-YYYY_XBosonAI.zip          │
│  ├── 0_0_change_mask.tif          ❌   │
│  ├── 0_0_change_vectors.shp       ❌   │
│  ├── Change_Mask_22_28.tif        ❌   │ Array indices!
│  ├── Change_Mask_22_28.shp        ❌   │
│  ├── Change_Mask_23_32.tif        ❌   │
│  └── model_md5.txt                ✓    │
└─────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│ PS-10 Portal Validation                 │
│                                         │
│ 1. Extract ZIP                     ✓    │
│ 2. Read filename: "0_0_"           ✓    │
│ 3. Parse coordinates               ✗    │
│    Looking for: lat, long               │
│    Found: 0, 0 (array indices)          │
│ 4. Validate format                 ✗    │
│ 5. Score submission                ✗    │
└─────────────────────────────────────────┘
          ↓
      REJECTED ❌
      
Feedback: "Incorrect format"
```

---

## 🟢 THE SOLUTION

### What Portal Expects:

```
┌──────────────────────────────────────────────┐
│ Correct Submission (October 31)              │
│                                              │
│  PS10_31-Oct-2025_XBosonAI.zip               │
│  ├── Change_Mask_28.1740_77.6126.tif  ✅    │
│  ├── Change_Mask_28.1740_77.6126.shp  ✅    │ DECIMAL
│  ├── Change_Mask_28.1740_77.6126.shx  ✅    │ COORDINATES
│  ├── Change_Mask_28.1740_77.6126.dbf  ✅    │
│  ├── Change_Mask_28.1740_77.6126.prj  ✅    │
│  ├── Change_Mask_28.1740_77.6126.cpg  ✅    │
│  ├── Change_Mask_23.7380_84.2129.tif  ✅    │
│  ├── Change_Mask_23.7380_84.2129.shp  ✅    │ NEXT
│  └── model_md5.txt                     ✅    │ LOCATION
└──────────────────────────────────────────────┘
          ↓
┌──────────────────────────────────────────────┐
│ PS-10 Portal Validation                      │
│                                              │
│ 1. Extract ZIP                          ✓    │
│ 2. Read filename: "28.1740_77.6126"     ✓    │
│ 3. Parse coordinates                    ✓    │
│    Found: lat=28.1740, long=77.6126     ✓    │
│ 4. Validate format                      ✓    │
│    Decimal coordinates ✓                     │
│    Valid latitude/longitude ✓                │
│ 5. Score submission                     ✓    │
└──────────────────────────────────────────────┘
          ↓
      ACCEPTED ✅
      
Status: "Processing submission"
```

---

## 🔄 HOW COORDINATES ARE EXTRACTED

### From GeoTIFF Image to Filename:

```
GeoTIFF File: 0_0_change_mask.tif
│
├─ Geographic Metadata (stored in file)
│  ├─ Bounds: Left=77.50, Right=77.70, Bottom=28.10, Top=28.30
│  └─ CRS: EPSG:4326 (WGS84)
│
├─ Calculate Center
│  ├─ Latitude = (28.10 + 28.30) / 2 = 28.20°
│  └─ Longitude = (77.50 + 77.70) / 2 = 77.60°
│
└─ Create Filename
   └─ Change_Mask_28.20_77.60.tif ✓
```

### Python Code:
```python
import rasterio

with rasterio.open('0_0_change_mask.tif') as src:
    bounds = src.bounds
    
    # Calculate center
    lat = (bounds.bottom + bounds.top) / 2
    lon = (bounds.left + bounds.right) / 2
    
    # New filename
    new_name = f"Change_Mask_{lat}_{lon}.tif"
    # Result: Change_Mask_28.2_77.6.tif
```

---

## 📈 WORKFLOW DIAGRAM

### Your Processing Pipeline on October 31:

```
┌─────────────────────┐
│  12:00 PM: Start    │
│  Download data      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│ 12:15 PM: Run Automation        │
│ Command:                        │
│ python master_ps10_fixed.py ... │
└──────────┬──────────────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────────┐  ┌──────────────┐
│ Inference  │  │ Format Fix   │
│ (15 min)   │  │ (Automatic)  │
└────┬───────┘  └───────┬──────┘
     │                  │
     └────────┬─────────┘
              │
              ▼
     ┌────────────────┐
     │ Create ZIP     │
     │ With Coords    │
     └────────┬───────┘
              │
              ▼
     ┌────────────────────────┐
     │ Validate              │
     │ ✓ Filenames           │
     │ ✓ Pixel values        │
     │ ✓ Shapefiles         │
     │ ✓ MD5 hash           │
     └────────┬───────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ 15:30 PM: SUBMIT!   │
   │ ZIP Ready to Upload │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ 16:00 PM: DEADLINE! │
   │     SUCCESS! ✅      │
   └──────────────────────┘
```

---

## 📋 FILE NAMING COMPARISON

### Side-by-Side Comparison:

```
Location: Northern India (Delhi-NCR)
Geographic Bounds:
  Latitude: 28.1°N to 28.3°N
  Longitude: 77.5°E to 77.7°E

┌──────────────────────┬──────────────────────────────┐
│ WRONG (Rejected ❌) │ CORRECT (Accepted ✅)       │
├──────────────────────┼──────────────────────────────┤
│ 0_0_change_mask.tif  │ Change_Mask_28.20_77.60.tif │
│ 0_0_change_vectors   │ Change_Mask_28.20_77.60.shp │
│                      │ Change_Mask_28.20_77.60.shx │
│ Change_Mask_22_28    │ Change_Mask_28.20_77.60.dbf │
│ (integers!)          │ Change_Mask_28.20_77.60.prj │
│                      │ Change_Mask_28.20_77.60.cpg │
│                      │ (complete shapefile set!)    │
└──────────────────────┴──────────────────────────────┘
```

---

## 🎯 VALIDATION FLOWCHART

### How Portal Checks Your Submission:

```
           Receive ZIP
              │
              ▼
        Extract Files
              │
              ▼
    ┌────────────────────┐
    │ Read Filename:     │
    │ "Change_Mask_X_Y"  │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Parse X, Y         │
    │ Are they decimal?  │
    └────────┬───────────┘
             │
        ┌────┴────┐
        │          │
    YES ▼         ▼ NO
       ✓         ✗ → REJECT
       │
       ▼
    Are X, Y valid
    latitude/longitude?
    (-90 to 90, -180 to 180)
       │
    ┌──┴──┐
    │     │
YES ▼     ▼ NO
   ✓     ✗ → REJECT
   │
   ▼
Check TIF file
Pixel values 0 or 1?
   │
┌──┴──┐
│     │
YES▼  ▼ NO
  ✓  ✗ → REJECT
  │
  ▼
Check Shapefiles
All 5 components?
  │
┌──┴──┐
│     │
YES▼  ▼ NO
  ✓  ✗ → REJECT
  │
  ▼
✅ ACCEPTED
Score Submission
```

---

## 📞 SUPPORT REFERENCE

### If Filenames Are Still Wrong:

```
Problem: Files not renamed
Solution 1: Run format fix manually
  python fix_submission_format.py input_dir model.pt "Team"

Solution 2: Check GeoTIFF metadata
  python -c "import rasterio; 
             src=rasterio.open('file.tif'); 
             print(src.bounds)"

Solution 3: Use fallback naming
  Files still named correctly even if metadata extraction fails
```

### If Coordinates Don't Make Sense:

```
Expected Ranges:
  Latitude: -90 to +90 degrees (N-S)
  Longitude: -180 to +180 degrees (E-W)

India Examples:
  Delhi: 28.6139° N, 77.2090° E
  Mumbai: 19.0760° N, 72.8777° E
  Bangalore: 12.9716° N, 77.5946° E

Format in filename: Change_Mask_28.6139_77.2090.tif ✓
```

---

## 🚀 ONE-LINE SUMMARY

**WRONG:** `0_0_change_mask.tif` (array index, rejected ❌)  
**RIGHT:** `Change_Mask_28.6139_77.2090.tif` (geographic coordinates, accepted ✅)  
**AUTOMATION:** `python master_ps10_fixed.py --run PS10_shortlisting_data` (does everything!)

---

**Ready? Let's go! October 31 will be YOUR day! 🎯**
