# PS-10 INPUT DATA STRUCTURE - OCTOBER 31, 2025

## Ready for Real Data Submission!

---

## Data Directory Structure

```
c:\Users\harsh\PS-10\PS10_Input_Data\
├── Sentinel2_L1C\           ← PUT SENTINEL-2 L1C DATA HERE
│   ├── T*.SAFE\             (original SAFE format, or extracted TIFs)
│   ├── Band_*.tif           (if extracted)
│   └── metadata.xml
│
├── LISS4_L2\                ← PUT LISS4 MX70 L2 DATA HERE
│   ├── *.tif                (GeoTIFF files)
│   ├── *.jp2                (JPEG2000 files)
│   ├── *.hdf                (HDF5 files if needed)
│   └── metadata.xml
│
└── Metadata\                ← PUT METADATA/INFO HERE
    ├── README.txt           (your notes on data)
    ├── data_info.json       (location, dates, etc.)
    └── processing_log.txt   (track what you do)
```

---

## How to Use This Structure

### Step 1: Put Your Data
Place your downloaded data files in the appropriate folders:
- **Sentinel-2 L1C** → `Sentinel2_L1C/` folder
- **LISS4 MX70 L2** → `LISS4_L2/` folder

### Step 2: Create Paired Images
The system expects image pairs:
```
Image_Time1_t1.tif    (older/first image)
Image_Time1_t2.tif    (newer/second image)

Image_Time2_t1.tif    (older/first image)
Image_Time2_t2.tif    (newer/second image)
```

### Step 3: Run the Submission
Once you have organized the data:
```powershell
cd C:\Users\harsh\PS-10
python master_ps10_windows.py --run PS10_Input_Data
```

---

## Data Format Requirements

### Sentinel-2 L1C
- **Format:** GeoTIFF (.tif) or original SAFE format
- **Bands:** 11 spectral bands (coastal, blue, green, red, NIR, SWIR, etc.)
- **Resolution:** 10m, 20m, or 60m (will be resampled)
- **CRS:** WGS84 (EPSG:4326) or projected
- **Naming:** `S2_LocationID_YYYYMMDD_t1.tif` (recommended)

### LISS4 MX70 L2
- **Format:** GeoTIFF (.tif) or JPEG2000 (.jp2)
- **Bands:** 4 multispectral bands (Pan, SWIR, NIR, Red, Green, Blue)
- **Resolution:** 5.8m (standard)
- **CRS:** WGS84 (EPSG:4326) or projected  
- **Naming:** `LISS_LocationID_YYYYMMDD_t2.tif` (recommended)

---

## File Organization Checklist

### Before Running Submission:

- [ ] Sentinel-2 L1C files downloaded to `Sentinel2_L1C/`
- [ ] LISS4 MX70 L2 files downloaded to `LISS4_L2/`
- [ ] Files are in GeoTIFF format (.tif) or can be converted
- [ ] Image pairs are properly named (with _t1 and _t2 suffixes)
- [ ] All files have proper georeference (CRS info)
- [ ] All files have same spatial extent or overlapping regions
- [ ] Metadata/info files in `Metadata/` folder

---

## Expected Processing Steps

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. VERIFICATION (10-15 seconds)                                │
│     ├─ Check data files exist                                   │
│     ├─ Verify georeference information                          │
│     ├─ Check pixel values and bands                             │
│     └─ Validate image pairs                                     │
│                                                                 │
│  2. PRE-PROCESSING (~depends on data size)                      │
│     ├─ Resample to common resolution                            │
│     ├─ Align georeferencing                                     │
│     ├─ Extract overlapping regions                              │
│     └─ Normalize to standard format                             │
│                                                                 │
│  3. INFERENCE (depends on image count)                          │
│     ├─ Load XBoson change detector model                        │
│     ├─ Process each image pair                                  │
│     ├─ Generate change detection masks                          │
│     └─ Create vector polygons (shapefiles)                      │
│                                                                 │
│  4. POST-PROCESSING & PACKAGING (5-15 seconds)                 │
│     ├─ Format filenames to PS-10 standard                       │
│     ├─ Extract coordinates from GeoTIFF                         │
│     ├─ Create output shapefiles                                 │
│     ├─ Verify all components present                            │
│     └─ Create submission ZIP file                               │
│                                                                 │
│  5. VALIDATION (5-10 seconds)                                   │
│     ├─ Verify file format compliance                            │
│     ├─ Check shapefile integrity                                │
│     ├─ Validate pixel values                                    │
│     └─ Create submission report                                 │
│                                                                 │
│  6. OUTPUT                                                       │
│     └─ PS10_31-Oct-2025_XBosonAI.zip (ready for upload)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Commands to Execute

### Prepare Data (Manual - you do this)
```powershell
# Copy your data to the right folders
# Sentinel-2: C:\Users\harsh\PS-10\PS10_Input_Data\Sentinel2_L1C\
# LISS4:      C:\Users\harsh\PS-10\PS10_Input_Data\LISS4_L2\
```

### Run Full Submission (Automated)
```powershell
cd C:\Users\harsh\PS-10
python master_ps10_windows.py --run PS10_Input_Data
```

### If You Want to Skip Tests (Faster)
```powershell
python master_ps10_windows.py --quick PS10_Input_Data
```

### Test First (Optional)
```powershell
python master_ps10_windows.py --test
```

---

## Output Will Be

```
C:\Users\harsh\PS-10\
├── ps10_predictions\              (raw inference output)
├── ps10_predictions_formatted\    (formatted to PS-10 standard)
├── PS10_31-Oct-2025_XBosonAI.zip  ← SUBMIT THIS FILE
└── ps10_submission_report.txt     (validation details)
```

---

## Data Upload Instructions

### For Sentinel-2 L1C
1. Download from: ESA Copernicus Open Access Hub
2. Extract SAFE folders to `Sentinel2_L1C/`
3. Or convert to GeoTIFF and place there

### For LISS4 MX70 L2
1. Download Level-2 processed data
2. Extract to `LISS4_L2/` folder
3. Should be in GeoTIFF or JP2 format

---

## File Naming Convention (Recommended)

For consistency, name your paired images:
```
S2_location_20240101_t1.tif     (Sentinel-2, first acquisition)
S2_location_20240101_t2.tif     (Sentinel-2, second acquisition)

LISS_location_20240115_t1.tif   (LISS4, first acquisition)
LISS_location_20240115_t2.tif   (LISS4, second acquisition)

OR use any naming as long as:
- Images are in the same directory
- Times are clear (t1 = earlier, t2 = later)
- Extensions are .tif or .jp2
```

---

## Troubleshooting

### "No image pairs found"
- Ensure files end with `_t1.tif` and `_t2.tif`
- Check files are in correct folder
- Verify both files exist for each pair

### "Georeferencing error"
- Ensure files have CRS information
- Convert to GeoTIFF if needed
- Check EPSG codes match or are compatible

### "Band mismatch"
- Sentinel-2 should have 11 bands
- LISS4 should have 4 bands
- Resampling will be handled automatically

### "Insufficient disk space"
- Need at least 10+ GB available
- Check: `Get-Volume`
- Clean up temp files if needed

---

## Questions?

Check these files for more info:
- `WINDOWS_SUBMISSION_GUIDE.md` - Windows specific help
- `OCTOBER_31_FINAL_CHECKLIST.md` - Full checklist
- `test_report.txt` - Last test results

---

## READY TO START?

1. Put your Sentinel-2 L1C data in: `PS10_Input_Data/Sentinel2_L1C/`
2. Put your LISS4 MX70 L2 data in: `PS10_Input_Data/LISS4_L2/`
3. Organize as image pairs (t1 = first, t2 = second)
4. Run: `python master_ps10_windows.py --run PS10_Input_Data`
5. Upload the generated ZIP file

**System is ready. Let me know once data is placed and I'll guide you through execution!**
