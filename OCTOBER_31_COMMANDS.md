# 🚀 PS-10 QUICK START - OCTOBER 31 COMMANDS

## ⚡ THE ISSUE (In 30 Seconds)
Your submissions failed because filenames were WRONG.
- ❌ You sent: `0_0_change_mask.tif` (array indices)
- ✅ Portal wants: `Change_Mask_28.1740_77.6126.tif` (coordinates from GeoTIFF)

## ✅ THE FIX (In 30 Seconds)
New script automatically extracts coordinates from image metadata and renames files correctly!

---

## 📅 OCTOBER 31 - EXACT COMMANDS TO RUN

### 12:00 PM - Download data (manual step)
```
1. Go to PS-10 portal
2. Download shortlisting dataset
3. Save to folder: PS10_shortlisting_data/
```

### 12:15 PM - Run automated workflow
```powershell
python master_ps10_fixed.py --run PS10_shortlisting_data
```

That's it! This command:
- ✅ Runs inference (15-30 min)
- ✅ Fixes filenames (adds coordinates)
- ✅ Creates ZIP package
- ✅ Validates everything

### 14:00 PM - Check results
```powershell
ls PS10_31-Oct-2025_XBosonAI.zip

# Should exist with all files correctly named
```

### 15:30 PM - Submit!
1. Go to portal
2. Upload: `PS10_31-Oct-2025_XBosonAI.zip`
3. Upload hash from inside ZIP
4. Submit before 16:00

---

## 🆘 IF SOMETHING GOES WRONG

### No time for inference?
```powershell
# Use quick mode (skip setup tests)
python master_ps10_fixed.py --quick PS10_shortlisting_data
```

### Want to debug before Oct 31?
```powershell
# Test your setup now
python master_ps10_fixed.py --test

# Should show all GREEN ✓
```

### Manual format correction only?
```powershell
python fix_submission_format.py PS10_predictions models/xboson_change_detector.pt "XBoson AI"
```

---

## ✅ SUCCESS INDICATORS

After running the command, you should see:

```
✓ Inference completed
✓ Filenames corrected to: Change_Mask_LAT_LONG.tif
✓ Shapefiles copied with correct names
✓ ZIP package created: PS10_31-Oct-2025_XBosonAI.zip
✓ Model hash: [hex string]
✓ Validation passed
```

---

## 📂 EXPECTED FILE STRUCTURE AFTER RUNNING

```
PS10_31-Oct-2025_XBosonAI.zip
├── Change_Mask_28.1740_77.6126.tif        ✅ (renamed from 0_0_change_mask.tif)
├── Change_Mask_28.1740_77.6126.shp        ✅ (renamed from 0_0_change_vectors.shp)
├── Change_Mask_28.1740_77.6126.shx        ✅ (all shapefile components)
├── Change_Mask_28.1740_77.6126.dbf        ✅
├── Change_Mask_28.1740_77.6126.prj        ✅
├── Change_Mask_28.1740_77.6126.cpg        ✅
├── Change_Mask_23.7380_84.2129.tif        ✅ (next location)
├── Change_Mask_23.7380_84.2129.shp        ✅
├── ... (more locations)
└── model_md5.txt                           ✅ (inside ZIP)
```

---

## 🎯 THE FORMAT THAT WORKS

Each location needs these 6 files (or at minimum .tif + .shp + .shx + .dbf + .prj):
```
Change_Mask_[LAT]_[LONG].tif      ← Change detection raster
Change_Mask_[LAT]_[LONG].shp      ← Vector geometry
Change_Mask_[LAT]_[LONG].shx      ← Shapefile index
Change_Mask_[LAT]_[LONG].dbf      ← Attribute data
Change_Mask_[LAT]_[LONG].prj      ← Projection info
Change_Mask_[LAT]_[LONG].cpg      ← Code page (optional but safe)
```

Where `[LAT]` and `[LONG]` are DECIMAL coordinates from the image!

---

## 🔍 HOW PORTAL VALIDATES

Portal automatically checks:
1. ✅ Filename format: `Change_Mask_LAT_LONG.*` pattern
2. ✅ Coordinates: Valid decimal latitude/longitude
3. ✅ TIF values: Only 0 or 1 pixels
4. ✅ Shapefiles: All 5 components present
5. ✅ Model hash: Matches submitted model

All must pass!

---

## ⏰ TIMELINE SUMMARY

| Time | Action | Command |
|------|--------|---------|
| 11:55 | Final check | Review this document |
| 12:00 | Download data | Manual: portal → download |
| 12:15 | **RUN WORKFLOW** | `python master_ps10_fixed.py --run PS10_shortlisting_data` |
| 13:45 | Check results | `ls PS10_31-Oct-2025_XBosonAI.zip` |
| 14:00 | Validate | `python validate_ps10_compliance.py PS10_31-Oct-2025_XBosonAI.zip` |
| 15:30 | **SUBMIT!** | Portal: Upload ZIP + hash |
| 15:59 | **DONE!** | ✅ Submission complete |

---

## 📝 LAST REMINDERS

1. **The FORMAT FIX is automatic** - Don't manually rename!
2. **One command runs everything** - `master_ps10_fixed.py --run INPUT_DIR`
3. **Coordinates come from image metadata** - No manual coordinate input needed
4. **Submit BEFORE 16:00** - Don't wait until last minute!
5. **Check filenames in ZIP** - Must be `Change_Mask_LAT_LONG.*` format

---

**Good luck! You've got this! 🚀**

*When you submit, you'll finally have the correct format that the portal expects!*
