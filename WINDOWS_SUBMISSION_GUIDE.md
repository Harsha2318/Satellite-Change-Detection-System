# PS-10 Windows-Compatible Submission Guide

## Status: ✓ READY FOR SUBMISSION

All tests are passing! Your system is ready for the October 31 submission deadline.

---

## Quick Start Commands

### 1. Run Setup Tests (Optional - already passed)
```powershell
python master_ps10_windows.py --test
```

### 2. Run Full Workflow (Recommended for Oct 31)
```powershell
python master_ps10_windows.py --run PS10_shortlisting_data
```

This will:
- Run comprehensive tests
- Execute inference on input data
- Format all outputs correctly
- Create submission ZIP package
- Validate everything

### 3. Quick Mode (Skip Tests)
```powershell
python master_ps10_windows.py --quick PS10_shortlisting_data
```

---

## Key Files

| File | Purpose |
|------|---------|
| `master_ps10_windows.py` | Main orchestrator script (USE THIS) |
| `test_complete_workflow_windows.py` | Windows-compatible test suite |
| `prepare_ps10_final_fixed.py` | Submission package creator |
| `validate_ps10_compliance.py` | Validates output format |
| `oct31_rapid_inference.py` | Runs inference on image pairs |

---

## Environment Setup (Already Complete)

✓ pandas - reinstalled (fixed import issue)
✓ geopandas - verified working
✓ rasterio - installed
✓ torch - installed  
✓ numpy - installed
✓ shapely - installed

### If You Need to Reinstall:
```powershell
pip install pandas==2.3.3 --no-deps --ignore-installed
pip install geopandas
```

---

## Expected Output

When running the full workflow, you'll get:

1. **Test Results** (10/10 passing)
2. **Inference Progress** (processing image pairs)
3. **Format Correction** (renaming files to PS-10 standard)
4. **Submission Package** (`PS10_DD-MMM-YYYY_XBosonAI.zip`)
5. **Validation Report** (verifying package contents)

---

## Submission Package Contents

The ZIP file will contain:
- `Change_Mask_*.tif` - 16 change detection masks
- `Change_Mask_*.shp` - Shapefile components (shp, shx, dbf, prj, cpg)
- `xboson_change_detector.pt` - Model file
- `model_md5.txt` - Model hash

---

## Windows-Specific Fixes Applied

1. **Removed Unicode/Emoji**: All output uses ASCII-safe characters (`[+]`, `[!]`, `[i]`, `[*]`)
2. **Fixed Encoding**: UTF-8 environment variable set automatically
3. **Path Handling**: All paths use Python's Path objects for cross-platform compatibility
4. **Module Imports**: Fixed geopandas/pandas circular dependency issue
5. **Subprocess Output**: Proper encoding and error handling for Windows subprocess calls

---

## Troubleshooting

### If tests fail:
1. Reinstall pandas: `pip install pandas==2.3.3 --no-deps --ignore-installed`
2. Reinstall geopandas: `pip install geopandas --upgrade`
3. Run tests again: `python master_ps10_windows.py --test`

### If submission package creation fails:
- Run with verbose output: `python prepare_ps10_final_fixed.py predictions_final models/xboson_change_detector.pt XBoson`

### If encoding errors appear:
- Restart PowerShell terminal
- Run: `$env:PYTHONIOENCODING = "utf-8"`
- Then run your command again

---

## Oct 31 Deadline Timeline

| Time | Action |
|------|--------|
| Before 23:59 | Download `PS10_shortlisting_data` |
| 23:59 - 24:00 | Run: `python master_ps10_windows.py --run PS10_shortlisting_data` |
| Before 23:59:59 | Upload the generated ZIP file |

---

## Support

- Check `test_report.txt` for detailed test results
- View submission logs by running with `2>&1` redirection
- All scripts output progress messages with timestamps `[HH:MM:SS]`

---

**Last Updated:** October 30, 2025  
**System:** Windows PowerShell v5.1  
**Status:** ✓ READY FOR SUBMISSION
