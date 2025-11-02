# ⚡ OCT 31 QUICK REFERENCE CARD

**DEADLINE: October 31, 2025 @ 16:00 (4:00 PM)**

---

## ⏰ **TIME BLOCKS**

| Time | Activity | Commands |
|------|----------|----------|
| **12:00** | Download data | Visit website, download 4 pairs (~10GB) |
| **13:00** | Start inference | `python changedetect/src/inference.py --input PS10_shortlisting_data --output PS10_final_predictions --model models/xboson_change_detector.pt` |
| **15:00** | Create package | `python prepare_ps10_final.py PS10_final_predictions models/xboson_change_detector.pt "XBoson AI"` |
| **15:30** | Validate | `python validate_ps10_compliance.py PS10_31-Oct-2025_XBosonAI.zip` |
| **15:45** | **SUBMIT** | Upload ZIP + Hash to portal |

---

## 📦 **SUBMISSION CHECKLIST**

- [ ] 4 × Change_Mask_Lat_Long.tif files
- [ ] 4 × Change_Mask_Lat_Long.shp (+ .shx, .dbf, .prj, .cpg)
- [ ] model_md5.txt file
- [ ] ZIP named: PS10_31-Oct-2025_XBosonAI.zip
- [ ] Submitted before 16:00
- [ ] Confirmation received

---

## 🚨 **EMERGENCY TROUBLESHOOTING**

**Inference too slow?**
```powershell
python changedetect/src/inference.py --batch_size 1 --input ... --output ...
```

**Out of memory?**
```powershell
python changedetect/src/inference.py --tile_size 256 --input ... --output ...
```

**Validation failed?**
- Check georeferencing: `python -c "import rasterio; print(rasterio.open('file.tif').crs)"`
- Check pixel values: `python -c "import rasterio; import numpy as np; print(np.unique(rasterio.open('file.tif').read()))"`

---

## 📊 **EXPECTED OUTPUTS**

```
PS10_final_predictions/
├── Location1_Lat_Long_change_mask.tif
├── Location1_Lat_Long_change_vectors.shp (.shx, .dbf, .prj, .cpg)
├── Location2_Lat_Long_change_mask.tif
├── Location2_Lat_Long_change_vectors.shp (.shx, .dbf, .prj, .cpg)
├── Location3_Lat_Long_change_mask.tif
├── Location3_Lat_Long_change_vectors.shp (.shx, .dbf, .prj, .cpg)
├── Location4_Lat_Long_change_mask.tif
└── Location4_Lat_Long_change_vectors.shp (.shx, .dbf, .prj, .cpg)
```

---

## ✅ **VALIDATION CHECKS**

1. **File Count:** 25 files total (4 TIF + 4×5 SHP components + 1 hash)
2. **File Names:** Format `Change_Mask_Lat_Long.*`
3. **Pixel Values:** Only 0 and 1 in TIF files
4. **Georeferencing:** All files have same CRS as input
5. **Shapefiles:** Can be opened without errors

---

## 💾 **MODEL HASH**

Your model hash (for submission):
```powershell
# Calculate if needed:
python -c "import hashlib; print(hashlib.md5(open('models/xboson_change_detector.pt', 'rb').read()).hexdigest())"
```

Should be in `model_md5.txt` in submission package.

---

## 🎯 **KEY POINTS**

1. **Only 4 hours total** - manage time carefully
2. **Submission portal opens at 12:00**
3. **Must submit before 16:00** - no extensions
4. **Top 15-20 selected** based on Jaccard Index
5. **Model hash verified** during offline evaluation

---

## 📞 **IF SOMETHING GOES WRONG**

1. Don't panic - you have backup time
2. Check error messages carefully
3. Try processing one image at a time
4. Document all issues
5. Submit whatever you have before deadline

---

**REMEMBER: Something submitted is better than perfect but late!**

**GOOD LUCK! 🚀**
