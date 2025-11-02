# PS-10 FINAL SUBMISSION CHECKLIST
## Due: October 31, 2025 @ 16:00 Hrs (4:00 PM)

---

## 🎯 **SUBMISSION WINDOW: 4 HOURS ONLY**
- Dataset Release: 12:00 PM (Noon)
- Submission Deadline: 16:00 (4:00 PM)
- **You have EXACTLY 4 hours to process and submit!**

---

## ✅ **PRE-SUBMISSION PREPARATION (Do NOW)**

### 1. **Test Your Complete Pipeline**
```powershell
# Test with mock data to ensure everything works
python prepare_ps10_final.py predictions_threshold_0.1 models\xboson_change_detector.pt "XBoson AI"
```

### 2. **Verify Model File Location**
- [ ] Model file exists: `c:\Users\harsh\PS-10\models\xboson_change_detector.pt`
- [ ] Model file size: ____ MB
- [ ] Model loads without errors

### 3. **Test MD5 Hash Generation**
```powershell
# Verify hash calculation works
python -c "import hashlib; print(hashlib.md5(open('models/xboson_change_detector.pt', 'rb').read()).hexdigest())"
```
- [ ] Hash generated successfully: ________________

### 4. **Download Tools Ready**
- [ ] Bhoonidhi account active
- [ ] Copernicus account active  
- [ ] Fast internet connection verified
- [ ] ~10GB disk space available

### 5. **Inference Speed Test**
```powershell
# Time your inference on sample data
Measure-Command { python changedetect/src/inference.py --input test_data --output test_output }
```
- [ ] Time per image pair: ____ minutes
- [ ] Total estimated time for 4 pairs: ____ minutes

---

## 📥 **ON OCT 31 @ 12:00 PM - IMMEDIATE ACTIONS**

### **HOUR 1 (12:00 - 13:00): DOWNLOAD DATA**

1. **Navigate to website immediately:**
   - URL: [PS-10 Competition Website]
   - Check "Shortlisting Dataset" section

2. **Download 4 image pairs:**
   - [ ] LISS-4 Pair 1 - Location: _________, Size: ____GB
   - [ ] LISS-4 Pair 2 - Location: _________, Size: ____GB
   - [ ] Sentinel-2 Pair 3 - Location: _________, Size: ____GB
   - [ ] Sentinel-2 Pair 4 - Location: _________, Size: ____GB

3. **Organize downloaded data:**
```powershell
mkdir PS10_shortlisting_data
# Move all downloaded files to this folder
# Expected structure:
# PS10_shortlisting_data/
#   ├── pair1_t1.tif (or .jp2)
#   ├── pair1_t2.tif
#   ├── pair2_t1.tif
#   ├── pair2_t2.tif
#   ├── pair3_t1.tif
#   ├── pair3_t2.tif
#   ├── pair4_t1.tif
#   └── pair4_t2.tif
```

---

### **HOUR 2-3 (13:00 - 15:00): RUN INFERENCE**

4. **Run Change Detection:**
```powershell
# Activate environment if needed
# conda activate changedetect

# Run inference on all 4 pairs
python changedetect/src/inference.py `
    --input PS10_shortlisting_data `
    --output PS10_final_predictions `
    --model models/xboson_change_detector.pt `
    --batch_size 4
```

5. **Monitor Progress:**
   - [ ] Pair 1 processing started: __:__ 
   - [ ] Pair 1 completed: __:__
   - [ ] Pair 2 completed: __:__
   - [ ] Pair 3 completed: __:__
   - [ ] Pair 4 completed: __:__

6. **Verify Outputs Generated:**
   - [ ] All TIF files created (4 files)
   - [ ] All Shapefiles complete (4 × 5 files = 20 files)
   - [ ] All files georeferenced

---

### **HOUR 3-4 (15:00 - 16:00): CREATE & SUBMIT**

7. **Create Submission Package:**
```powershell
python prepare_ps10_final.py `
    PS10_final_predictions `
    models/xboson_change_detector.pt `
    "XBoson AI"
```

8. **Verify Package Contents:**
```powershell
# Extract and check
Expand-Archive -Path "PS10_31-Oct-2025_XBosonAI.zip" -DestinationPath temp_check
dir temp_check
```

Expected contents:
- [ ] 4 × Change_Mask_Lat_Long.tif files
- [ ] 4 × 5 shapefile components (.shp, .shx, .dbf, .prj, .cpg)
- [ ] model_md5.txt
- [ ] Total: 25 files

9. **Final Validation:**
```powershell
python validate_ps10_compliance.py temp_check
```
- [ ] All validations passed
- [ ] No errors reported
- [ ] File naming correct
- [ ] Pixel values 0/1 confirmed
- [ ] Georeferencing intact

10. **Submit Before 16:00:**
    - [ ] Navigate to submission portal
    - [ ] Upload ZIP file: PS10_31-Oct-2025_XBosonAI.zip
    - [ ] Upload model hash from model_md5.txt
    - [ ] Verify submission received
    - [ ] Save confirmation screenshot/email
    - [ ] **SUBMISSION TIME: __:__ (MUST BE BEFORE 16:00!)**

---

## 🔧 **TROUBLESHOOTING QUICK FIXES**

### If Inference Fails:
```powershell
# Try with smaller batch size
python changedetect/src/inference.py --batch_size 1 ...

# Or process one pair at a time
python changedetect/src/inference.py --input pair1 --output out1 ...
```

### If Out of Memory:
- Close all other applications
- Reduce tile size in inference config
- Process images one at a time

### If File Format Issues:
```powershell
# Manually verify with
python -c "import rasterio; print(rasterio.open('file.tif').crs)"
```

### If Submission Portal Down:
- Have backup submission method ready (email to organizers?)
- Document all attempts
- Take screenshots

---

## 📞 **EMERGENCY CONTACTS**

- Organizer Email: _______________
- Technical Support: _______________
- Your Team Lead: _______________

---

## ⏰ **TIME MANAGEMENT**

| Time Slot | Duration | Activity | Status |
|-----------|----------|----------|---------|
| 12:00-13:00 | 60 min | Download data | ⏳ |
| 13:00-14:30 | 90 min | Run inference | ⏳ |
| 14:30-15:30 | 60 min | Create package | ⏳ |
| 15:30-15:55 | 25 min | Final validation | ⏳ |
| 15:55-16:00 | 5 min | **SUBMIT!** | ⏳ |

**BUFFER TIME:** 5 minutes - DO NOT USE unless absolutely necessary!

---

## 🎯 **SUCCESS CRITERIA**

- [ ] Submission uploaded before 16:00 PM
- [ ] Confirmation received from portal
- [ ] All 4 image pairs processed
- [ ] Package format correct
- [ ] Model hash submitted
- [ ] No validation errors

---

## 💡 **FINAL TIPS**

1. **Start Early:** Be ready at 11:55 AM
2. **Don't Panic:** You've tested everything
3. **Monitor Time:** Set alarms for each phase
4. **Have Backup Plan:** Know alternative submission method
5. **Document Everything:** Take screenshots of each step

---

**REMEMBER: You have prepared well. Trust your system. Execute the plan.**

**GOOD LUCK! 🚀**
