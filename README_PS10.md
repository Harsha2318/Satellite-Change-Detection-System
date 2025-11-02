# PS-10: Man-made Change Detection Submission Tools

This repository contains specialized tools for preparing and validating submissions for the PS-10 Man-made Change Detection challenge.

## Overview

PS-10 requires detection of man-made changes in satellite imagery pairs. These tools ensure your submission meets the specific formatting requirements.

## Requirements

The PS-10 requirements specify:

1. **Output Formats**:
   - Raster change masks in GeoTIFF format (`Change_Mask_Lat_Long.tif`)
   - Vector files in Shapefile format (`Change_Mask_Lat_Long.shp`)
   - Raster values: 1 for change, 0 for no change
   - All outputs must be georeferenced

2. **Submission Package**:
   - Named `PS10_[DD-MMM-YYYY]_[Startup/Group Name without Space].zip`
   - Must include MD5 hash of model in `model_md5.txt`

## Setup

Install the required dependencies:

```bash
pip install rasterio geopandas numpy
```

## Scripts

### 1. create_ps10_submission.py

Creates a compliant submission package from your prediction outputs:

```bash
python create_ps10_submission.py <predictions_dir> <model_path> [startup_name]
```

Arguments:
- `predictions_dir`: Directory containing your change detection results
- `model_path`: Path to your model file (for MD5 hash calculation)
- `startup_name`: Your startup/group name (without spaces, default: XBoson)

Example:
```bash
python create_ps10_submission.py predictions_threshold_0.1 models/xboson_change_detector.pt XBoson
```

### 2. verify_ps10_submission.py

Verifies that a submission package meets all PS-10 requirements:

```bash
python verify_ps10_submission.py <submission_directory_or_zip>
```

Arguments:
- `submission_directory_or_zip`: Path to submission directory or ZIP file

Example:
```bash
python verify_ps10_submission.py PS10_31-Oct-2025_XBoson.zip
```

### 3. format_ps10_outputs.py

Converts your existing prediction outputs to PS-10 format:

```bash
python format_ps10_outputs.py <input_dir> <output_dir> [--force]
```

Arguments:
- `input_dir`: Directory containing your current prediction outputs
- `output_dir`: Directory where formatted outputs will be saved
- `--force`: Overwrite output directory if it exists

Example:
```bash
python format_ps10_outputs.py predictions_run1 ps10_formatted_outputs
```

### 4. validate_ps10_compliance.py

Checks if your output files meet PS-10 specifications:

```bash
python validate_ps10_compliance.py <output_directory>
```

Example:
```bash
python validate_ps10_compliance.py ps10_formatted_outputs
```

## PS-10 Terrain Types

PS-10 requires change detection across multiple terrain types:

| No | Terrain | Latitude (N) | Longitude (E) |
|----|---------|--------------|--------------|
| 1  | Snow    | 34.0531      | 74.3909      |
| 2  | Plain   | 13.3143      | 77.6157      |
| 3  | Hill    | 31.2834      | 76.7904      |
| 4  | Desert  | 26.9027      | 70.9543      |
| 5  | Forest  | 23.7380      | 84.2129      |
| 6  | Urban   | 28.1740      | 77.6126      |

## Submission Workflow

1. **Generate predictions** using our change detection model
2. **Format outputs** to meet PS-10 requirements
3. **Verify compliance** with PS-10 specifications
4. **Create submission package** with proper naming
5. **Submit** the ZIP file and model hash

## Important Dates

- Shortlisting Dataset Release: 31st October 2025 @ 1200 Hrs
- Submission Deadline: 31st October 2025 @ 1600 Hrs

## References

- [PS-10 Documentation](./PS%2010.pdf)

## Contact

For questions or issues, please contact the XBoson AI team.