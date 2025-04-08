# Geospatial Pattern Matching from Video using Satellite-Based Vector Templates

**Author:** Camilo Eduardo Fuentealba Vásquez
**PhD Candidate in Engineering Sciences, University of Chile**

---

## Project Status: Beta Phase

This project is currently in an **exploratory and development stage**. Various **contour similarity models** and geometric alignment techniques are being tested to build a robust system for geospatial recognition in video data.

Although the approach is still under evaluation, the code implements a **systematic detection model** based on geometric transformations and template matching.
**This is a functional prototype under continuous improvement.**

---

## Project Description

This script implements a geospatial recognition system for video sequences using vector geometries extracted from satellite imagery (GeoJSON format). The objective is to align geographical structures (such as coastlines, natural features, or infrastructure) with video frames, applying geometric transformations and normalized correlation evaluation.

This tool may serve as a foundation for satellite-based navigation, real-time geolocation, and remote sensing systems onboard platforms such as CubeSats.

---

## Key Features

- Processes vector geometries (`.geojson`) and resizes them to match video frame dimensions.
- Applies **geometric transformations** (translation and rotation) for pattern alignment.
- Detects matches using **OpenCV Template Matching**.
- Uses **normalized correlation coefficient** (`cv2.TM_CCOEFF_NORMED`) for evaluation.
- Displays results with graphical overlay showing vector alignment.

---

## Requirements

- Python 3.8 or higher
- `opencv-python`
- `numpy`
- `geopandas`
- `shapely`

Install via pip:

```bash
pip install opencv-python numpy geopandas shapely
```

---

## Expected Files

Place the following files in the **same directory** as `main.py`:

- `cf_B8A_douglas_r_areas.geojson` → reference vector file
- `47.mp4` → input video for analysis

> If uploading to GitHub, consider excluding large video files using `.gitignore`.

---

## Running the Script

From the terminal:

```bash
python main.py
```

Press `Q` to close the video window.

---

## Configurable Parameters

Inside the script (`main.py`), the following parameters can be adjusted:

```python
output_width, output_height = 320, 180
shrink_factor1 = 0.23
shrink_factor2 = 0.67
template_threshold = 0.60
```

---

## Output

For each frame, the system outputs:

- Match score (between 0 and 1)
- Match status (MATCH or NO MATCH)
- Applied translation (dx, dy)
- Applied rotation (angle)
- A rectangle showing the best match area

All results are displayed both in the terminal and visually on the video frame.

---

## License and Usage

This project is an original work by **Camilo Eduardo Fuentealba Vásquez**.
**Reproduction or commercial use is strictly prohibited without explicit permission.**
To collaborate or cite this work, please contact the author directly.

---
