import os
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.affinity import scale, translate, rotate

# === RUTAS RELATIVAS DESDE LA CARPETA DEL SCRIPT ===
base_dir = os.path.dirname(os.path.abspath(__file__))  # Ruta de la carpeta del script

ruta_geojson = os.path.join(base_dir, "cf_B8A_douglas_r_areas.geojson")
video_path = os.path.join(base_dir, "47.mp4")


output_width, output_height = 320, 180
shrink_factor1 = 0.23
shrink_factor2 = 0.67
template_threshold = 0.60

# Definimos combinaciones de desplazamientos y rotaciones
offsets = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
rotaciones = list(range(0, 360, 10))
combinaciones = [(dx, dy, angle) for (dx, dy) in offsets for angle in rotaciones]

# === FUNCIONES ===
def draw_geometry_to_mask(gdf, width, height):
    canvas = np.zeros((height, width), dtype=np.uint8)
    for geom in gdf.geometry:
        if geom.is_empty or geom is None:
            continue
        if isinstance(geom, (Polygon, MultiPolygon)):
            polys = [geom] if isinstance(geom, Polygon) else geom.geoms
            for poly in polys:
                coords = np.array(poly.exterior.coords)
                if len(coords) > 1:
                    coords = coords.astype(np.int32)
                    cv2.polylines(canvas, [coords], isClosed=True, color=255, thickness=1)
        elif isinstance(geom, LineString):
            coords = np.array(geom.coords)
            if len(coords) > 1:
                coords = coords.astype(np.int32)
                cv2.polylines(canvas, [coords], isClosed=False, color=255, thickness=1)
    return canvas

def procesar_geojson(ruta):
    gdf = gpd.read_file(ruta)
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon', 'LineString'])].copy()
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox_width = maxx - minx
    bbox_height = maxy - miny
    scale_x = (output_width / bbox_width) * shrink_factor1
    scale_y = (output_height / bbox_height) * shrink_factor2
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: translate(geom, xoff=-minx, yoff=-miny))
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: scale(geom, xfact=scale_x, yfact=scale_y, origin=(0, 0)))
    minx_new, miny_new, maxx_new, maxy_new = gdf.total_bounds
    dx = (output_width - (maxx_new - minx_new)) / 2 - minx_new
    dy = (output_height - (maxy_new - miny_new)) / 2 - miny_new
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: translate(geom, xoff=dx, yoff=dy))
    return gdf

def aplicar_transformaciones(gdf, dx, dy, angle):
    gdf_temp = gdf.copy()
    gdf_temp['geometry'] = gdf_temp['geometry'].apply(lambda geom: translate(geom, xoff=dx, yoff=dy))
    gdf_temp['geometry'] = gdf_temp['geometry'].apply(lambda geom: rotate(geom, angle, origin='center'))
    return draw_geometry_to_mask(gdf_temp, output_width, output_height)

# === MAIN ===
gdf = procesar_geojson(ruta_geojson)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.resize(frame_gray, (output_width, output_height))
    frame_canny = cv2.Canny(frame_gray, 50, 120)

    best_val = -1
    best_config = None
    best_mask = None
    best_top_left = (0, 0)

    for dx, dy, angle in combinaciones:
        transformed_mask = aplicar_transformaciones(gdf, dx*10, dy*10, angle)
        result = cv2.matchTemplate(frame_canny, transformed_mask, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val
            best_config = (dx, dy, angle)
            best_mask = transformed_mask
            best_top_left = max_loc

    matched = best_val >= template_threshold
    frame_display = cv2.cvtColor(frame_canny, cv2.COLOR_GRAY2BGR)
    h, w = best_mask.shape
    bottom_right = (best_top_left[0] + w, best_top_left[1] + h)
    color = (0, 255, 0) if matched else (0, 0, 255)
    cv2.rectangle(frame_display, best_top_left, bottom_right, color, 2)

    overlay_region = frame_display[best_top_left[1]:bottom_right[1], best_top_left[0]:bottom_right[0]]
    if overlay_region.shape[:2] == best_mask.shape:
        overlay_region[:, :, 2] = np.maximum(best_mask, overlay_region[:, :, 2])

    cv2.putText(frame_display, f"Score: {best_val:.2f} - {'MATCH' if matched else 'NO MATCH'}", (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame_display, f"dx={best_config[0]}, dy={best_config[1]}, rot={best_config[2]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    print(f"Frame {frame_idx}: Score={best_val:.3f} | Desplazamiento=({best_config[0]}, {best_config[1]}) | "
          f"Rotación={best_config[2]}° | {'MATCH' if matched else 'NO MATCH'}")

    cv2.imshow("Template Matching Barrido + Rotación", frame_display)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
