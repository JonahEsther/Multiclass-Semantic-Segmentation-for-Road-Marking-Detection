import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon, LineString
from tqdm import tqdm
import subprocess
from shutil import copyfile
import numpy as np

# Input paths
vector_path = "../data/Apeldoorn_grids.shp"  # Grid file
raster_path = "../data/D028_lufo_2022_Almere.ecw"  # Raster file
shapefile_path = "../data/Roads_Almere.shp"  # Road shapefile
dash_shp = "../data/DashedLines.shp"  # annotation shapefiles 
solid_shp = "../data/SolidLines.shp"
triangle_shp = "../data/TriangleMark.shp"
block_shp = "../data/BlockMark.shp"

# Output paths
output_dir = "../data/tiles"  # Directory for cropped tiles
clipped_images_dir = "../data/clipped_images"  # Directory for clipped images
training_data_dir = "../data/training"  # Directory for training data

# Subdirectories for training data
images_dir = os.path.join(training_data_dir, "images")
masks_dir = os.path.join(training_data_dir, "masks")

# Create directories if they don't exist
for dir_path in [output_dir, clipped_images_dir, images_dir, masks_dir]:
    os.makedirs(dir_path, exist_ok=True)

def tile_raster():
    """
    Tiles the raster into smaller images based on the grid file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vector_layer = gpd.read_file(vector_path)
    vector_layer['id'] = vector_layer['id'].astype(int)

    for idx, row in tqdm(vector_layer.iterrows(), total=vector_layer.shape[0], desc="Processing tiles"):
        try:
            minx, miny, maxx, maxy = row.geometry.bounds
            projwin = f"{minx} {maxx} {miny} {maxy} [EPSG:28992]"
            output_raster = os.path.join(output_dir, f"cropped_tile_{row['id']}.tif")
            command = (
                f'qgis_process run gdal:cliprasterbyextent '
                f'--distance_units=meters --area_units=m2 --ellipsoid=EPSG:7004 '
                f'--INPUT="{raster_path}" '
                f'--PROJWIN="{projwin}" '
                f'--OVERCRS=false --OPTIONS="" --DATA_TYPE=0 --EXTRA="" '
                f'--OUTPUT="{output_raster}"'
            )
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(f"Cropped tile saved as: {output_raster}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing tile {row['id']}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

def mask_tiles():
    """
    Masks the tiled images using the road shapefile.
    """
    road_polygon = gpd.read_file(shapefile_path)
    image_files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.tif', '.tiff'))]

    for image_file in image_files:
        image_path = os.path.join(output_dir, image_file)
        with rasterio.open(image_path) as src:
            out_image, out_transform = mask(src, road_polygon.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            output_image_path = os.path.join(clipped_images_dir, image_file)
            with rasterio.open(output_image_path, "w", **out_meta) as dest:
                dest.write(out_image)

    print(f"Clipped images saved to {clipped_images_dir}.")

def buffer_and_rasterize():
    """
    Buffers the annotation shapes and rasterizes them into masks.
    """
    folders = {
        "Dash": dash_shp,
        "Solid": solid_shp,
        "Triangle": triangle_shp,
        "Block": block_shp,
    }

    # Create subdirectories for each class
    for folder_name in folders.keys():
        os.makedirs(os.path.join(images_dir, folder_name), exist_ok=True)
        os.makedirs(os.path.join(masks_dir, folder_name), exist_ok=True)

    gdfs = {
        "Dash": gpd.read_file(dash_shp),
        "Solid": gpd.read_file(solid_shp),
        "Triangle": gpd.read_file(triangle_shp),
        "Block": gpd.read_file(block_shp),
    }

    # Apply buffers and fill triangles
    gdfs["Dash"]["geometry"] = gdfs["Dash"].geometry.buffer(0.215, cap_style=2)
    gdfs["Solid"]["geometry"] = gdfs["Solid"].geometry.buffer(0.215, cap_style=2)
    gdfs["Block"]["geometry"] = gdfs["Block"].geometry.buffer(0.33, cap_style=2)

    def fill_triangle(gdf):
        filled_triangles = []
        for geom in gdf.geometry:
            if isinstance(geom, LineString):
                filled_triangles.append(Polygon(geom))
            else:
                filled_triangles.append(geom)
        gdf.geometry = filled_triangles
        return gdf

    gdfs["Triangle"] = fill_triangle(gdfs["Triangle"])

    # Rasterize annotations
    for image_file in os.listdir(clipped_images_dir):
        if image_file.endswith('.tif'):
            image_path = os.path.join(clipped_images_dir, image_file)
            with rasterio.open(image_path) as src:
                transform = src.transform
                out_shape = (src.height, src.width)
                bounds = src.bounds
                crs = src.crs

            for folder_name, gdf in gdfs.items():
                filtered_gdf = gdf[gdf.geometry.intersects(Polygon.from_bounds(*bounds))]
                if not filtered_gdf.empty:
                    sub_image_dir = os.path.join(images_dir, folder_name)
                    sub_mask_dir = os.path.join(masks_dir, folder_name)

                    # Copy image to class-specific directory
                    dest_image_path = os.path.join(sub_image_dir, image_file)
                    copyfile(image_path, dest_image_path)

                    # Rasterize the mask
                    mask_array = np.zeros(out_shape, dtype=np.uint8)
                    shapes = ((geom, 1) for geom in filtered_gdf.geometry)
                    rasterized_mask = rasterio.features.rasterize(
                        shapes=shapes,
                        out=mask_array,
                        transform=transform,
                        dtype=np.uint8
                    )

                    # Save the mask
                    mask_path = os.path.join(sub_mask_dir, image_file)
                    with rasterio.open(
                        mask_path,
                        "w",
                        driver="GTiff",
                        height=rasterized_mask.shape[0],
                        width=rasterized_mask.shape[1],
                        count=1,
                        dtype=np.uint8,
                        crs=crs,
                        transform=transform
                    ) as dst:
                        dst.write(rasterized_mask, 1)

    print("Dataset creation complete!")
    
if __name__ == "__main__":
    tile_raster()  # Step 1: Tile the raster
    mask_tiles()  # Step 2: Mask tiles using road shapefile
    buffer_and_rasterize()  # Step 3: Buffer and rasterize annotations