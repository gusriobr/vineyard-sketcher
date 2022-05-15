"""
Post processing to create feature file out from raster images
"""
import logging
import os
import sys
from collections import OrderedDict

import fiona
import numpy as np
import rasterio
from fiona.crs import from_epsg
from rasterio import features
from shapely.geometry import shape

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from sketcher import cfg

cfg.configLog()


def vectorize_predictions(raster_file, shape_file):
    logging.info("Started image vectorization from raster {} into shapefile {}".format(raster_file, shape_file))
    polys = []
    with rasterio.open(raster_file) as src:
        img = src.read()
        # get mask different values
        feature_ids = np.unique(img)
        feature_ids = feature_ids[1:]  # remove first zero
        for feature_id in feature_ids:
            mask = img == feature_id
            feature_gen = features.shapes(img, mask=mask, transform=src.transform)
            polys.extend(list(feature_gen))

    logging.info("{} polygons extracted".format(len(polys)))

    # if file exist append otherwise create
    if os.path.exists(shape_file):
        open_f = fiona.open(shape_file, 'a')
    else:
        # output_driver = "GeoJSON"
        output_driver = 'ESRI Shapefile'
        vineyard_schema = {
            'geometry': 'Polygon',
            'properties': OrderedDict([('FID', 'int')])
        }
        crs = from_epsg(25830)
        open_f = fiona.open(shape_file, 'w', driver=output_driver, crs=crs, schema=vineyard_schema)

    with open_f as c:
        for idx, p in enumerate(polys):
            poly_feature = {"geometry": p[0], "properties": {"FID": idx}}
            c.write(poly_feature)

    logging.info("Vectorization finished.")


def filter_by_area(feature):
    area = shape(feature["geometry"]).area
    return area > 450


def filter_features(input_file, output_file):
    with fiona.open(input_file) as source:
        source_driver = source.driver
        source_crs = source.crs
        source_schema = source.schema
        polys_filtered = list(filter(filter_by_area, source))

    with fiona.open(output_file, "w", driver=source_driver, schema=source_schema, crs=source_crs) as dest:
        for r in polys_filtered:
            dest.write(r)


if __name__ == '__main__':
    input_folder = cfg.results("processed_v4/2020")
    # input_images = [os.path.join(input_folder, f_img) for f_img in os.listdir(input_folder) if f_img.endswith(".tif")]
    # output_file = cfg.results("vineyard_sketch_polygons.shp")

    input_images = [cfg.results('iteration3/predictions/aerial2_sketcher.tif')]
    output_file = cfg.results('iteration3/predictions/parcels.shp')
    total = len(input_images)
    for i, f_image in enumerate(input_images):
        logging.info("Vectorizing image {} of {}".format(i + 1, total))
        vectorize_predictions(f_image, output_file)

    filtered_output_file = output_file.replace(".shp", "_filtered.shp")
    logging.info("Filtering out small polygons")
    filter_features(output_file, filtered_output_file)

    logging.info("Filtered geometries successfully written")
    logging.info("Output file: {}".format(filtered_output_file))
