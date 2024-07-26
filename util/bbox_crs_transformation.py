from pyproj import Transformer

def transform_bbox_crs(bbox, src_crs, target_crs):
    """
    Transform the bounding box coordinates from the source CRS to the destination CRS.

    Parameters:
    bbox (tuple): A tuple of (left, bottom, right, top) coordinates in the source CRS.
    src_crs (str): The source CRS in string format (e.g., 'EPSG:3978').
    dest_crs (str): The destination CRS in string format (e.g., 'EPSG:4326').

    Returns:
    tuple: Transformed bounding box coordinates in the destination CRS.
    """
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)

    # Transform all four corners
    left_bottom = transformer.transform(bbox[0], bbox[1])
    right_bottom = transformer.transform(bbox[2], bbox[1])
    left_top = transformer.transform(bbox[0], bbox[3])
    right_top = transformer.transform(bbox[2], bbox[3])

    # Find the min and max for both latitude and longitude
    all_lons = [left_bottom[0], right_bottom[0], left_top[0], right_top[0]]
    all_lats = [left_bottom[1], right_bottom[1], left_top[1], right_top[1]]

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)

    new_bbox = (min_lon, min_lat, max_lon, max_lat)

    return new_bbox

