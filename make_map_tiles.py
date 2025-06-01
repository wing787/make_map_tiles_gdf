import argparse
import math
import os

import geopandas as gpd
import pandas as pd
import tqdm
from shapely.geometry import box

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def get_bbox_from_gdf(gdf: gpd.GeoDataFrame) -> dict:
    """
    GeoDataFrameからバウンディングボックスを取得する。
    戻り値は (minx, miny, maxx, maxy) の形式。
    """
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty.")

    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    return {"minx": bounds[0], "miny": bounds[1], "maxx": bounds[2], "maxy": bounds[3]}


def import_vector_file(file_path: str) -> gpd.GeoDataFrame:
    """
    指定されたベクターファイルを読み込み、GeoDataFrameとして返す。
    対応するファイル形式はGeoJSON, Shapefile, KMLなど。
    """
    try:
        gdf = gpd.read_file(file_path)
        print(f"Successfully read vector file: {file_path}")
        return gdf
    except Exception as e:
        raise ValueError(f"Error reading vector file: {e}")


def deg2num(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """
    指定された緯度、経度、ズームレベルに基づいてタイル座標を計算します。
    参考：https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Python
    """
    lat = math.radians(lat)
    n = 2.0**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat)) / math.pi) / 2.0 * n)
    return xtile, ytile


def get_tile_coords_df(lonlat_dict: dict, zoom_level: int = None) -> pd.DataFrame:
    """
    指定された緯度経度範囲とズームレベルに基づいて、importしたベクターデータの範囲をカバーするタイル座標のDataFrameを生成する。
    """
    print(f"Generating tile coordinates for zoom level {zoom_level}...")

    x_nw, y_nw = deg2num(
        lonlat_dict["max_lat"], lonlat_dict["min_lon"], zoom_level
    )  # 左上
    x_se, y_se = deg2num(
        lonlat_dict["min_lat"], lonlat_dict["max_lon"], zoom_level
    )  # 右下
    x_sw, y_sw = deg2num(
        lonlat_dict["min_lat"], lonlat_dict["min_lon"], zoom_level
    )  # 左下
    x_ne, y_ne = deg2num(
        lonlat_dict["max_lat"], lonlat_dict["max_lon"], zoom_level
    )  # 右上

    # タイル座標の範囲を計算
    x_start = min(x_nw, x_sw)
    x_end = max(x_ne, x_se)
    y_start = min(y_nw, y_ne)
    y_end = max(y_sw, y_se)

    # タイル座標のリストを生成
    tile_coords = []
    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
            tile_coords.append({"x": x, "y": y, "z": zoom_level})

    tiles_df = pd.DataFrame(tile_coords)
    print(f"Generated {len(tiles_df)} tiles for zoom level {zoom_level}.")
    return tiles_df


def single_num2deg(xt: int, yt: int, zm: int) -> tuple[float, float]:
    """
    タイル座標の北西端の（経度, 緯度）を返すヘルパー関数
    """
    n = 2.0**zm
    lon_deg = xt / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * yt / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def num_to_tile_bounds(
    xtile: int, ytile: int, zoom: int
) -> tuple[float, float, float, float]:
    """
    タイル座標（xtile, ytile, zoom）から、そのタイルの地理的境界（左端経度、下端緯度、右端経度、上端緯度）を計算する。
    これはShapley.geometry.boxの引数の順序（minx, miny, maxx, maxy）に対応する。
    """
    # タイルの左上（北西）の経度・緯度
    lon_left, lat_top = single_num2deg(xtile, ytile, zoom)
    # タイルの右下（南東）の経度・緯度（（xtile+1, ytile+1）の左上に対応）
    lon_right, lat_bottom = single_num2deg(xtile + 1, ytile + 1, zoom)

    return (lon_left, lat_bottom, lon_right, lat_top)


def create_geodatafrane_from_tiles(tiles_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    タイル座標のDataFrameからGeoDataFrameを生成する.
    各タイルはポリゴンジオメトリとして表現される.
    """
    geometries = []
    for idx, row in tqdm(
        tiles_df.iterrows(), total=len(tiles_df), desc="Creating geometries"
    ):
        x, y, z = int(row["x"]), int(row["y"]), int(row["z"])
        bounds = num_to_tile_bounds(x, y, z)
        geometries.append(box(*bounds))

    gdf = gpd.GeoDataFrame(tiles_df, geometry=geometries, crs="EPSG:4326")
    print(f"Created GeoDataFrame with {len(gdf)} tiles.")
    return gdf


def clip_tiles_with_vector(
    gdf: gpd.GeoDataFrame, tiles_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    タイルのGeoDataFrameをベクターファイルのジオメトリでクリップする。
    """
    if gdf.empty or tiles_gdf.empty:
        raise ValueError("One of the GeoDataFrames is empty.")

    print(f"Clipping {len(tiles_gdf)} tiles with vector geometry...")
    clipped_gdf = gpd.overlay(tiles_gdf, gdf, how="intersection")
    return clipped_gdf


def main(
    read_gdf: gpd.GeoDataFrame, zoom_level: int = None, output_dir: str = None
) -> None:
    gdf = import_vector_file(read_gdf)
    lonlat_dict = get_bbox_from_gdf(gdf)
    tiles_df = get_tile_coords_df(lonlat_dict, zoom_level)
    tiles_gdf = create_geodatafrane_from_tiles(tiles_df)
    clipped_tiles_gdf = clip_tiles_with_vector(gdf, tiles_gdf)

    output_filename = f"clipped_tiles_z{zoom_level}.fgb"
    if output_dir:
        output_file = os.path.join(output_dir, output_filename)
    else:
        output_file = os.path.join(CUR_DIR, output_filename)
    clipped_tiles_gdf.to_file(
        output_file, driver="FlatGeobuf", engine="pyogrio", encoding="utf-8"
    )
    print(f"Clipped tiles file saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate map tiles for Any vector file."
    )
    parser.add_argument(
        "--vector-filepath", required=True, type=str, help="タイル座標を生成するベクターファイルのパス"
    )
    parser.add_argument("--zoom-level", required=True, type=int, help="タイルのズームレベル")
    parser.add_argument(
        "--output-filedir",
        required=False,
        type=str,
        help="クリップ済みの生成されたタイルGeoDataFrameを保存するディレクトリ（オプション）",
    )
    args = parser.parse_args()

    main(args.vector_filepath, args.zoom_level, args.output_filedir)
