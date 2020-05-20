# from utils_cv.detection.bbox import _Bbox

# class TrackingBbox(_Bbox):
#     """Inherits from _Bbox"""
#     def __init__(
#         self,
#         left: int,
#         top: int,
#         right: int,
#         bottom: int,
#         frame_id: int,
#         track_id: int,
#     ):
#         """ Initialize TrackingBbox """
#         super().__init__(left, top, right, bottom)
#         self.frame_id = frame_id
#         self.track_id = track_id

from pathlib import Path

def format(p: Path = ""):
    print(f"--path {p}")

format()