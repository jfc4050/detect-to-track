"""data management and loading"""

from pathlib import Path
from os import PathLike
from typing import Tuple, NamedTuple, Dict
import re

import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from ml_utils import data_utils

from .datamanager import DataManager

__all__ = ['FrameInstance', 'VIDDataManager']


class _RawFrameInstance(NamedTuple):
    """unprocessed, immutable frame instance for storage"""
    impath: Path
    class_ids: Tuple[str]
    bboxes: Tuple[Tuple[float]]


class FrameInstance(NamedTuple):
    """human readable frame instance"""
    im: JpegImageFile
    classes: np.ndarray
    bboxes: np.ndarray


class VIDDataManager(DataManager):
    """handles data loading for Imagenet VID Dataset.

    Args:
        data_root: dataset root directory.
        seq_len (int): unlinked sequence length.
    """
    def __init__(self, data_root: PathLike, seq_len: int = 2):
        self._cls_mappings = self._load_cls_mappings(data_root)
        self._index_mappings = self._preload_instances(data_root, seq_len)

    @staticmethod
    def _load_cls_mappings(data_root: PathLike) -> Dict[str, int]:
        """load mappings as dictionary.

        Args:
            data_root: dataset root directory.

        Returns:
            cls_mappings: vid_id -> int_id.
        """
        cls_mappings = dict()
        with open(Path(data_root, 'devkit', 'data', 'map_vid.txt')) as mapfile:
            for line in mapfile:
                cls_id, cls_int, _ = line.split()
                cls_mappings[cls_id] = int(cls_int)

        return cls_mappings

    @staticmethod
    def _preload_instances(
            data_root: PathLike,
            seq_len: int
    ) -> Tuple[Tuple[_RawFrameInstance, ...], ...]:
        """partially preload and store instances.

        Args:
            seq_len: desired short sequence length.

        Returns:
            index_mappings: int -> sequence of RawFrameInstances.
        """
        index_mappings = list()
        snippet_dirs = set(
            p.parent for p in Path(data_root, 'Data', 'VID').rglob('*.JPEG')
        )  # snippet dirs are at different levels depending on train or val
        for snippet_dir in snippet_dirs:
            for frame0_num in range(
                    sum(1 for _ in snippet_dir.glob('*.JPEG')) - seq_len + 1
            ):
                raw_frame_seq = list()
                for frame_num in range(frame0_num, frame0_num + seq_len):
                    impath = Path(snippet_dir, f'{frame_num:06d}.JPEG')
                    labelpath = Path(re.sub(
                        '/Data/', '/Annotations/',
                        str(impath.with_suffix('.xml'))
                    ))
                    cls_ids, bboxes = data_utils.parse_pascal_xmlfile(labelpath)
                    raw_frame_seq.append(_RawFrameInstance(
                        impath=impath,
                        class_ids=cls_ids,
                        bboxes=bboxes
                    ))

                index_mappings.append(tuple(raw_frame_seq))

        return tuple(index_mappings)

    def __getitem__(self, i: int) -> Tuple[FrameInstance, ...]:
        """load instance specified by i.

        Args:
            i (int): index of requested instance.

        Returns:
            frame_instances: human-readable sequence of frame instances.
        """
        frame_instances = tuple([
            FrameInstance(
                im=Image.open(raw_instance.impath),  # load image
                classes=np.array([
                    self._cls_mappings[c] for c in raw_instance.class_ids
                ]),  # str -> int and convert to array
                bboxes=np.array(raw_instance.bboxes)  # convert to array
            )
            for raw_instance in self._index_mappings[i]
        ])

        return frame_instances

    def __len__(self) -> int:
        return len(self._index_mappings)
