"""data management and loading"""

from pathlib import Path
import re
from os import PathLike
from typing import Tuple, Dict

from PIL import Image
from ml_utils.data.pascal import parse_pascal_xmlfile, PascalObjectLabel

from . import DataManager
from . import ObjectLabel, RawImageInstance, ImageInstance


class VIDDataManager(DataManager):
    """handles data loading for Imagenet VID Dataset and conversion to common
    format.

    Args:
        data_root: dataset root directory.
        seq_len: unlinked sequence length.
    """
    def __init__(self, data_root: PathLike, seq_len: int = 2):
        self._id_to_int, self._id_to_name = self._load_cls_mappings(data_root)
        self._index_mappings = self._preload_instances(data_root, seq_len)

    @staticmethod
    def _load_cls_mappings(
            data_root: PathLike
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """load mappings as dictionaries.

        Args:
            data_root: dataset root directory.

        Returns:
            id_to_int: vid_id -> int_id.
            id_to_name: vid_id -> class name.
        """
        id_to_int, id_to_name = dict(), dict()
        with open(Path(data_root, 'devkit', 'data', 'map_vid.txt')) as mapfile:
            for line in mapfile:
                cls_id, cls_int, cls_name = line.split()
                id_to_int[cls_id] = int(cls_int)
                id_to_name[cls_id] = cls_name

        return id_to_int, id_to_name

    def _preload_instances(
            self,
            data_root: PathLike,
            seq_len: int
    ) -> Tuple[Tuple[RawImageInstance, ...], ...]:
        """partially preload and store instances. Object labels are converted
        from Pascal objects to common objects
        (see self._translate_pascal_object).

        Args:
            data_root: dataset root directory.
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
                    raw_frame_seq.append(RawImageInstance(
                        impath=impath,
                        object_labels=[
                            self._translate_pascal_object(pascal_object)
                            for pascal_object in parse_pascal_xmlfile(labelpath)
                        ]
                    ))

                index_mappings.append(tuple(raw_frame_seq))

        return tuple(index_mappings)

    def _translate_pascal_object(
            self,
            pascal_object: PascalObjectLabel
    ) -> ObjectLabel:
        """pascal xmlfile parser doesnt have access to mappings from pascal_id
        to integer and class name, so we handle the mapping here.

        Args:
            pascal_object: pascal object label to be translated.

        Returns:
            object_label: translated object label.
        """
        return ObjectLabel(
            class_id=self._id_to_int[pascal_object.class_id],
            class_name=self._id_to_name[pascal_object.class_id],
            bbox=pascal_object.bbox,
            track_id=pascal_object.track_id
        )

    def __getitem__(self, i: int) -> Tuple[ImageInstance, ...]:
        """get impath and labels for instance specified by i, then return
        loaded image and labels.

        Args:
            i (int): index of requested instance.

        Returns:
            frame_instances: human-readable sequence of frame instances.
        """
        frame_instances = tuple([
            ImageInstance(
                im=Image.open(raw_instance.impath),  # load image
                object_labels=raw_instance.object_labels
            )
            for raw_instance in self._index_mappings[i]
        ])

        return frame_instances

    def __len__(self) -> int:
        return len(self._index_mappings)
