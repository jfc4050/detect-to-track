"""data management and loading"""

from pathlib import Path
import re
from os import PathLike
from typing import Tuple

from PIL import Image
from ml_utils.data.pascal import parse_pascal_xmlfile

from . import ImageNetDataManager
from . import RawImageInstance, ImageInstance


class VIDDataManager(ImageNetDataManager):
    """handles data loading for Imagenet VID Dataset and conversion to common
    format.

    Args:
        data_root: dataset root directory.
        seq_len: unlinked sequence length.
    """
    def __init__(self, data_root: PathLike, seq_len: int = 2):
        super().__init__(data_root, 'vid')
        self._index_mappings = self._preload_instances(data_root, seq_len)

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

    def __getitem__(self, i: int) -> Tuple[ImageInstance, ...]:
        """get impath and labels for instance specified by i, then return
        loaded image and labels.

        Args:
            i: index of requested instance.

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
