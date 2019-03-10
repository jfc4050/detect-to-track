"""imagenet DET data manager"""

from pathlib import Path
import re
from os import PathLike
from typing import Tuple

from PIL import Image
from ml_utils.data.pascal import parse_pascal_xmlfile

from . import ImageNetDataManager
from . import RawImageInstance, ImageInstance


class DETDataManager(ImageNetDataManager):
    """handles data loading and label conversion to common format. Outputs
    (ImageInstances) are used by Dataset objects.

    Args:
        data_root: dataset root directory.
    """
    def __init__(self, data_root: PathLike, mode: str) -> None:
        super().__init__(data_root, 'det')
        if mode not in {'train', 'val'}:
            raise ValueError(f'invalid mode {mode}')
        self.mode = mode
        self._index_mappings = self._preload_instances(data_root)

    def _preload_instances(self, data_root: PathLike) -> Tuple[RawImageInstance]:
        """partially preload and store instances. Object labels are converted
        from pascal objects to RawImageInstances.

        Args:
            data_root: dataset root directory.

        Returns:
            index_mappings: mapping from int -> raw frame instance (has an
                image path instead of an image).
        """
        index_mappings = list()
        label_dirs = set(
            p.parent for p
            in Path(data_root, 'Annotations', 'DET', self.mode).rglob('*.xml')
        )  # snippet dirs are at different levels depending on train or val
        for label_dir in label_dirs:
            for label_path in label_dir.glob('*.xml'):
                im_path = Path(re.sub(
                    '/Annotations/', '/Data/',
                    str(label_path.with_suffix('.JPEG'))
                ))
                raw_instance = RawImageInstance(
                    impath=im_path,
                    labels=[
                        self._translate_pascal_object(pascal_object)
                        for pascal_object in parse_pascal_xmlfile(label_path)
                    ]
                )
                index_mappings.append(raw_instance)

        return tuple(index_mappings)

    def __getitem__(self, i: int) -> ImageInstance:
        """retrive raw frame instance specified by i, then return loaded
        image and labels.

        Args:
            i: index of requested instance.

        Returns:
            image_instance: tuple containing image and its associated
                ground-truth labels.
        """
        raw_instance = self._index_mappings[i]
        return ImageInstance(
            im=Image.open(raw_instance.impath),
            object_labels=raw_instance.object_labels
        )

    def __len__(self):
        return len(self._index_mappings)
