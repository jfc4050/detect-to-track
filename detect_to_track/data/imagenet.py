"""datamanagers for imagenet"""

import abc
from pathlib import Path
import re
from os import PathLike
from typing import Tuple, Dict

from PIL import Image
from ml_utils.data.pascal import parse_pascal_xmlfile, PascalObjectLabel

from .datamanager import DataManager
from .instances import ObjectLabel, RawImageInstance, ImageInstance


class ImageNetDataManager(DataManager):
    """handles dataloading form imagenet datasets."""
    allowed_tasks = {'det', 'vid'}

    def __init__(self, data_root: PathLike, task: str) -> None:
        if task not in self.allowed_tasks:
            raise ValueError(f'task {task} not available.')
        self._id_to_int, self._id_to_name = self._load_cls_mappings(data_root, task)

    @staticmethod
    def _load_cls_mappings(
            data_root: PathLike, task: str
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """load mappings as dictionaries.

        Args:
            data_root: dataset root directory.
            task: name of classification task.

        Returns:
            id_to_int: vid_id -> int_id.
            id_to_name: vid_id -> class name.
        """
        id_to_int, id_to_name = dict(), dict()
        with open(
                Path(data_root, 'devkit', 'data', f'map_{task}.txt')
        ) as mapfile:
            for line in mapfile:
                cls_id, cls_int, cls_name = line.split()
                id_to_int[cls_id] = int(cls_int)
                id_to_name[cls_id] = cls_name

        return id_to_int, id_to_name

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
            box=pascal_object.bbox,
            track_id=pascal_object.track_id
        )

    @abc.abstractmethod
    def __getitem__(self, i):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError


class DETDataManager(ImageNetDataManager):
    """handles data loading and label conversion to common format on behalf of
    dataset objects.

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
        """partially preload (images loaded lazily, labels loaded eagerly)
        and store instances. Object labels are converted from pascal objects
        to RawImageInstances.

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
        """partially preload (sequences loaded lazily, labels loaded eagerly)
        and store instances. Object labels are converted from Pascal objects
        to common objects (see self._translate_pascal_object).

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
                        labels=[
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
                labels=raw_instance.object_labels
            )
            for raw_instance in self._index_mappings[i]
        ])

        return frame_instances

    def __len__(self) -> int:
        return len(self._index_mappings)
