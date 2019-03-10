"""abstract datamanager class"""

import abc
from pathlib import Path
from os import PathLike
from typing import Tuple, Dict

from ml_utils.data.pascal import PascalObjectLabel

from . import ObjectLabel


class ImageNetDataManager(abc.ABC):
    """handles dataloading"""
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
            bbox=pascal_object.bbox,
            track_id=pascal_object.track_id
        )

    @abc.abstractmethod
    def __getitem__(self, i):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError
