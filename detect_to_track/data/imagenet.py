"""datamanagers for imagenet"""

from pathlib import Path
from os import PathLike
from typing import Tuple, Dict
import random
from collections import defaultdict

from PIL import Image
from scipy.stats import dlaplace, binom
from ml_utils.data.pascal import parse_pascal_xmlfile, PascalObjectLabel

from .datamanager import DataManager
from .instances import ObjectLabel, RawImageInstance, ImageInstance


class VIDTrnSampler(object):
    """only samples from VID training set, and allows variable values of tau."""
    def __init__(self, data_root: PathLike, a: float = 0.8) -> None:
        self.label_root = Path(data_root, 'Annotations', 'VID', 'train')
        self.frame_root = Path(data_root, 'Data', 'VID', 'train')
        self._pascal_translator = _PascalTranslator(data_root, 'VID')

        self.a = a

        self.snippet_dirs = [
            snippet_dir
            for batch_dir in self.label_root.iterdir()
            for snippet_dir in batch_dir.iterdir()
        ]

    def sample(self) -> Tuple[ImageInstance, ImageInstance]:
        """sample frames from times t and t+tau.

        time step tau is sampled from a discrete laplacian distribution to
        create a bias towards small displacements.

        original paper only takes 10 frames from each snippet because
        of large differences in snippet lengths limiting sample diversity.

        uniformly sampling a snippet before sampling frames from that
        snippet solves the same problem, while also maximizing sample
        diversity within snippets.
        """
        # randomly sample a snippet dir and get associated sorted label paths
        snippet_dir = random.choice(self.snippet_dirs)
        label_paths = Path(self.label_root, snippet_dir).glob('*.xml')
        label_paths = sorted(list(label_paths))

        # randomly sample tau from discrete laplacian distribution
        tau = abs(dlaplace.rvs(self.a))

        # randomly sample first frame
        ind_0 = random.randrange(len(label_paths) - tau)
        ind_1 = ind_0 + tau

        instances = list()
        for label_ind in [ind_0, ind_1]:
            label_path = label_paths[label_ind]
            frame_path = Path(
                self.frame_root,
                label_path.relative_to(self.label_root).with_suffix('.JPEG')
            )

            instance = ImageInstance(
                im=Image.open(frame_path),
                labels=[
                    self._pascal_translator(pascal_object)
                    for pascal_object in parse_pascal_xmlfile(label_path)
                ]
            )
            instances.append(instance)

        instances = tuple(instances)

        return instances


class DETSampler(object):
    """samples from DET train + val. Instances containing classes that are not
    part of the VID dataset are ignored."""
    def __init__(self, data_root: PathLike) -> None:
        self.label_root = Path(data_root, 'Annotations', 'DET')
        self.frame_root = Path(data_root, 'Data', 'DET')
        self._pascal_translator = _PascalTranslator(data_root, 'VID')

        # mapping from class_name to list of label paths containing that
        # class. This will be useful for sampling later on.
        self.cls_label_paths = defaultdict(list)

        # populate cls_label_paths, ignoring classes that are not in VID
        allowed_class_ids = set(self._pascal_translator.id_to_int.keys())

        for label_path in list(self.label_root.rglob('*.xml')):
            class_ids = {
                pascal_object.class_id
                for pascal_object in parse_pascal_xmlfile(label_path)
            }

            # if any object has a class_id is not a VID class_id, this
            # instance is skipped
            if class_ids.issubset(allowed_class_ids):
                class_names = {
                    self._pascal_translator.id_to_name[class_id]
                    for class_id in class_ids
                }
                for class_name in class_names:
                    self.cls_label_paths[class_name].append(label_path)

    def sample(self) -> ImageInstance:
        """randomly sample instance from full DET dataset.

        original paper samples a fixed number of instances from each class
        because of class imbalance. This implementation uniformly samples from
        available classes first, then samples instances that contain that
        class. This solves the same problem without throwing data away.
        """
        class_name = random.choice(self.cls_label_paths.keys())

        label_path = random.choice(self.cls_label_paths[class_name])
        frame_path = Path(
            self.frame_root,
            label_path.relative_to(self.label_root).with_suffix('.JPEG')
        )

        instance = ImageInstance(
            im=Image.open(frame_path),
            object_labels=[
                self._pascal_translator(pascal_object)
                for pascal_object in parse_pascal_xmlfile(label_path)
            ]
        )

        return instance


class ImagenetTrnManager(DataManager):
    """samples from VID training set and entire DET dataset"""
    def __init__(self, data_root: PathLike, p_det: float = 0.5):
        self._det_sampler = DETSampler(data_root)
        self._vid_sampler = VIDTrnSampler(data_root)

        self.p_det = p_det

    def __getitem__(self, i: int) -> Tuple[ImageInstance, ImageInstance]:
        """sample from DET with probability p_det, or VID with probability
        1 - p_det.
        If sampling from DET use the same image, pretending that they are
        adjacent frames in a sequence.

        Args:
            i: not used.

        Returns:
            instance: pair of adjacent frames from a sequence along with labels.
        """
        sample_det = binom.rvs(1, self.p_det)

        if sample_det:
            instance = self._det_sampler.sample()

            # add arbitrary track_ids to DET instance
            instance = ImageInstance(
                im=instance.im,
                labels=tuple(
                    ObjectLabel(
                        class_id=label.class_id,
                        class_name=label.class_name,
                        box=label.box,
                        track_id=t_id
                    )
                    for t_id, label in enumerate(instance.labels)
                )
            )

            instance = (instance, instance)
        else:
            instance = self._vid_sampler.sample()

        return instance

    def __len__(self):
        raise ValueError('infinite')


class ImagenetValManager(DataManager):
    """handles data loading for Imagenet VID Dataset and conversion to common
    format.

    labels are loaded eagerly, images are loaded lazily.

    Args:
        data_root: dataset root directory.
        sample_size: number of instances to sample for validation set.
    """
    def __init__(self, data_root: PathLike, sample_size: int) -> None:
        """initialize and populate index_mappings."""
        label_root = Path(data_root, 'Annotations', 'VID', 'val')
        frame_root = Path(data_root, 'Data', 'VID', 'val')
        pascal_translator = _PascalTranslator(data_root, 'VID')

        snippet_dirs = {p for p in label_root.iterdir() if p.is_dir()}

        index_mappings = list()
        for _ in range(sample_size):
            snippet_dir = random.choice(snippet_dirs)

            label_paths = sorted(list(snippet_dir.glob('*.xml')))

            ind_0 = random.randrange(len(label_paths) - 2)

            instance_pair = list()
            for label_ind in [ind_0, ind_0 + 1]:
                label_path = label_paths[label_ind]
                frame_path = Path(
                    frame_root,
                    label_path.relative_to(label_root).with_suffix('.JPEG')
                )

                instance = RawImageInstance(
                    impath=frame_path,
                    labels=[
                        pascal_translator(pascal_object)
                        for pascal_object in parse_pascal_xmlfile(label_path)
                    ]
                )
                instance_pair.append(instance)

            index_mappings.append(tuple(instance_pair))

        self._index_mappings = tuple(index_mappings)

    def __getitem__(self, i: int) -> Tuple[ImageInstance, ImageInstance]:
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


class _PascalTranslator(object):
    """translates pascal labels."""
    def __init__(self, data_root: PathLike, task: str) -> None:
        self.id_to_int, self.id_to_name = self._load_cls_mappings(data_root, task)

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
        task = task.lower()
        if task not in {'vid', 'det'}:
            raise NotImplementedError(f'translating for {task} not implemented')

        id_to_int, id_to_name = dict(), dict()

        map_filepath = Path(data_root, 'devkit', 'data', f'map_{task}.txt')
        with open(map_filepath) as mapfile:
            for line in mapfile:
                cls_id, cls_int, cls_name = line.split()
                id_to_int[cls_id] = int(cls_int)
                id_to_name[cls_id] = cls_name

        return id_to_int, id_to_name

    def __call__(self, pascal_object: PascalObjectLabel) -> ObjectLabel:
        """pascal xmlfile parser doesnt have access to mappings from pascal_id
        to integer and class name, so we handle the mapping here.

        Args:
            pascal_object: pascal object label to be translated.

        Returns:
            object_label: translated object label.
        """
        return ObjectLabel(
            class_id=self.id_to_int[pascal_object.class_id],
            class_name=self.id_to_name[pascal_object.class_id],
            box=pascal_object.bbox,
            track_id=pascal_object.track_id
        )
