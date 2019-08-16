"""data management for imagenet."""

from pathlib import Path
from os import PathLike
from typing import Tuple, Dict, Sequence, Set
import random
from collections import defaultdict

from PIL import Image
import numpy as np
from scipy.stats import dlaplace, bernoulli
from ml_utils.data.pascal import parse_pascal_xmlfile, PascalObjectLabel
from ml_utils.data.misc import partition_items

from .types import (
    DataSampler,
    DataManager,
    DataManagerWrapper,
    ObjectLabel,
    RawImageInstance,
    ImageInstance,
)


class _VIDRawSampler:
    """non-deterministically samples raw image instances (paths only)
    from VID training set. training example images and labels are loaded lazily.

    Args:
        data_root: ILSVRC dataset root.
        snippet_ids: snippet identifiers that this sampler can sample from.
        a: shape parameter for laplacian distribution (of tau).
            PMF: f(x) = tanh(a/2)exp(-a|x|).
    """

    def __init__(self, data_root: Path, snippet_ids: Sequence[str], a: float) -> None:
        self.label_root = Path(data_root, "Annotations", "VID", "train")
        self.frame_root = Path(data_root, "Data", "VID", "train")
        self.a = a

        self._snippet_framecounts = dict()
        for snippet_id in snippet_ids:
            n_frames = len(list(Path(self.frame_root, snippet_id).glob("*.JPEG")))
            n_labels = len(list(Path(self.label_root, snippet_id).glob("*.xml")))

            if n_frames != n_labels:
                raise RuntimeError(
                    f"for snippet {snippet_id} "
                    f"found {n_frames} frames but {n_labels} labels"
                )
            self._snippet_framecounts[snippet_id] = n_frames

    def sample(self) -> Tuple[RawImageInstance, RawImageInstance]:
        """sample frames from time steps t and t+tau.

        time delta tau is sampled from a discrete laplacian distribution to
        create a bias towards small displacements.

        original paper suggests only taking 10 frames from each snippet because
        of large differences in snippet lengths limiting dataset diversity.

        uniformly sampling a snippet before sampling frames from that
        snippet solves the same problem, while also maximizing sample
        diversity within snippets.
        """
        snippet_id, snippet_n_frames = random.choice(
            list(self._snippet_framecounts.items())
        )

        # sample tau from discrete laplacian distribution,
        # then clip so that it will not result in an invalid pair of frames
        tau = np.clip(dlaplace.rvs(self.a), 0, snippet_n_frames - 1)
        i0 = random.randrange(0, snippet_n_frames - tau)

        raw_instances = tuple(
            RawImageInstance(
                impath=Path(self.frame_root, snippet_id, f"{idx:06d}.JPEG"),
                labelpath=Path(self.label_root, snippet_id, f"{idx:06d}.xml"),
            )
            for idx in (i0, i0 + tau)
        )

        return raw_instances


class VIDSampler(DataSampler):
    """samples from VID training set at query time, and allows variable values of tau.
    queries are non-deterministic.

    Args:
        data_root: ILSVRC dataset root.
        snippet_ids: snippet identifiers that this sampler can sample from.
        a: shape parameter for laplacian distribution (of tau).
            PMF: f(x) = tanh(a/2)exp(-a|x|).
    """

    def __init__(
        self, data_root: PathLike, snippet_idents: Sequence[str], a: float
    ) -> None:
        self._raw_sampler = _VIDRawSampler(data_root, snippet_idents, a)
        self._pascal_translator = _PascalTranslator(data_root, "VID")

    def sample(self) -> Tuple[ImageInstance, ImageInstance]:
        """non-deterministically query `raw_sampler` for image and label paths,
        then load images and labels."""

        return tuple(
            ImageInstance(
                im=Image.open(ri.impath),
                labels=tuple(
                    self._pascal_translator(pascal_object)
                    for pascal_object in parse_pascal_xmlfile(ri.labelpath)
                ),
            )
            for ri in self._raw_sampler.sample()
        )


class VIDManager(DataManager):
    """samples from VID training set at init time to form a dataset, which can
    then be queried deterministically.

    Args:
        data_root: ILSVRC dataset root.
        snippet_ids: snippet identifiers that this sampler can sample from.
        n_samples: number of samples to draw (non-deterministically) from VID
            training set to form dataset.
    """

    def __init__(
        self, data_root: PathLike, snippet_idents: Sequence[str], n_samples: int
    ) -> None:
        raw_sampler = _VIDRawSampler(data_root, snippet_idents, 0.5)
        self._raw_samples = [raw_sampler.sample() for _ in range(n_samples)]
        self._pascal_translator = _PascalTranslator(data_root, "VID")

    def __getitem__(self, i: int) -> Tuple[ImageInstance, ImageInstance]:
        return tuple(
            ImageInstance(
                im=Image.open(ri.impath),
                labels=tuple(
                    self._pascal_translator(pascal_object)
                    for pascal_object in parse_pascal_xmlfile(ri.labelpath)
                ),
            )
            for ri in self._raw_samples[i]
        )

    def __len__(self) -> int:
        return len(self._raw_samples)


class DETRawSampler:
    """randomly samples raw instances (paths to images and labels) from
    DET train+val."""

    def __init__(
        self, data_root: Path, allowed_class_ids: Set[str], allowed_class_ints: Set[int]
    ) -> None:
        label_root = Path(data_root, "Annotations", "DET")
        frame_root = Path(data_root, "Data", "DET")

        # mapping from class_name to list of label paths containing that
        # class. This will be useful for sampling later on.
        self._rawinstances_by_cls = defaultdict(list)

        trn_files = [f"train_{cls_id}" for cls_id in allowed_class_ints]
        val_files = ["val"]
        for mode, files in zip(["train", "val"], [trn_files, val_files]):
            for f in files:
                instance_list_path = Path(data_root, "ImageSets", "DET", f"{f}.txt")
                with open(instance_list_path) as instance_list:
                    for line in instance_list:
                        instance_id, _ = line.split()
                        if "extra" in instance_id:
                            continue
                        framepath = Path(frame_root, mode, f"{instance_id}.JPEG")
                        labelpath = Path(label_root, mode, f"{instance_id}.xml")

                        class_ids = {
                            pascal_object.class_id
                            for pascal_object in parse_pascal_xmlfile(labelpath)
                        }
                        if class_ids.issubset(allowed_class_ids):
                            ri = RawImageInstance(impath=framepath, labelpath=labelpath)
                            for class_id in class_ids:
                                self._rawinstances_by_cls[class_id].append(ri)

    def sample(self) -> RawImageInstance:
        """paper suggests sampling a fixed number of instances from each class
        because of class imbalance. This implementation instead uniformly samples from
        available classes first, then samples instances that contain that
        class. This solves the same problem without throwing data away.
        """
        sampled_cls_id = random.choice(list(self._rawinstances_by_cls.keys()))
        raw_instance = random.choice(self._rawinstances_by_cls[sampled_cls_id])

        return raw_instance


class DETSampler(DataSampler):
    """non-deterministically samples from DET train + val.
    Instances containing classes that are not part of the VID dataset are ignored."""

    def __init__(self, data_root: PathLike) -> None:
        self._pascal_translator = _PascalTranslator(data_root, "VID")
        allowed_class_ids = set(self._pascal_translator.id_to_int.keys())
        allowed_class_ints = set(self._pascal_translator.id_to_int.values())
        self._raw_sampler = DETRawSampler(
            data_root, allowed_class_ids, allowed_class_ints
        )

    def sample(self) -> ImageInstance:
        raw_instance = self._raw_sampler.sample()
        instance = ImageInstance(
            im=Image.open(raw_instance.impath),
            labels=tuple(
                self._pascal_translator(pascal_object)
                for pascal_object in parse_pascal_xmlfile(raw_instance.labelpath)
            ),
        )
        return instance


class ImagenetSampler(DataSampler):
    """samples from union of VID training set and entire DET dataset.

    Args:
        vid_sampler: samples from VID training set.
        det_sampler: samples from DET train+val dataset.
        p_det: probability of sampling from DET sampler.
            p(DET) = p_det, p(VID) = 1 - p_det
    """

    def __init__(
        self, vid_sampler: DataSampler, det_sampler: DataSampler, p_det: float
    ) -> None:
        self._vid_sampler = vid_sampler
        self._det_sampler = det_sampler
        self.p_det = p_det

    def sample(self) -> Tuple[ImageInstance, ImageInstance]:
        sample_det = bernoulli.rvs(self.p_det)

        if sample_det:
            instance = self._det_sampler.sample()
            instance = ImageInstance(
                im=instance.im,
                labels=tuple(
                    ObjectLabel(
                        class_id=label.class_id,
                        class_name=label.class_name,
                        box=label.box,
                        track_id=t_id,  # add arbitrary track_ids to DET instance.
                    )
                    for t_id, label in enumerate(instance.labels)
                ),
            )
            # when sampling from DET use the same image twice, pretending that they are
            # adjacent frames in a sequence.
            instance = (instance, instance)
        else:
            instance = self._vid_sampler.sample()

        return instance


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
        if task not in {"vid", "det"}:
            raise NotImplementedError(f"translating for {task} not implemented")

        id_to_int, id_to_name = dict(), dict()

        map_filepath = Path(data_root, "devkit", "data", f"map_{task}.txt")
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
            track_id=pascal_object.track_id,
        )


def find_vid_trn_snippet_ids(data_root: Path) -> Tuple[str, ...]:
    """find snippet identifiers for VID training set. identifiers take the form of
    $BATCH_NAME/$SNIPPET_NAME."""
    imagesets_dir = Path(data_root, "ImageSets", "VID")
    frame_root = Path(data_root, "Data", "VID", "train")
    label_root = Path(data_root, "Annotations", "VID", "train")

    trn_snippet_ids = list()
    for trn_list_path in imagesets_dir.glob("train_[0-9]?.txt"):
        with open(trn_list_path) as trn_list_file:
            for line in trn_list_file:
                snippet_id, _ = line.split()

                for sub_dir in [frame_root, label_root]:
                    snippet_sub_dir = Path(sub_dir, snippet_id)
                    if not snippet_sub_dir.is_dir():
                        raise FileNotFoundError(f"couldn't find {snippet_sub_dir}")

                trn_snippet_ids.append(snippet_id)

    trn_snippet_ids = tuple(trn_snippet_ids)

    return trn_snippet_ids


def setup_vid_datasets(
    data_root: Path,
    vid_partition_sizes: Tuple[int, int],
    trn_size: int,
    val_size: int,
    rep_size: int,
    p_det: float,
    a: float,
) -> Tuple[DataManager, DataManager, DataManager]:
    """put together datasets for training on VID+DET."""
    vid_snippet_ids = find_vid_trn_snippet_ids(data_root)
    trn_snippets, val_snippets = partition_items(vid_snippet_ids, vid_partition_sizes)

    trn_vid_sampler = VIDSampler(data_root, trn_snippets, a)
    val_manager = VIDManager(data_root, val_snippets, val_size)
    rep_manager = VIDManager(data_root, trn_snippets, rep_size)

    det_sampler = DETSampler(data_root)

    trn_sampler = ImagenetSampler(trn_vid_sampler, det_sampler, p_det)
    trn_manager = DataManagerWrapper(trn_sampler, trn_size)

    return trn_manager, val_manager, rep_manager


def make_mock_dataset(data_root: Path, n_samples: int) -> DataManager:
    """make small subset of full dataset for quick iteration."""
    vid_snippet_ids = find_vid_trn_snippet_ids(data_root)
    vid_manager = VIDManager(data_root, vid_snippet_ids, n_samples)

    return vid_manager
