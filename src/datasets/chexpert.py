import os
import logging
import numpy as np
import pandas as pd
import torch
from PIL import Image

logger = logging.getLogger(__name__)

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]


class CheXpertDataset(torch.utils.data.Dataset):
    """CheXpert chest X-ray dataset for multi-label classification.

    Filters to frontal views only. Uncertain labels (-1) and NaN are treated as 0 (negative).

    Concept-drift support: each frontal sample is paired with the lateral view
    from the same (patient, study) when one exists. Before drift, the dataset
    always serves the frontal image. When ``drift_active`` is flipped to True,
    samples whose study also has a lateral view serve the lateral image
    instead (labels unchanged). Samples without a lateral twin are unaffected.
    """
    def __init__(self, root, csv_file, transform=None, raw_fraction=1.0):
        self.root = root
        self.transform = transform
        self.drift_active = False

        df = pd.read_csv(csv_file)

        # build (patient, study) -> lateral path lookup before filtering
        lateral_df = df[df['Frontal/Lateral'] == 'Lateral']
        lateral_lookup = {}
        for csv_path in lateral_df['Path']:
            patient, study = self._patient_study(csv_path)
            # if multiple laterals exist for one study, keep the first (deterministic via CSV order)
            lateral_lookup.setdefault((patient, study), self._fix_path(csv_path))

        # frontal views only
        df = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

        # subsample if requested
        if raw_fraction < 1.0:
            n = max(1, int(len(df) * raw_fraction))
            df = df.sample(n=n, random_state=42).reset_index(drop=True)

        # image paths (fix prefix to match local directory structure)
        self.paths = df['Path'].apply(lambda p: self._fix_path(p)).tolist()
        # aligned lateral paths (None when no lateral twin exists for this study)
        self.lateral_paths = [
            lateral_lookup.get(self._patient_study(p)) for p in df['Path']
        ]
        # boolean mask: True iff this frontal sample has a lateral twin. Used as
        # the axis for the Dirichlet-based `custom` heterogeneous split.
        self.has_lateral = np.asarray(
            [p is not None for p in self.lateral_paths], dtype=bool
        )

        # multi-label targets: NaN -> 0, -1 (uncertain) -> 0, keep 1.0 as 1
        label_df = df[CHEXPERT_LABELS].fillna(0.0)
        label_df = label_df.replace(-1.0, 0.0)
        self.targets = label_df.values.astype(np.float32)

    @staticmethod
    def _patient_study(csv_path):
        parts = csv_path.split('/')
        return parts[-3], parts[-2]

    def _fix_path(self, csv_path):
        """Convert CSV path like 'CheXpert-v1.0-small/train/...' to local path."""
        # strip the 'CheXpert-v1.0-small/' prefix and join with root
        parts = csv_path.split('/')
        # find 'train' or 'valid' in the path and take from there
        for i, part in enumerate(parts):
            if part in ('train', 'valid'):
                return os.path.join(self.root, *parts[i:])
        return os.path.join(self.root, csv_path)

    def __getitem__(self, index):
        path = self.paths[index]
        if self.drift_active and self.lateral_paths[index] is not None:
            path = self.lateral_paths[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        return img, target

    def __len__(self):
        return len(self.paths)


class CheXpertWrapper(torch.utils.data.Dataset):
    """Wrapper to provide a `.targets` attribute for compatibility with split logic."""
    def __init__(self, dataset, dataset_name, suffix):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.suffix = suffix
        # For multi-label, use argmax of label vector as a pseudo-target for stratified splitting
        self.targets = self.dataset.targets.argmax(axis=1).tolist()
        # forward the pairing axis used by the `custom` split
        self.has_lateral = self.dataset.has_lateral

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return f'[{self.dataset_name}] {self.suffix}'


def fetch_chexpert(args, root, transforms):
    """Fetch the CheXpert dataset.

    Args:
        args: parsed arguments (uses args.rawsmpl for subsampling fraction)
        root: path to CheXpert directory (containing train/, valid/, train.csv, valid.csv)
        transforms: [train_transform, test_transform]

    Returns:
        raw_train, raw_test, args (with in_channels, num_classes set)
    """
    logger.info('[LOAD] [CHEXPERT] Fetching dataset!')

    raw_fraction = getattr(args, 'rawsmpl', 1.0)

    raw_train = CheXpertDataset(
        root=root,
        csv_file=os.path.join(root, 'train.csv'),
        transform=transforms[0],
        raw_fraction=raw_fraction
    )
    raw_test = CheXpertDataset(
        root=root,
        csv_file=os.path.join(root, 'valid.csv'),
        transform=transforms[1],
        raw_fraction=1.0  # always use full validation set
    )

    train_paired = sum(p is not None for p in raw_train.lateral_paths)
    test_paired = sum(p is not None for p in raw_test.lateral_paths)

    raw_train = CheXpertWrapper(raw_train, 'CHEXPERT', 'CLIENT')
    raw_test = CheXpertWrapper(raw_test, 'CHEXPERT', 'SERVER')

    args.in_channels = 3
    args.num_classes = len(CHEXPERT_LABELS)

    logger.info(
        f'[LOAD] [CHEXPERT] ...fetched dataset! '
        f'(train: {len(raw_train)} [{train_paired} with lateral twin], '
        f'test: {len(raw_test)} [{test_paired} with lateral twin])'
    )
    return raw_train, raw_test, args
