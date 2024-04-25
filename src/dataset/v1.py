import json
import operator
from functools import cached_property, partial
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict
import numpy as np
import torch
from torch.utils.data import default_collate
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from src.dataset.captcha import CaptchaDataset
from src.definitions import DATA_PATH


class V1Prompt(TypedDict):
    category_id: int
    prompt_id: int
    prompt_text: str


class V1Question(TypedDict):
    question_text: str
    category_id: int
    # Which prompts should result in a human selecting the image for this question?
    positive_prompt_ids: List[int]


class V1Category(TypedDict):
    # We don't include a path here for flexibility of storage
    category_id: int
    category_name: str
    prompts: List[V1Prompt]
    questions: List[V1Question]


class V1Annnotation(TypedDict):
    path: str
    category_id: int
    prompt_id: int


class V1Challenge(CaptchaDataset):
    def __init__(self, img_root: str, annotations_path: str, challenge_size: int = 9, **data_kwargs):
        super().__init__(**data_kwargs)
        self._img_root_path: Path = DATA_PATH / img_root
        self._annotations_path: Path = DATA_PATH / annotations_path
        with open(str(self._annotations_path), "r") as annots_file:
            self._annotations = json.load(annots_file)
        self._images: List[V1Annnotation] = self._annotations["images"]
        self._categories: List[V1Category] = self._annotations["categories"]
        self._loader = default_loader
        self._challenge_size = challenge_size

    @property
    def _collate_fn(self):
        return None

    @property
    def challenge_size(self):
        return self._challenge_size

    @cached_property
    def _questions(self):
        all_questions = []
        for cat in self._categories:
            all_questions += cat["questions"]
        return all_questions

    @cached_property
    def _category_indices(self):
        return np.array([x["category_id"] for x in self._images])

    @cached_property
    def _category_map(self):
        return {c["category_id"]: c for c in self._categories}

    def _sample_images(self, n: int, category_id: int) -> List[Dict]:
        assert category_id in self._category_map
        cat_indices = np.flatnonzero(self._category_indices == category_id)
        sample_idx = np.random.choice(cat_indices, n, replace=False)
        if n == 1:
            return (self._images[sample_idx[0]],)
        return operator.itemgetter(*sample_idx)(self._images)

    def __len__(self):
        return len(self._questions)

    def __getitem__(self, idx):
        question = self._questions[idx]  
        question_text = question["question_text"].replace("Question: ", "")  # used to remove the prepended "Question:"
        category_id: int = question["category_id"]
        target_prompt_ids: List[int] = question["positive_prompt_ids"]
        im_data = self._sample_images(self.challenge_size,
                                      category_id)  # TODO temp changed to 1, also wrapped in [] brackets because it didn't want to make it a list when there was 1 sample. Basically, I had to make it a list so that I can un-make it a list at the end :)
        paths = [self._img_root_path / p["path"] for p in im_data]
        imgs = [self._loader(str(p)) for p in paths]

        # Unsqueezing the targets so shape is (batch_size, 1)
        targets = torch.tensor([x["prompt_id"] in target_prompt_ids for x in im_data]).float().unsqueeze(0)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_tensors = torch.stack([transform(i) for i in imgs])
        questions = [question_text for _ in imgs]

        return questions, img_tensors, targets.T


class V1SingleImage(V1Challenge):
    def __init__(self, shuffle: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shuffle = shuffle

    @cached_property
    def _idx_ordering(self):
        ordering = np.arange(len(self))
        if self._shuffle:
            ordering = np.random.permutation(ordering)
        return ordering

    def __len__(self):
        return len(self._images)

    @property
    def should_flatten(self):
        return False

    @cached_property
    def _collate_fn(self):
        return super()._collate_fn

    def __getitem__(self, idx):
        img = self._images[self._idx_ordering[idx]]
        cat_id: int = img["category_id"]
        category = self._category_map[cat_id]
        question = np.random.choice(category["questions"])
        question_text = question["question_text"].replace("Question: ", "")  # Used to remove the prepended "Question:"
        

        target_prompt_ids: List[int] = question["positive_prompt_ids"]
        path = self._img_root_path / img["path"]
        img_data = self._loader(str(path))
        target = torch.tensor(img["prompt_id"] in target_prompt_ids).float().unsqueeze(0)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_data = transform(img_data)
        return question_text, img_data, target
