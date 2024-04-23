import json
import operator
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict
import sys
import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor
from PIL import Image

# Add the root directory to the Python module search path (otherwise can't find src)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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


def _parse_question(category_id, questions_path: Path) -> Tuple[List[V1Prompt], List[V1Question]]:
    # Read the question text and prompts
    with open(str(questions_path), 'r') as file:
        lines = file.read().split('\n')
        q_text = lines[0]
        n_prompt = V1Prompt(category_id=category_id, prompt_id=0, prompt_text=lines[2])
        p_prompt = V1Prompt(category_id=category_id, prompt_id=1, prompt_text=lines[1])
        questions = [V1Question(question_text=q_text, category_id=category_id, positive_prompt_ids=[1])]
        prompts = [n_prompt, p_prompt]

    return prompts, questions


def _extract_image_names(imgs_path: Path) -> List[Path]:
    return [x for x in imgs_path.iterdir() if x.suffix.lower().endswith(".jpg")]


def _generate_category_annotations(
        category_path: Path,
        base_path: Path,
        category_id: int = None,
        category_name: str = ""
) -> Tuple[V1Category, List[V1Annnotation]]:
    if not category_id:
        category_id = int(category_path.stem)

    # This parsing would be more stable if the question.txt file included pointers to directories
    # rather than relying on a static map. This would also be required if we add different types
    # of prompt per category.
    prompts, questions = _parse_question(category_id, category_path / "question.txt")

    category: V1Category = dict(category_id=category_id,
                                category_name=category_name,
                                prompts=prompts,
                                questions=questions)

    prompt_map = {"negative": category["prompts"][0], "positive": category["prompts"][1]}
    images = []

    current_idx = 0
    for prompt_path in sorted(category_path.iterdir()):
        if not prompt_path.is_dir():
            continue

        current_prompt: V1Prompt = prompt_map[prompt_path.stem]
        prompt_annot: List[V1Annnotation] = [
            dict(
                path=x.relative_to(base_path).as_posix(),
                category_id=category["category_id"],
                prompt_id=current_prompt["prompt_id"]
            ) for x in _extract_image_names(prompt_path)
        ]
        images += prompt_annot
        current_idx += 1
    return category, images


def _generate_annotations(data_path: Path):
    assert data_path.exists() and data_path.is_dir()

    images = []
    categories: List[V1Category] = []

    # Iterate over each question ID in the outputs directory
    for category_path in sorted(data_path.iterdir()):
        # Ensure that the question ID is a directory
        if not category_path.is_dir():
            continue
        cat, cat_annots = _generate_category_annotations(category_path, data_path)
        images += cat_annots
        categories.append(cat)

    return {
        "images": images,
        "categories": [c for c in categories]
    }


def create_annotations_json(train_path: Path, val_path: Path, out_path: Path, overwrite=False):
    print(f'train_path: {train_path}')
    train_annots = _generate_annotations(train_path)
    val_annots = _generate_annotations(val_path)

    assert not out_path.exists() or val_path.is_dir()

    with open(str(out_path / "train.json"), "w") as train_file:
        json.dump(train_annots, train_file, indent=2)

    with open(str(out_path / "val.json"), "w") as val_file:
        json.dump(val_annots, val_file, indent=2)


class V1Dataset(CaptchaDataset):
    def __init__(self, img_root: str, annotations_path: str, challenge_size: int = 9, **data_kwargs):
        super().__init__(((data_kwargs)))
        self._img_root_path: Path = DATA_PATH / img_root
        self._annotations_path: Path = DATA_PATH / annotations_path
        with open(str(self._annotations_path), "r") as annots_file:
            self._annotations = json.load(annots_file)
        self._images: List[V1Annnotation] = self._annotations["images"]
        self._categories: List[V1Category] = self._annotations["categories"]
        self._loader = default_loader
        self._challenge_size = challenge_size

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
        question_text = question["question_text"]
        category_id: int = question["category_id"]
        target_prompt_ids: List[int] = question["positive_prompt_ids"]
        im_data = self._sample_images(self.challenge_size, category_id)  # TODO temp changed to 1, also wrapped in [] brackets because it didn't want to make it a list when there was 1 sample. Basically, I had to make it a list so that I can un-make it a list at the end :)
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

        return question_text, img_tensors, targets


if __name__ == "__main__":
    base_path = DATA_PATH / "v1"
    train_path = base_path / "train"
    val_path = base_path / "val"
    out_path = base_path
    create_annotations_json(train_path, val_path, out_path)
