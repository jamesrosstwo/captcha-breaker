# Add the root directory to the Python module search path (otherwise can't find src)
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.dataset.v1 import V1Prompt, V1Question, V1Category, V1Annnotation
from src.definitions import DATA_PATH

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


def create_annotations_json(train_path: Path, val_path: Path, test_path: Path, out_path: Path, overwrite=False):
    print(f'train_path: {train_path}')
    train_annots = _generate_annotations(train_path)
    val_annots = _generate_annotations(val_path)
    test_annots = _generate_annotations(test_path)

    assert not out_path.exists() or val_path.is_dir()

    with open(str(out_path / "train.json"), "w") as train_file:
        json.dump(train_annots, train_file, indent=2)

    with open(str(out_path / "val.json"), "w") as val_file:
        json.dump(val_annots, val_file, indent=2)

    with open(str(out_path / "test.json"), "w") as val_file:
        json.dump(test_annots, val_file, indent=2)


if __name__ == "__main__":
    base_path = DATA_PATH / "v1"
    train_path = base_path / "train"
    val_path = base_path / "val"
    test_path = base_path / "test"
    out_path = base_path
    create_annotations_json(train_path, val_path, test_path, out_path)