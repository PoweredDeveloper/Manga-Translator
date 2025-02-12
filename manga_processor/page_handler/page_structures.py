import json

from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from enum import Enum, StrEnum
from pathlib import Path
from typing import Sequence

from ..mt_config import FONT_ROBOTO_PATH
from ..mt_logger import logger

from .ocr import LanguageCode, to_language_code

class BoxType(Enum):
    BOX = 0
    EXTENDED_BOX = 1
    MERGED_EXT_BOX = 2
    REFERENCE_BOX = 3

class Box:
    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def as_tuple_xywh(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __str__(self) -> str:
        return f"{self.x1},{self.y1},{self.x2},{self.y2}"

    def __contains__(self, point: tuple[int, int]) -> bool:
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def merge(self, box: "Box") -> "Box":
        x_min = min(self.x1, box.x1)
        y_min = min(self.y1, box.y1)
        x_max = max(self.x2, box.x2)
        y_max = max(self.y2, box.y2)

        return Box(x_min, y_min, x_max, y_max)

    def overlaps(self, other: "Box", threshold: float) -> bool:
        x_overlap = max(0, min(self.x2, other.x2) - max(self.x1, other.x1))
        y_overlap = max(0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = x_overlap * y_overlap
        smaller_area = min(self.area, other.area) or 1

        return intersection / smaller_area > (threshold / 100)

    def overlaps_center(self, other: "Box") -> bool:
        return self.center in other or other.center in self

    def pad(self, amount: int, canvas_size: tuple[int, int]) -> "Box":
        x1_new = max(self.x1 - amount, 0)
        y1_new = max(self.y1 - amount, 0)
        x2_new = min(self.x2 + amount, canvas_size[0])
        y2_new = min(self.y2 + amount, canvas_size[1])

        return Box(x1_new, y1_new, x2_new, y2_new)

    def right_pad(self, amount: int, canvas_size: tuple[int, int]) -> "Box":
        x2_new = min(self.x2 + amount, canvas_size[0])

        return Box(self.x1, self.y1, x2_new, self.y2)

    def scale(self, factor: float) -> "Box":
        x1_new = int(self.x1 * factor)
        y1_new = int(self.y1 * factor)
        x2_new = int(self.x2 * factor)
        y2_new = int(self.y2 * factor)

        return Box(x1_new, y1_new, x2_new, y2_new)

    def translate(self, x: int, y: int) -> "Box":
        return Box(self.x1 + x, self.y1 + y, self.x2 + x, self.y2 + y)

class ReadingOrder(StrEnum):
    AUTO = 'auto'
    MANGA = 'manga'
    COMIC = 'comic'

class MaskerData:
    def __init__(self, json_path: Path, cache_dir: Path, extract_text: bool, show_masks: bool) -> None:
        self.json_path: Path = json_path
        self.cache_dir: Path = cache_dir
        self.extract_text: bool = extract_text
        self.show_masks: bool = show_masks

class MaskResults:
    def __init__(self, image_path: Path, fited: bool, index: int, std_deviation: float, thickness: int | None) -> None:
        self.image_path: Path = image_path
        self.fited: bool = fited
        self.index: int = index
        self.std_deviation: float = std_deviation
        self.thickness: int | None = thickness

class MaskFitting:
    def __init__(self,
                 best_mask: Image.Image,
                 median_color: tuple[int, int, int],
                 mask_coords: tuple[int, int],
                 page_path: Path,
                 std_deviation: float,
                 mask_index: int,
                 thickness: int | None,
                 mask_box: Box,
                 debug_masks: list[Image.Image]
                ) -> None:
        self.best_mask: Image.Image = best_mask
        self.median_color: tuple[int, int, int] = median_color
        self.mask_coords: tuple[int, int] = mask_coords
        self.page_path: Path = page_path
        self.std_deviation: float = std_deviation
        self.mask_index: int = mask_index
        self.thickness: int | None = thickness
        self.mask_box: Box = mask_box
        self.debug_masks: list[Image.Image] = debug_masks
    
    @property
    def results(self) -> MaskResults:
        return MaskResults(
            self.page_path,
            not self.failed,
            self.mask_index,
            self.std_deviation,
            self.thickness,
        )

    @property
    def failed(self) -> bool:
        return self.best_mask is None

    @property
    def mask_data(self) -> tuple[Image.Image, tuple[int, int, int], tuple[int, int]]:
        return self.best_mask, self.median_color, self.mask_coords

    @property
    def noise_mask_data(self) -> tuple[Box, float]:
        return self.mask_box, self.std_deviation

class MaskData:
    def __init__(self, original_path: Path, base_image_path: Path, mask_path: Path, scale: float, boxes_with_stats: Sequence[tuple[Box, float, bool, int | None]]) -> None:
        self.original_path: Path = original_path
        self.base_image_path: Path = base_image_path
        self.mask_path: Path = mask_path
        self.scale: float = scale
        self.boxes_with_stats: Sequence[tuple[Box, float, bool, int | None]] = boxes_with_stats

    @classmethod
    def from_json(cls, json_str: str) -> "MaskData":
        data = json.loads(json_str)
        return cls(
            Path(data["original_path"]),
            Path(data["base_image_path"]),
            Path(data["mask_path"]),
            data["scale"],
            [
                (Box(*box), deviation, failed, thickness)
                for box, deviation, failed, thickness in data["boxes_with_stats"]
            ],
        )

    def to_json(self) -> str:
        data = {
            "original_path": str(self.original_path),
            "base_image_path": str(self.base_image_path),
            "mask_path": str(self.mask_path),
            "scale": self.scale,
            "boxes_with_stats": [
                (box.as_tuple, deviation, failed, thickness)
                for box, deviation, failed, thickness in self.boxes_with_stats
            ],
        }
        return json.dumps(data, indent=4)


class PageData:
    def __init__(self,
                 image_path: str,
                 mask_path: str,
                 original_path: str,
                 scale: float,
                 box_language: list[LanguageCode | None],
                 boxes: list[Box],
                 extended_boxes: list[Box],
                 merged_extended_boxes: list[Box],
                 reference_boxes: list[Box],
                 _image_size: tuple[int, int] = (None)) -> None:
        self.image_path: str = image_path
        self.mask_path: str = mask_path
        self.original_path: str = original_path
        self.scale: float = scale
        self.box_language: list[LanguageCode | None] = box_language
        self.boxes: list[Box] = boxes
        self.extended_boxes: list[Box] = extended_boxes
        self.merged_extended_boxes: list[Box] = merged_extended_boxes
        self.reference_boxes: list[Box] = reference_boxes
        self._image_size: tuple[int, int] = _image_size

    @classmethod
    def from_json(cls, json_str: str) -> "PageData":
        json_data = json.loads(json_str)
        return cls(
            json_data["image_path"],
            json_data["mask_path"],
            json_data["original_path"],
            json_data["scale"],
            [to_language_code(lang) for lang in json_data["box_language"]],
            [Box(*b) for b in json_data["boxes"]],
            [Box(*b) for b in json_data["extended_boxes"]],
            [Box(*b) for b in json_data["merged_extended_boxes"]],
            [Box(*b) for b in json_data["reference_boxes"]],
        )

    def to_json(self) -> str:
        data = {
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "original_path": self.original_path,
            "scale": self.scale,
            "box_language": [str(lang) for lang in self.box_language],
            "boxes": [b.as_tuple for b in self.boxes],
            "extended_boxes": [b.as_tuple for b in self.extended_boxes],
            "merged_extended_boxes": [b.as_tuple for b in self.merged_extended_boxes],
            "reference_boxes": [b.as_tuple for b in self.reference_boxes],
        }
        return json.dumps(data, indent=4)

    @property
    def image_size(self) -> tuple[int, int]:
        if self._image_size is None:
            self._image_size = Image.open(self.image_path).size
        return self._image_size

    def boxes_from_type(self, box_type: BoxType) -> list[Box]:
        match box_type:
            case BoxType.BOX:
                return self.boxes
            case BoxType.EXTENDED_BOX:
                return self.extended_boxes
            case BoxType.MERGED_EXT_BOX:
                return self.merged_extended_boxes
            case BoxType.REFERENCE_BOX:
                return self.reference_boxes
            case _:
                raise ValueError("Invalid box type.")

    def grow_boxes(self, padding: int, box_type: BoxType) -> None:
        boxes = self.boxes_from_type(box_type)
        for i, box in enumerate(boxes):
            boxes[i] = box.pad(padding, self.image_size)

    def right_pad_boxes(self, padding: int, box_type: BoxType) -> None:
        boxes = self.boxes_from_type(box_type)
        for i, box in enumerate(boxes):
            boxes[i] = box.right_pad(padding, self.image_size)

    def visualize(self, image_path: Path | str, output_path: Path | str) -> None:
        image = Image.open(image_path)
        font_path = FONT_ROBOTO_PATH
        logger.debug(f"Loading included font from {FONT_ROBOTO_PATH}")
        font_size = int(image.size[0] / 50) + 5

        FILL_ALPHA = 48
        fill_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(fill_layer)
        for box in self.reference_boxes:
            draw.rectangle(box.as_tuple, fill="deepskyblue")
        for box in self.merged_extended_boxes:
            draw.rectangle(box.as_tuple, fill="mediumorchid")
        for box in self.extended_boxes:
            draw.rectangle(box.as_tuple, fill="crimson")
        for box in self.boxes:
            draw.rectangle(box.as_tuple, fill="mediumaquamarine")

        alpha = fill_layer.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(FILL_ALPHA / 255)
        fill_layer.putalpha(alpha)

        image = Image.alpha_composite(image.convert("RGBA"), fill_layer)
        draw = ImageDraw.Draw(image)

        for index, box in enumerate(self.boxes):
            draw.rectangle(box.as_tuple, outline="mediumaquamarine")
            draw.text(
                (box.x1 + 4, box.y1),
                str(index + 1),
                fill="mediumaquamarine",
                font=ImageFont.truetype(font_path, font_size),
                stroke_fill="white",
                stroke_width=3,
            )

        for box in self.extended_boxes:
            draw.rectangle(box.as_tuple, outline="crimson")
        for box in self.merged_extended_boxes:
            draw.rectangle(box.as_tuple, outline="mediumorchid")
        for box in self.reference_boxes:
            draw.rectangle(box.as_tuple, outline="deepskyblue")

        image.save(output_path)

    def make_box_mask(self, image_size: tuple[int, int], box_type: BoxType) -> Image:
        box_mask = Image.new("1", image_size, (0,))
        draw = ImageDraw.Draw(box_mask)
        boxes = self.boxes_from_type(box_type)
        for box in boxes:
            draw.rectangle(box.as_tuple, fill=(1,))
        return box_mask

    def resolve_total_overlaps(self) -> None:
        merge_queue = self.boxes.copy()
        merged_boxes = []
        while merge_queue:
            box = merge_queue.pop(0)
            overlapping_boxes = [b for b in merge_queue if box.overlaps_center(b)]
            for b in overlapping_boxes:
                box = box.merge(b)
                merge_queue.remove(b)
            merged_boxes.append(box)

        self.boxes = merged_boxes

    def resolve_overlaps(self, from_type: BoxType, to_type: BoxType, threshold: float) -> None:
        merge_queue = set(self.boxes_from_type(from_type))
        merged_boxes = []
        while merge_queue:
            box = merge_queue.pop()
            overlapping_boxes = [b for b in merge_queue if box.overlaps(b, threshold)]
            for b in overlapping_boxes:
                box = box.merge(b)
                merge_queue.remove(b)
            merged_boxes.append(box)

        boxes_reference = self.boxes_from_type(to_type)
        boxes_reference.clear()
        boxes_reference.extend(merged_boxes)
