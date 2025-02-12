import re
import cv2
import tqdm
import json
import time
import numpy
import torch
import scipy
import shutil
import colorsys
import multiprocessing as mp

from copy import copy
from pathlib import Path
from itertools import cycle
from math import floor, ceil
from collections import Counter
from typing import NewType, Any, Sequence, Generator, TypeVar, Iterable
from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageFilter, ImageColor

from ..mt_logger import logger
from ..mt_paths import MT_PathGen
from ..utils.io import NumpyJSONEncoder
from ..mt_config import FONT_ROBOTO_PATH

from ..libs.comic_text_detector.inference import TextDetector
from ..libs.comic_text_detector.utils.io_utils import imwrite
from ..libs.comic_text_detector.utils.textmask import REFINEMASK_ANNOTATION

from .ocr import create_ocr_engine, OCR, LanguageCode, available_languages, language_names, OCRData, OCREngine
from .page_structures import Box, PageData, BoxType, ReadingOrder, MaskerData, MaskResults, MaskFitting, MaskData

class BlankBubbleException(Exception): pass

class MT_ImageProcessor:
    def __init__(self, cache_path: Path, model_path: Path) -> None:
        if not cache_path.exists(): 
            logger.critical('Cache folder doesn\'t exist')
            return
        
        self.cache_path = cache_path

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            logger.warning('CUDA isn\'t available. CPU device was enabled.')

        if not model_path.exists():
            logger.critical('Model path not found')

        self.model_path = model_path
        self.model = TextDetector(model_path = str(self.model_path), input_size = 1024, device = self.device)

    def clean(self, image_list: list[Path], output_direction: Path, generate_debug_boxes: bool = False) -> None:
        logger.info('Starting cleaning images')
        self.generate_mask_data(image_list, generate_debug_boxes)

        logger.info('Start Pre-processing')
        self.preprocess_images(True, LanguageCode.detect_box, False, OCR.AUTO, generate_debug_boxes)

        logger.info('Running masker')
        self.mask_images(0, True, True)

    def mask_images(self, max_threads: int = 0, extract_text: bool = True, show_masks: bool = False) -> None:
        json_files = self.cache_path.glob('*#clean.json')
        data: list[MaskerData] = [MaskerData(json_file, self.cache_path, extract_text, show_masks) for json_file in json_files]

        masking_results: list[MaskResults] = []

        if max_threads == 0:
            max_threads = mp.cpu_count()
        pool_size = min(max_threads, len(data))

        if pool_size > 1:
            with mp.Pool(processes = pool_size) as pool:
                for results in tqdm.tqdm(pool.imap(self._mask_page, data), total = len(data)):
                    masking_results.extend(results)
        else:
            for masking_data in tqdm.tqdm(data):
                result = self._mask_page(masking_data)
                masking_results.extend(result)

        if masking_results:
            logger.info('==== There should be Masking results ====')

    def _mask_page(self, masking_data: MaskerData, debug_mask_color: tuple[int, int, int, int] = (108, 30, 240, 127)) -> Sequence[MaskResults]:
        mask_max_standard_deviation = 15.0
        page_data = PageData.from_json(masking_data.json_path.read_text(encoding = 'utf-8'))

        original_path = Path(page_data.original_path)
        path = MT_PathGen(original_path, self.cache_path, masking_data.json_path)

        def save_mask(image: Image.Image, path: Path) -> None:
            if masking_data.show_masks:
                image.save(path)

        base_image = Image.open(page_data.image_path)
        box_mask = page_data.make_box_mask(base_image.size, BoxType.EXTENDED_BOX)

        save_mask(box_mask, path.box_mask)

        mask = Image.open(page_data.mask_path)
        mask = mask.convert('1', dither = Image.NONE)
        cut_mask = Image.composite(mask, Image.new("1", mask.size, 0), box_mask)

        save_mask(cut_mask, path.cut_mask)

        mask_fitments: list[MaskFitting] = [
            self._pick_mask(
                base_image = base_image,
                cut_mask = cut_mask,
                box_mask = box_mask,
                masking_box = masking_box,
                reference_box = reference_box,
                page_path = Path(original_path),
                mask_max_standard_deviation = mask_max_standard_deviation
            )
            for masking_box, reference_box in zip(page_data.merged_extended_boxes, page_data.reference_boxes)
        ]

        mask_fitments = [mask for mask in mask_fitments if mask is not None]
        results: Sequence[MaskResults] = tuple(mask.results for mask in mask_fitments)
        best_masks = [mask for mask in mask_fitments if not mask.failed]

        combined_mask = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        for mask_fitment in best_masks:
            mask, color, coords = mask_fitment.mask_data

            array_1 = numpy.array(mask)
            if len(color) == 3:
                color = (color[0], color[1], color[2], 255)
            array_rgba = numpy.zeros((*array_1.shape, 4), dtype = numpy.uint8)
            array_rgba[array_1 != 0] = color

            mask = Image.fromarray(array_rgba)
            combined_mask.alpha_composite(mask, coords)

        if masking_data.show_masks:
            self._visualize_mask_fitments(base_image, mask_fitments, path.mask_fitments)
            combined_mask_debug = self._apply_debug_filter_to_mask(combined_mask, debug_mask_color)
            base_image_copy = base_image.copy()
            base_image_copy.paste(combined_mask_debug, (0, 0), combined_mask_debug)
            save_mask(base_image_copy, path.masked)
            self._visualize_standard_deviations(base_image, mask_fitments, path.std_deviations, mask_max_standard_deviation)

        combined_mask.save(path.combined_mask)
        
        boxes_with_deviation = [
            (m.noise_mask_data[0], m.noise_mask_data[1], m.failed, m.thickness)
            for m in mask_fitments
        ]

        mask_data = MaskData(Path(page_data.original_path), Path(page_data.image_path), path.combined_mask, page_data.scale, boxes_with_deviation)
        path.mask_data_json.write_text(mask_data.to_json(), encoding="utf-8")

        if page_data.scale != 1:
            cleaned_image = Image.open(page_data.original_path)
            combined_mask = combined_mask.resize(cleaned_image.size, Image.Resampling.NEAREST)
        else:
            cleaned_image = base_image.copy()

        cleaned_image.paste(combined_mask, (0, 0), combined_mask)
        cleaned_image.save(path.clean)

        if masking_data.extract_text:
            logger.info(f'Extracting text from {original_path}')
            base_image_new = Image.open(original_path)
            text_image = self._extract_text(base_image_new, combined_mask)
            text_image.save(path.text)

        return results
    
    def _extract_text(self, base_image: Image.Image, mask: Image.Image) -> None:
        text_image = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
        if mask.size != base_image.size:
            mask = mask.resize(base_image.size, Image.Resampling.NEAREST)
        text_image.paste(base_image, (0, 0), mask)
        return text_image

    def _visualize_mask_fitments(
        self, base_image: Image.Image, mask_fitments: list[MaskFitting], output_path: Path
    ) -> None:
        masks = []
        if mask_fitments:
            max_masks = max(len(mask_fitment.debug_masks) for mask_fitment in mask_fitments)
            for index in range(max_masks):
                mask_list = []
                for mask_fitment in mask_fitments:
                    if index >= len(mask_fitment.debug_masks):
                        continue
                    mask_tuple = (mask_fitment.debug_masks[index], mask_fitment.mask_coords)
                    mask_list.append(mask_tuple)
            
                base_mask = Image.new("1", base_image.size, 0)
                for masked, pos in mask_list:
                    base_mask.paste(masked, pos, masked)
                mask = base_mask

                masks.append(mask)
        else:
            logger.warning(f"No masks generated for {output_path.name}")

        base_image = base_image.copy()

        color_tuple = (
            ImageColor.getcolor('#ffc100', 'RGB'),
            ImageColor.getcolor('#5aff54', 'RGB'),
            ImageColor.getcolor('#24ff92', 'RGB'),
            ImageColor.getcolor('#00dbff', 'RGB'),
            ImageColor.getcolor('#0049ff', 'RGB'),
            ImageColor.getcolor('#ff0076', 'RGB'),
            ImageColor.getcolor('#ff0014', 'RGB'),
            ImageColor.getcolor('#ff7300', 'RGB'),
        )
        colors = cycle(color_tuple)
        colored_masks = (self._apply_debug_filter_to_mask(mask, next(colors)) for mask in reversed(masks))

        combined_mask = Image.new("RGBA", base_image.size)
        for mask in colored_masks:
            mask = mask.convert("RGBA")
            combined_mask.alpha_composite(mask)
        alpha_mask = combined_mask.split()[3]
        alpha_mask = alpha_mask.point(lambda x: min(x, 153))
        base_image.paste(combined_mask, (0, 0), alpha_mask)
        base_image.save(output_path)

    def _visualize_standard_deviations(
        self,
        base_image: Image.Image,
        mask_fitments: list[MaskFitting],
        output_path: Path,
        mask_max_standard_deviation: float
    ) -> None:
        text_offset_x: int = 5
        text_offset_y: int = 4

        base_image = base_image.copy()

        font_path = FONT_ROBOTO_PATH
        logger.debug(f"Loading included font from {font_path}")
        font_size = int(base_image.size[0] / 50) + 5
        font = ImageFont.truetype(font_path, font_size)

        for fitment in mask_fitments:
            fitment: MaskFitting
            text_x = fitment.mask_box.x1
            text_y = fitment.mask_box.y1
            if fitment.best_mask is not None:
                mask = fitment.best_mask
                color_offset = (1 - fitment.std_deviation / mask_max_standard_deviation) ** 4

                hue = color_offset * 265 / 360
                r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.8)
                color = (int(r * 255), int(g * 255), int(b * 255), int(0.7 * 255))

                mask = self._apply_debug_filter_to_mask(mask, color)
                base_image.paste(mask, fitment.mask_coords, mask)

                std_deviation = fitment.std_deviation
                thickness = fitment.thickness
                text = f"\u03C3={std_deviation:.2f}"
                draw = ImageDraw.Draw(base_image)
                draw.rectangle(fitment.mask_box.as_tuple, outline='#00fa9a', width=3)
                draw.text(
                    (text_x + text_offset_x, text_y + text_offset_y),
                    text,
                    font=font,
                    fill="black",
                    stroke_width=3,
                    stroke_fill="white",
                )
                if thickness is not None:
                    text = f"{thickness}px"
                    draw.text(
                        (text_x + text_offset_x, text_y + text_offset_y + font_size + 2),
                        text,
                        font=font,
                        fill="black",
                        stroke_width=3,
                        stroke_fill="white",
                    )
            else:
                draw = ImageDraw.Draw(base_image)
                draw.line(
                    (
                        fitment.mask_box.x1 + 1,
                        fitment.mask_box.y1 + 1,
                        fitment.mask_box.x2 - 1,
                        fitment.mask_box.y2 - 1,
                    ),
                    fill="#dc143c",
                    width=3,
                )
                draw.line(
                    (
                        fitment.mask_box.x1 + 1,
                        fitment.mask_box.y2 - 1,
                        fitment.mask_box.x2 - 1,
                        fitment.mask_box.y1 + 1,
                    ),
                    fill="#dc143c",
                    width=3,
                )
                draw.rectangle(fitment.mask_box.as_tuple, outline='#dc143c', width=3)
                std_deviation = fitment.std_deviation
                text = f"\u03C3={std_deviation:.2f}"
                draw.text(
                    (text_x + text_offset_x, text_y + text_offset_y),
                    text,
                    font=font,
                    fill="#900",
                    stroke_width=3,
                    stroke_fill="white",
                )

        base_image.save(output_path)

    def _convert_mask_to_rgba(
        self,
        mask: Image.Image,
        color: tuple[int, int, int] | tuple[int, int, int, int] = (255, 255, 255, 255),
    ) -> Image.Image:
        array_1 = numpy.array(mask)
        if len(color) == 3:
            color = (color[0], color[1], color[2], 255)
        array_rgba = numpy.zeros((*array_1.shape, 4), dtype=numpy.uint8)
        array_rgba[array_1 != 0] = color
        return Image.fromarray(array_rgba)

    def _apply_debug_filter_to_mask(
        self, img: Image.Image, color: tuple[int, int, int, int] = (108, 30, 240, 127)
    ) -> Image.Image:
        if img.mode == "1":
            return self._convert_mask_to_rgba(img, color)
        elif img.mode == "RGBA":
            array_rgba = numpy.array(img)
            array_rgba[array_rgba[:, :, 3] != 0] = color
            return Image.fromarray(array_rgba)
        else:
            raise ValueError(f"Unknown mode: {img.mode}")

    def _pick_mask(self,
                   base_image: Image.Image,
                   cut_mask: Image.Image,
                   box_mask: Image.Image,
                   masking_box: Box,
                   reference_box: Box,
                   page_path: Path,
                   mask_growth_step_pixels: int = 2,
                   mask_growth_steps: int = 11,
                   min_mask_thickness: int = 4,
                   allow_colored_masks: bool = True,
                   off_white_max_threshold: int = 240,
                   mask_improvement_threshold: float = 0.1,
                   mask_max_standard_deviation: float = 15.0,
                   mask_selection_fast: bool = False
                  ) -> MaskFitting | None:
        x_offset = masking_box.x1 - reference_box.x1
        y_offset = masking_box.y1 - reference_box.y1

        base_image = base_image.crop(reference_box.as_tuple)
        
        cut_mask = cut_mask.crop(masking_box.as_tuple)
        if base_image.size is not None:
            padded_mask = Image.new("1", base_image.size, 0)
            padded_mask.paste(cut_mask, (x_offset, y_offset))
            cut_mask = padded_mask

        if cut_mask.getbbox() is None:
            logger.warning(f"Found an empty mask. Image: {page_path.name} Masking box: {masking_box}, x offset: {x_offset}, y offset: {y_offset}. Skipping")
            return None
        
        box_mask = box_mask.crop(masking_box.as_tuple)
        if base_image.size is not None:
            padded_mask = Image.new("1", base_image.size, 0)
            padded_mask.paste(box_mask, (x_offset, y_offset))
            box_mask = padded_mask

        box_mask_thick: tuple[Image.Image, int | None] = (box_mask, None)
        mask_gen = self._convolate_mask(cut_mask, mask_growth_step_pixels, mask_growth_steps, min_mask_thickness)

        T = TypeVar("T")
        def generator_with_first(generator: Iterable[T], first: T) -> Generator[T, None, None]:
            yield first
            yield from generator

        def generator_with_last(generator: Iterable[T], last: T) -> Generator[T, None, None]:
            yield from generator
            yield last

        if mask_selection_fast:
            mask_stream = generator_with_first(mask_gen, box_mask_thick)
        else:
            mask_stream = generator_with_last(mask_gen, box_mask_thick)

        border_deviations: list[tuple[float, tuple[int, int, int]]] = []
        masks = []
        thicknesses = []
        for mask, thickness in mask_stream:
            try:
                masks.append(mask)
                thicknesses.append(thickness)
                current_deviation = self._deviate_border(
                    base_image, mask, off_white_max_threshold, allow_colored_masks
                )
                border_deviations.append(current_deviation)
                if mask_selection_fast and current_deviation[0] == 0:
                    break
            except BlankBubbleException:
                return None
                
        best_mask = None
        lowest_border_deviation = None
        lowest_deviation_color: int | tuple[int, int, int] | None = None
        chosen_thickness = None
        for i, border_deviation in enumerate(border_deviations):
            mask_deviation, mask_color = border_deviation
            if i == 0 or mask_deviation <= (
                lowest_border_deviation * (1 - mask_improvement_threshold)
            ):
                lowest_border_deviation = mask_deviation
                lowest_deviation_color = mask_color
                best_mask = masks[i]
                chosen_thickness = thicknesses[i]

        if lowest_border_deviation > mask_max_standard_deviation:
            return MaskFitting(best_mask = None,
                median_color = lowest_deviation_color,
                mask_coords = (reference_box.x1, reference_box.y1),
                page_path = page_path,
                mask_index = masks.index(best_mask),
                std_deviation = lowest_border_deviation,
                thickness = chosen_thickness,
                mask_box = masking_box,
                debug_masks = masks
            )
        return MaskFitting(best_mask = best_mask,
            median_color = lowest_deviation_color,
            mask_coords = (reference_box.x1, reference_box.y1),
            page_path = page_path,
            mask_index = masks.index(best_mask),
            std_deviation = lowest_border_deviation,
            thickness = chosen_thickness,
            mask_box = masking_box,
            debug_masks = masks
        )

    def _deviate_border(self, base_image: Image.Image, mask: Image.Image, off_white_threshold: int, allow_color: bool) -> tuple[float, tuple[int, int, int]]:
        if not allow_color:
            base_image = base_image.convert('L')
        
        mask = mask.filter(ImageFilter.FIND_EDGES)
        base_data = numpy.array(base_image)
        mask_data = numpy.array(mask)

        border_pixels = base_data[mask_data == 1]
        num_pixels = len(border_pixels)
        if num_pixels == 0:
            raise BlankBubbleException
        
        def color_std(colors: numpy.ndarray) -> float:
            colors = colors.astype(numpy.float64)
            mean_color = numpy.mean(colors, axis = 0)
            distances = numpy.linalg.norm(colors - mean_color, axis = 1)
            std_dev = numpy.std(distances, ddof = 1)
            return std_dev
        
        def heuristic_median_color(colors: numpy.ndarray) -> numpy.ndarray | None:
            color_tuples = [tuple(color) for color in colors]

            color_counts = Counter(color_tuples)
            total_colors = len(colors)

            for color, count in color_counts.items():
                if count > total_colors / 2:
                    median_color = numpy.array(color)
                    return median_color

            return None
        
        def geometric_median(points: numpy.ndarray, epsilon=1e-5, max_iterations=500) -> numpy.ndarray:
            median = numpy.mean(points, axis=0)

            for _ in range(max_iterations):
                distances = numpy.linalg.norm(points - median, axis=1)
                non_zero_distances = distances > 0

                if not numpy.any(non_zero_distances):
                    return median

                inverse_distances = 1 / distances[non_zero_distances]
                total_inverse_distance = numpy.sum(inverse_distances)
                weights = inverse_distances / total_inverse_distance

                new_median = numpy.sum(weights[:, numpy.newaxis] * points[non_zero_distances], axis=0)

                median_shift = numpy.linalg.norm(new_median - median)
                if median_shift < epsilon:
                    return new_median
                median = new_median

            return median

        
        if len(border_pixels.shape) == 1:
            std = float(numpy.std(border_pixels))
            median_color = int(numpy.median(border_pixels))
            median_color = (median_color, median_color, median_color)
        else:
            std = color_std(border_pixels)
            median_color = heuristic_median_color(border_pixels)
            if median_color is None:
                median_color = geometric_median(border_pixels)
            median_color = tuple(int(color) for color in median_color)

        if min(median_color) > off_white_threshold:
            median_color = (255, 255, 255)

        return std, median_color

    def _convolate_mask(self, mask: Image.Image, growth_step: int, steps: int, min_thickness: int) -> Generator[tuple[Image.Image, int | None], None, None]:
        padding_for_kernel = max(min_thickness, growth_step) * 2
        mask = mask.convert("L")
        mask_array = numpy.array(mask, dtype = numpy.uint8)
        kernel = self._make_growth_kernel(growth_step)
        kernel_first = self._make_growth_kernel(min_thickness)

        padded_mask = numpy.pad(
            mask_array,
            ((padding_for_kernel, padding_for_kernel), (padding_for_kernel, padding_for_kernel)),
            mode="edge",
        )
        padded_mask = scipy.signal.convolve2d(padded_mask, kernel_first, mode = "same")
        cropped_mask = padded_mask[
            padding_for_kernel:-padding_for_kernel, padding_for_kernel:-padding_for_kernel
        ]
        yield Image.fromarray(numpy.where(cropped_mask > 0, 255, 0).astype(numpy.uint8)).convert("1"), min_thickness

        for index in range(steps - 1):
            padded_mask = scipy.signal.convolve2d(padded_mask, kernel, mode="same")
            cropped_mask = padded_mask[
                padding_for_kernel:-padding_for_kernel, padding_for_kernel:-padding_for_kernel
            ]
            yield Image.fromarray(numpy.where(cropped_mask > 0, 255, 0).astype(numpy.uint8)).convert("1"), min_thickness + (index + 1) * growth_step

    def _make_growth_kernel(self, thickness: int) -> numpy.ndarray:
        diameter = thickness * 2 + 1

        if diameter <= 5:
            kernel = numpy.ones((diameter, diameter), dtype = numpy.float64)
            kernel[0, 0] = 0
            kernel[0, -1] = 0
            kernel[-1, 0] = 0
            kernel[-1, -1] = 0
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
            kernel = kernel.astype(numpy.float64)

        return kernel

    def preprocess_images(self,
                          ocr_enabled: bool,
                          ocr_language: LanguageCode.detect_box,
                          use_tesseract: bool,
                          ocr_model: OCR,
                          cache_masks: bool) -> None:
        time.sleep(0.1)
        ocr_engine = create_ocr_engine(use_tesseract, ocr_model) if ocr_enabled else None

        if ocr_language not in (LanguageCode.detect_box, LanguageCode.detect_page):
            if ocr_language not in available_languages():
                language = language_names[ocr_language]
                logger.warning(f'"{language}" is not supported by any of your OCR engines')

        ocr_data: list = [OCRData]
        for json_file in tqdm.tqdm(list(self.cache_path.glob('*.json'))):
            ocr_data_of_page = self._postprocess_json_file(
                json_file,
                cache_masks,
                ocr_language,
                ocr_engine = ocr_engine
            )

            if ocr_data_of_page is not None:
                ocr_data.append(ocr_data_of_page)

        if ocr_data:
            # TODO: OCR data info
            logger.info('==== There should be OCR data ====')
    
    def _postprocess_json_file(self,
                               json_path: Path,
                               cache_masks: bool,
                               ocr_language: LanguageCode.detect_box,
                               ocr_engine: OCREngine | None = None,
                               reading_order: ReadingOrder = ReadingOrder.AUTO,
                               do_ocr: bool = False,
                               strict_language: bool = False,
                               box_minimum_size: int = 20 * 20,
                               box_minimum_size_containing_text: int = 200 * 200,
                               ocr_max_size: int = 30 * 100,
                               box_padding_start: int = 2,
                               box_right_padding_start: int = 3,
                               box_extending_padding: int = 5,
                               box_extending_right_padding: int = 5,
                               box_reference_padding: int = 20,
                               box_overlap_threshold: float = 20.0
                              ) -> OCRData | None:
        logger.info(f'Starting preprocessing json: {json_path}')
        if not json_path.name.endswith('#raw.json'): return None

        json_data = json.loads(json_path.read_text(encoding = 'utf-8'))

        image_path: str = json_data['image_path']
        mask_path: str = json_data['mask_path']
        scale: float = json_data['scale']
        base_path: str = json_data['base_path']

        boxes: list[Box] = []
        box_languages: list[LanguageCode | None] = []

        path = MT_PathGen(Path(base_path), Path(mask_path).parent, Path(mask_path))

        if ocr_language in (LanguageCode.detect_box, LanguageCode.detect_page):
            for data in json_data['blk_list']:
                if data['language'] == 'ja':
                    data['language'] = LanguageCode.jpn
                elif data['language'] == 'eng':
                    data['language'] = LanguageCode.eng
                elif data["language"] == "unknown":
                    data["language"] = None
                else:
                    logger.warning(f'Unknown language code: {data['language']} in a box')
                    data['language'] = None
        else:
            for data in json_data['blk_list']:
                data['language'] = ocr_language

        for data in json_data['blk_list']:
            if do_ocr and data['language'] is None and strict_language: continue
            box = Box(*data['xyxy'])

            if box.area < box_minimum_size: continue
            if data['language'] is None and box.area < box_minimum_size_containing_text: continue

            box_languages.append(data['language'])
            boxes.append(box)

        # Detect page language
        box_langs_cleared = [lang for lang in box_languages if lang is not None]
        page_language: LanguageCode | None = (Counter(box_langs_cleared).most_common(1)[0][0] if box_langs_cleared else None)

        if ocr_language == LanguageCode.detect_page:
            logger.info(f'Detected languages: {page_language} in page: {base_path}')
            box_languages = [page_language] * len(boxes)

        page_data = PageData(image_path, mask_path, base_path, scale, box_languages, boxes, [], [], [])
        page_data.resolve_total_overlaps()
        page_data.grow_boxes(box_padding_start, BoxType.BOX)
        page_data.right_pad_boxes(box_right_padding_start, BoxType.BOX)

        x_factor = -0.4
        y_factor = 1

        languages_right_to_left_order: set[LanguageCode] = {
            LanguageCode.jpn,
            LanguageCode.chi_sim,
            LanguageCode.chi_tra,
            LanguageCode.ara,
            LanguageCode.fas,
            LanguageCode.heb,
        }

        if (reading_order == ReadingOrder.COMIC) or (page_language not in languages_right_to_left_order and reading_order == ReadingOrder.AUTO):
            x_factor *= -1

        if len(page_data.boxes) > 1:
            page_data.boxes, page_data.box_language = zip(
                *sorted(
                    zip(page_data.boxes, page_data.box_language),
                    key = lambda x: x_factor * x[0].x1 + y_factor * x[0].y1
                )
            )

        page_data.boxes = list(page_data.boxes)
        page_data.box_language = list(page_data.box_language)

        if cache_masks:
            page_data.visualize(page_data.image_path, path.boxes)

        data: OCRData | None = None
        if ocr_engine is not None:
            page_data, data = self._check_boxes(page_data, ocr_engine, ocr_max_size)

        page_data.extended_boxes = copy(page_data.boxes)
        page_data.grow_boxes(box_extending_padding, BoxType.EXTENDED_BOX)
        page_data.right_pad_boxes(box_extending_right_padding, BoxType.EXTENDED_BOX)

        page_data.resolve_overlaps(
            from_type = BoxType.EXTENDED_BOX,
            to_type = BoxType.MERGED_EXT_BOX,
            threshold = box_overlap_threshold,
        )

        page_data.reference_boxes = copy(page_data.merged_extended_boxes)
        page_data.grow_boxes(box_reference_padding, BoxType.REFERENCE_BOX)

        json_out_path = path.clean_json
        json_out_path.write_text(page_data.to_json(), encoding="utf-8")

        if cache_masks:
            page_data.visualize(page_data.image_path, path.final_boxes)

        return data

    def _check_boxes(self, page_data: PageData, ocr_engine: OCREngine, max_box_size: int) -> tuple[PageData, OCRData]:
        check_pattern = "[～．ー！？０-９~.!?0-9-]*"

        base_image = Image.open(page_data.image_path)
        scale: float = page_data.scale
        languages: list[LanguageCode | None] = page_data.box_language
        boxes: list[Box, None] = page_data.boxes

        candidate_small_bubble_indices = [i for i, box in enumerate(boxes) if box.area < max_box_size]
        if not candidate_small_bubble_indices:
            return page_data, OCRData(Path(page_data.original_path), len(boxes), (), (), ())
        
        box_sizes = []
        discarded_box_sizes = []
        discarded_box_texts: list[tuple[str, Box]] = []
        for i in candidate_small_bubble_indices:
            box = boxes[i]
            lang = languages[i]
            cutout = base_image.crop(box.as_tuple)
            ocr = ocr_engine(lang)
            text = ocr(cutout)
            remove = re.fullmatch(check_pattern, text, re.DOTALL)
            box_sizes.append(box.area)
            if remove:
                discarded_box_texts.append((text, box.scale(1 / scale)))
                discarded_box_sizes.append(box.area)
                boxes[i] = None
                languages[i] = None

        page_data.boxes = [box for box in boxes if box is not None]
        page_data.box_language = [lang for lang in languages if lang is not None]

        return (
            page_data,
            OCRData(
                Path(page_data.original_path),
                len(boxes),
                box_sizes,
                discarded_box_sizes,
                discarded_box_texts,
            ),
        )

    def denoise_image(self, json_path: Path, boxes_with_stats) -> tuple[Sequence[float], Path]:
        mask_data = json_path.read_text(encoding = 'utf-8')
        mask_image = Image.open(mask_data['mask_path'])
        base_path = Path(mask_data['base_path'])
        path = MT_PathGen(base_path, self.cache_path, json_path)

        def save_mask(image, path: Path) -> None:
            image.save(path)

        clear_image = Image.open(mask_data['base_path'])
        if clear_image.mode == '1':
            logger.info(f'Skipping denoising (1-bit image): {base_path}')
            shutil.copyfile(path.clean, path.clean_denoised)
            noise_mask = Image.new('LA', clear_image.size, (0, 0))
            save_mask(noise_mask, path.noise_mask)
            return (tuple(), base_path)
        
        clear_image = clear_image.convert('RGB')
        scale_factor = 1.0
        if clear_image.size != mask_image.size:
            scale_factor = clear_image.size[0] / mask_image.size[0]
            mask_image = mask_image.resize(clear_image.size, resample = Image.Resampling.NEAREST)

        clear_image.paste(mask_image, (0, 0), mask_image)
        base_path: Path = mask_data['base_path']

        noise_min_standard_deviation = 0.25
        boxes_to_denoise: list[tuple[int, int, int, int]] = [
            box
            for box, deviation, failed, _ in boxes_with_stats
            if not failed and deviation > noise_min_standard_deviation
        ]


        return (tuple(), base_path)

    def generate_mask_data(self, image_list: list[Path], raw_boxes: bool = False) -> None:
        processes_amount = min(1, len(image_list))

        if processes_amount > 1:
            mp.freeze_support()
            mp.set_start_method('spawn')

            with mp.Pool(processes_amount) as pool:
                batches = [list() for _ in range(processes_amount)]
                for i, image_path in enumerate(image_list):
                    batches[i % processes_amount].append(image_path)

                for _ in tqdm.tqdm(pool.imap_unordered(self._generate_mask_batch, [batches, raw_boxes]), total = len(batches)): pass
        
        else:
            for _, image_path in enumerate(tqdm.tqdm(image_list)):
                self.generate_single_mask(image_path, raw_boxes)

    def _generate_mask_batch(self, batch: list[list], raw_boxes: bool = False) -> None:
        for image_path in batch:
            self.generate_single_mask(image_path, raw_boxes)

        if self.device == 'cuda':
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def generate_single_mask(self,
                             img_path: Path,
                             raw_boxes: bool = False,
                             height_lower_target: int = 1000,
                             height_upper_target: int = 4000
                            ) -> None:
        if not img_path.exists():
            logger.error('Page path not found. Skipping..')
            return
        
        path = MT_PathGen(img_path, self.cache_path)
        
        image: numpy.ndarray = cv2.imdecode(numpy.fromfile(str(img_path), dtype = numpy.uint8,), cv2.IMREAD_COLOR)
        image, scale = self.resize_image_to(image, height_lower_target, height_upper_target)
        imwrite(str(path.image), image)

        _, mask_refined, blk_list = self.model(image, refine_mode = REFINEMASK_ANNOTATION, keep_undetected_mask = True)

        blk_dict_list = []
        for blk in blk_list:
            blk_dict_list.append(blk.to_dict())

        json_data = {
            'image_path': str(path.image),
            'mask_path': str(path.mask),
            'base_path': str(img_path),
            'scale': scale,
            'blk_list': blk_dict_list
        }

        json_path = path.raw_json
        logger.info(f'Saving image json to: {json_path}')
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii = False, indent = 2, cls = NumpyJSONEncoder)

        imwrite(str(path.mask), mask_refined)
        logger.debug('Mask created')

        if raw_boxes:
            self._process_raw_boxes(image, blk_dict_list, path.raw_boxes)

        logger.info(f'Image {path.image.stem} processed')

    def resize_image_to(self, image: numpy.ndarray, height_lower_target: int, height_upper_target: int) -> tuple[numpy.ndarray, float]:
        height, width, _ = image.shape

        def calculate_new_size_and_scale(
            width: int, height: int, height_target_lower: int, height_target_upper: int
        ) -> tuple[int, int, float]:
            if height_target_lower <= 0 or height_target_upper <= 0 or height <= height_target_upper:
                return width, height, 1.0

            if height_target_lower >= height_target_upper:
                scale = height_target_upper / height
                new_width = round(width * scale)
                new_height = height_target_lower
            else:
                inv_scale_lower = height / height_target_lower
                inv_scale_upper = height / height_target_upper
                inv_scale_upper_nearest = ceil(inv_scale_upper)
                if inv_scale_upper_nearest <= inv_scale_lower:
                    scale = 1 / inv_scale_upper_nearest
                    new_width = round(width * scale)
                    new_height = round(height * scale)
                else:
                    max_height = round(height / inv_scale_upper)
                    min_height = round(height / inv_scale_lower)
                    new_height = floor(max_height / 4) * 4
                    if new_height < min_height:
                        new_height = max_height
                    scale = new_height / height
                    new_width = round(width * scale)

            return new_width, new_height, scale
        
        new_width, new_height, scale = calculate_new_size_and_scale(width, height, height_lower_target, height_upper_target)
        return cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA), scale

    def _process_raw_boxes(self, image: numpy.ndarray, boxes: list[dict[str, Any]], save_path: Path) -> None:
        image = Image.fromarray(image[:, :, ::-1])
        font_size = int(image.size[0] / 100) + 5

        empty_image = Image.new('RGBA', image.size)
        draw = ImageDraw.Draw(empty_image)

        for index, box in enumerate(boxes):
            xyxy: tuple[int, int, int, int] = box['xyxy']
            draw.rectangle(xyxy, fill = 'darkturquoise')

        alpha_channel = empty_image.split()[3]
        alpha_channel = ImageEnhance.Brightness(alpha_channel).enhance(48 / 255)
        empty_image.putalpha(alpha_channel)

        image = Image.alpha_composite(image.convert('RGBA'), empty_image)
        draw = ImageDraw.Draw(image)

        for index, box in enumerate(boxes):
            xyxy: tuple[int, int, int, int] = box['xyxy']
            Point = NewType('Point', tuple[int, int])
            lines: list[tuple[Point, Point, Point, Point]] = box['lines']
            language: str = box['language']

            for line in lines:
                for i in range(4):
                    x1, y1 = line[i]
                    x2, y2 = line[(i + 1) % 4]
                    draw.line((x1, y1, x2, y2), fill = 'hotpink', width = 1)

            draw.rectangle(xyxy, outline = 'darkturquoise')

            # Box Index
            draw.text(
                (xyxy[0] + 4, xyxy[1]),
                str(index + 1),
                fill = 'darkturquoise',
                font = ImageFont.truetype(FONT_ROBOTO_PATH, font_size),
                stroke_fill = 'white',
                stroke_width = 3
            )

            # Language
            draw.text(
                (xyxy[0] + 4, xyxy[1] + font_size + 4),
                language,
                fill = 'crimson',
                font = ImageFont.truetype(FONT_ROBOTO_PATH, font_size),
                stroke_fill = 'white',
                stroke_width = 3
            )
        
        logger.info(f'Boxes {save_path.name} created')
        image.save(save_path)
