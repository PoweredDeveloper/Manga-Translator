import cv2
import tqdm
import json
import numpy
import torch
import shutil
import multiprocessing as mp

from pathlib import Path
from typing import NewType, Any, Sequence
from PIL import Image, ImageEnhance, ImageDraw, ImageFont

from ..mt_logger import logger
from ..mt_paths import MT_PathGen
from ..mt_config import FONT_ROBOTO_PATH
from ..utils.io import NumpyJSONEncoder

from ..libs.comic_text_detector.inference import TextDetector
from ..libs.comic_text_detector.utils.io_utils import imwrite
from ..libs.comic_text_detector.utils.textmask import REFINEMASK_ANNOTATION

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

    def clean(self, image_list: list[Path], output_direction: Path) -> None:
        logger.info('Starting cleaning images')

        self.generate_mask_data(image_list, True)

        # json_files = Path(self.cache_path).glob('*.json')
        # data = []

        logger.debug('Running denoising')

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

                for _ in tqdm.tqdm(pool.imap_unordered(self._genreate_mask_batch, [batches, raw_boxes]), total = len(batches)): pass
        
        else:
            for _, image_path in enumerate(tqdm.tqdm(image_list)):
                self.generate_single_mask(image_path, raw_boxes)

    def _genreate_mask_batch(self, batch: list[list], raw_boxes: bool = False) -> None:
        for image_path in batch:
            self.generate_single_mask(image_path, raw_boxes)

        if self.device == 'cuda':
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def generate_single_mask(self, img_path: Path, raw_boxes: bool = False) -> None:
        if not img_path.exists():
            logger.error('Page path not found. Skipping..')
            return
        
        path = MT_PathGen(img_path, self.cache_path)
        
        image: numpy.ndarray = cv2.imdecode(numpy.fromfile(str(img_path), dtype = numpy.uint8,), cv2.IMREAD_COLOR)
        _, mask_refined, blk_list = self.model(image, refine_mode = REFINEMASK_ANNOTATION, keep_undetected_mask = True)

        blk_dict_list = []
        for blk in blk_list:
            blk_dict_list.append(blk.to_dict())

        json_data = {
            'image_path': str(path.image),
            'mask_path': str(path.mask),
            'base_path': str(img_path),
            'blk_list': blk_dict_list
        }

        json_path = path.json
        logger.info(f'Saving image json to: {json_path}')
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii = False, indent = 2, cls = NumpyJSONEncoder)

        imwrite(str(path.mask), mask_refined)
        logger.debug('Mask created')

        if raw_boxes:
            self._process_raw_boxes(image, blk_dict_list, path.raw_boxes)

        logger.info(f'Image {path.image.stem} processed')

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
