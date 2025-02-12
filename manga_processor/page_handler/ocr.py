import pytesseract

from PIL import Image
from pathlib import Path
from functools import cache
from enum import auto, StrEnum
from manga_ocr import MangaOcr as MangaOCRModel
from typing import Protocol, TypeAlias, Callable, Sequence

from ..mt_logger import logger

class OCR(StrEnum):
    AUTO = 'auto'
    MANGA_OCR = 'manga-ocr'
    TESSERACT_OCR = 'tesseract'

class LanguageCode(StrEnum):
    detect_box = auto()
    detect_page = auto()
    jpn = auto()
    eng = auto()
    kor = auto()
    kor_vert = auto()
    chi_sim = auto()
    chi_tra = auto()
    sqi = auto()
    ara = auto()
    aze = auto()
    aze_cyrl = auto()
    ben = auto()
    bul = auto()
    mya = auto()
    cat = auto()
    hrv = auto()
    ces = auto()
    dan = auto()
    nld = auto()
    epo = auto()
    est = auto()
    fin = auto()
    fra = auto()
    kat = auto()
    deu = auto()
    ell = auto()
    heb = auto()
    hin = auto()
    hun = auto()
    ind = auto()
    ita = auto()
    kaz = auto()
    lat = auto()
    lit = auto()
    ltz = auto()
    msa = auto()
    mon = auto()
    nep = auto()
    nor = auto()
    fas = auto()
    pol = auto()
    por = auto()
    ron = auto()
    rus = auto()
    srp = auto()
    srp_latn = auto()
    slk = auto()
    slv = auto()
    spa = auto()
    swe = auto()
    tgl = auto()
    tam = auto()
    tel = auto()
    tha = auto()
    tur = auto()
    ukr = auto()
    vie = auto()

    def __str__(self):
        return self.value

language_names: dict[LanguageCode, str] = {
    LanguageCode.detect_box: "Detect per box",
    LanguageCode.detect_page: "Detect per page",
    LanguageCode.jpn: "Japanese",
    LanguageCode.eng: "English",
    LanguageCode.kor: "Korean",
    LanguageCode.kor_vert: "Korean (vertical)",
    LanguageCode.chi_sim: "Chinese - Simplified",
    LanguageCode.chi_tra: "Chinese - Traditional",
    LanguageCode.sqi: "Albanian",
    LanguageCode.ara: "Arabic",
    LanguageCode.aze: "Azerbaijani",
    LanguageCode.aze_cyrl: "Azerbaijani - Cyrilic",
    LanguageCode.ben: "Bengali",
    LanguageCode.bul: "Bulgarian",
    LanguageCode.mya: "Burmese",
    LanguageCode.cat: "Catalan; Valencian",
    LanguageCode.hrv: "Croatian",
    LanguageCode.ces: "Czech",
    LanguageCode.dan: "Danish",
    LanguageCode.nld: "Dutch; Flemish",
    LanguageCode.epo: "Esperanto",
    LanguageCode.est: "Estonian",
    LanguageCode.fin: "Finnish",
    LanguageCode.fra: "French",
    LanguageCode.kat: "Georgian",
    LanguageCode.deu: "German",
    LanguageCode.ell: "Greek",
    LanguageCode.heb: "Hebrew",
    LanguageCode.hin: "Hindi",
    LanguageCode.hun: "Hungarian",
    LanguageCode.ind: "Indonesian",
    LanguageCode.ita: "Italian",
    LanguageCode.kaz: "Kazakh",
    LanguageCode.lat: "Latin",
    LanguageCode.lit: "Lithuanian",
    LanguageCode.ltz: "Luxembourgish",
    LanguageCode.msa: "Malay",
    LanguageCode.mon: "Mongolian",
    LanguageCode.nep: "Nepali",
    LanguageCode.nor: "Norwegian",
    LanguageCode.fas: "Persian",
    LanguageCode.pol: "Polish",
    LanguageCode.por: "Portuguese",
    LanguageCode.ron: "Romanian; Moldavian",
    LanguageCode.rus: "Russian",
    LanguageCode.srp: "Serbian",
    LanguageCode.srp_latn: "Serbian - Latin",
    LanguageCode.slk: "Slovak",
    LanguageCode.slv: "Slovenian",
    LanguageCode.spa: "Spanish; Castilian",
    LanguageCode.swe: "Swedish",
    LanguageCode.tgl: "Tagalog",
    LanguageCode.tam: "Tamil",
    LanguageCode.tel: "Telugu",
    LanguageCode.tha: "Thai",
    LanguageCode.tur: "Turkish",
    LanguageCode.ukr: "Ukrainian",
    LanguageCode.vie: "Vietnamese",
}

def to_language_code(lang: str) -> LanguageCode | None:
    return LanguageCode[lang] if lang in LanguageCode.__members__ else None

class TesseractOCR:
    def __init__(self, language: str | None = None) -> None:
        self.language = language

    @staticmethod
    def tesseract_available() -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except (pytesseract.TesseractNotFoundError, SystemExit) as error:
            logger.error(f'Failed to cerify tesseract: {error}')
            return False
    
    @staticmethod
    def languages() -> bool:
        try:
            return set(
                LanguageCode(code)
                for code in pytesseract.get_languages()
                if code in LanguageCode.__members__
            )
        except pytesseract.TesseractNotFoundError as error:
            logger.error(f'Failed to check Tesseract langauge: {error}')
            return set()
        
    def __call__(self, img: Image.Image | Path | str, language: str | None = None, **kwargs) -> str:
        if not self.tesseract_available():
            raise RuntimeError('Tesseract OCR is not found.')
        if language and language not in self.languages():
            raise RuntimeError(f'Language {language} is not found in Tesseract OCR')
        
        if isinstance(img, (str, Path)):
            image = Image.open(img)
        elif isinstance(img, Image.Image):
            image = img
        else:
            raise ValueError(f"img must be a path or PIL.Image, got: {type(img)}")
        
        try:
            text_raw = pytesseract.image_to_string(
                image, lang = language or self.language, config = r''
            )
        except Exception:
            text_raw = ''

        if not text_raw:
            try:
                data = pytesseract.image_to_data(
                    image, lang = language or self.language, config = r'--psm 11', output_type=pytesseract.Output.DICT
                )
                text_raw = ' '.join(_ for _ in data.get('text', ()) if _)
            except Exception as error:
                logger.error(f'Failed to run tesseract: {error}')

        return text_raw.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\f", " ").strip()

class MangaOCR:
    _instance = None
    _model = None
    _init_args = ((), {})

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info("Creating the MangaOcr instance")
            cls._instance = super(MangaOCR, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._init_args = (args, kwargs)
        return cls._instance

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def languages() -> set[LanguageCode]:
        return {LanguageCode.jpn}

    def initialize_model(self, *args, **kwargs):
        if self._model is None:
            if args or kwargs:
                self._model = MangaOCRModel(*args, **kwargs)
            else:
                init_args = self._init_args
                self._model = MangaOCRModel(*init_args[0], **init_args[1])
        return self._model

    def __call__(self, image: Image.Image | Path | str, **kwargs) -> str:
        model = self.initialize_model()
        return model(image)

class OCRData:
    def __init__(self,
                 path: Path,
                 boxes_amount: int,
                 box_sizes_ocr: Sequence[int],
                 box_sizes_removed: Sequence[int],
                 removed_box_data: Sequence[tuple[str, tuple[int]]]
                ) -> None:
        self.path: Path = path
        self.boxes_amount: int = boxes_amount
        self.box_sizes_ocr: Sequence[int] = box_sizes_ocr
        self.box_sizes_removed: Sequence[int] = box_sizes_removed
        self.removed_box_data: Sequence[tuple[str, tuple[int]]] = removed_box_data

class OCRModel(Protocol):
    def __call__(self, img: Image.Image | Path | str, language: str | None = None, **kwargs) -> str: ...

OCREngine: TypeAlias = Callable[[LanguageCode], OCRModel]

def create_ocr_engine(use_tesseract: bool, prefered_ocr_engine: OCR) -> OCREngine:
    tesseract_languages: set[LanguageCode] = TesseractOCR.languages()
    manga_ocr_languages: set[LanguageCode] = MangaOCR.languages()

    if use_tesseract and not tesseract_languages:
        logger.error('Tesseract OCR is not found. Please install it correctly. Now back to using manga-ocr')

    if not use_tesseract:
        prefered_ocr_engine = OCR.MANGA_OCR

    def closure(langauage: LanguageCode) -> OCRModel:
        if prefered_ocr_engine == OCR.AUTO:
            if langauage in manga_ocr_languages: return MangaOCR()
            if langauage in tesseract_languages: return TesseractOCR(langauage)
        elif prefered_ocr_engine == OCR.TESSERACT_OCR:
            if langauage in tesseract_languages: return TesseractOCR(langauage)
            logger.error(f'Tesseract doesn\'t have language {langauage}. Try to install Tesseract correctly. Now back to using manga-ocr')
        
        return MangaOCR()

    return closure

def available_languages() -> set[LanguageCode]:
    langs = set()
    for ocr in (MangaOCR(), TesseractOCR()):
        langs |= ocr.languages()
    return langs