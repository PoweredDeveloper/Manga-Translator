from pathlib import Path
from uuid import uuid4

class MT_PathGen:
    def __init__(self, base_path: Path, save_directory: Path, uuid: Path | str | None = None) -> None:
        self.base_path = base_path.resolve()
        self.save_directory = save_directory.resolve()

        if uuid is None:
            self.uuid = uuid4()
        else:
            if isinstance(uuid, Path):
                self.uuid = uuid.stem.split('_')[0]
            elif isinstance(uuid, str):
                self.uuid = uuid
            else:
                raise TypeError('uuid parameter should be Path or str')
            
        self._output_path = str(save_directory / f'{self.uuid}_{base_path.stem}')

    @property
    def image(self) -> Path:
        return Path(self._output_path + '_image.png')
    
    @property
    def mask(self) -> Path:
        return Path(self._output_path + '_mask.png')
    
    @property
    def masked(self) -> Path:
        return Path(self._output_path + '_masked.png')
    
    @property
    def mask_fitments(self) -> Path:
        return Path(self._output_path + '_mask_fitments.png')
    
    @property
    def combined_mask(self) -> Path:
        return Path(self._output_path + '_combined_mask.png')
    
    @property
    def std_deviations(self) -> Path:
        return Path(self._output_path + '_std_deviations.png')
    
    @property
    def raw_boxes(self) -> Path:
        return Path(self._output_path + '_raw_boxes.png')
    
    @property
    def boxes(self) -> Path:
        return Path(self._output_path + '_boxes.png')
    
    @property
    def box_mask(self) -> Path:
        return Path(self._output_path + '_box_mask.png')
    
    @property
    def cut_mask(self) -> Path:
        return Path(self._output_path + '_cut_mask.png')
    
    @property
    def final_boxes(self) -> Path:
        return Path(self._output_path + '_final_boxes.png')
    
    @property
    def clean_json(self) -> Path:
        return Path(self._output_path + '#clean.json')
    
    @property
    def raw_json(self) -> Path:
        return Path(self._output_path + '#raw.json')
    
    @property
    def mask_data_json(self) -> Path:
        return Path(self._output_path + '#mask_data.json')
    
    @property
    def text(self) -> Path:
        return Path(self._output_path + '_text.png') 

    @property
    def clean(self) -> Path:
        return Path(self._output_path + '_clean.png')
    
    @property
    def clean_denoised(self) -> Path:
        return Path(self._output_path + '_clean_denoised.png')
    
    @property
    def noise_mask(self) -> Path:
        return Path(self._output_path + '_noise_mask.png')