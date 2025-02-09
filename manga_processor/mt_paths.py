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
    def raw_boxes(self) -> Path:
        return Path(self._output_path + '_raw_boxes.png')
    
    @property
    def json(self) -> Path:
        return Path(self._output_path + '.json')

    @property
    def clean(self) -> Path:
        return Path(self._output_path + '_clean.png')
    
    @property
    def clean_denoised(self) -> Path:
        return Path(self._output_path + '_clean_denoised.png')
    
    @property
    def noise_mask(self) -> Path:
        return Path(self._output_path + '_noise_mask.png')