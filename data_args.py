from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    data_folder: str = field(default="./data", metadata={"help": "data folder"})
    size: int = field(default=256, metadata={"help": "Image height"})
    num_frames: int = field(default=13, metadata={"help": "num frames used to train CNN, should be odd number"})
    frame_steps: int = field(default=4, metadata={"help": "number of skipped frames between two selected frames"})
    fold: int = field(default=0, metadata={"help": "Fold"})