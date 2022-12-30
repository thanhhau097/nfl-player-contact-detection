from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    data_folder: str = field(default="./data", metadata={"help": "data folder"})
    size: int = field(default=256, metadata={"help": "crop size height/width"})
    num_frames: int = field(default=13, metadata={"help": "num frames used to train CNN, should be odd number"})
    frame_steps: int = field(default=4, metadata={"help": "number of skipped frames between two selected frames"})
    num_center_frames: int = field(default=3, metadata={"help": "num center frames used to train CNN, not including middle frame"})
    fold: int = field(default=0, metadata={"help": "Fold"})
    use_heatmap: bool = field(default=False, metadata={"help": "Use heatmap instead of crop"})
    heatmap_sigma: int = field(default=128, metadata={"help": "sigma for drawing heatmap"})
    img_height: int = field(default=720, metadata={"help": "image height"})
    img_width: int = field(default=1280, metadata={"help": "image width"})
    num_cache: int = field(
        default=-1, metadata={"help": "num train frames cached on RAM"}
    )
