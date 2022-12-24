from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """
    data_folder: str = field(default="./data/develop", metadata={"help": "data folder"})
    height: int = field(default=512, metadata={"help": "Image height"})
    width: int = field(default=512, metadata={"help": "Image width"})
    fold: int = field(default=0, metadata={"help": "Fold"})
