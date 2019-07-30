"""
Contains useful functions for interacting with directories.
"""
from pathlib import Path


def show_dir_contents(dir_path=Path('')):
    """
    Shows the contents of the directory pointed to by 'dir_path'
    :param dir_path: pathlib.Path or string, optional (default: current directory)
                     The path of the directory
    :return: None
    """
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)

    if not dir_path.exists():
        raise ValueError("Directory does not exist.")
    if not dir_path.is_dir():
        raise ValueError("The given path is not a path to a directory.")
    name_lengths = [len(item.name) for item in dir_path.iterdir()]

    if len(name_lengths) == 0:
        return

    align_num = max([len(file.name) for file in dir_path.iterdir()]) + 4

    print("Contents of \'{}\':\n".format(dir_path))
    print("{0:<{align_len}} {1}".format("Name", "Length (kB)", align_len=align_num))
    print("{0:<{align_len}} {1}".format("----", "-----------", align_len=align_num))

    contents = sorted((item for item in dir_path.iterdir()), key=lambda x: not x.is_dir())

    for item in contents:
        if item.is_file():
            print("{0:<{align_len}} {1}".format(item.name, round(item.stat().st_size / 1024),
                  align_len=align_num))
        else:
            print(f"{item.name}")


