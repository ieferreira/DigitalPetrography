import os
import glob


def clean_folder(folder="responses/*"):
    """Cleans given folder of files

    :param folder: folder to be cleaned, defaults to "responses/*"
    :type folder: str, optional
    """
    files = glob.glob(folder)
    for f in files:
        os.remove(f)
