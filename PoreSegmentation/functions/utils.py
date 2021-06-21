import os, glob

def clean_folder(folder="certificados/*"):
    files = glob.glob(folder)
    for f in files:
        os.remove(f)

        