from PIL import Image

def import_image(file):
    """Imports PIL image for display and processing inside streamlit application

    :param file: File as given by streamlit file selector
    :type file: Streamlit file upload
    :return: Image to use
    :rtype: PIL.Image
    """    
    img = Image.open(file)
    return img
