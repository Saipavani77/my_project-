import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    # Load the image using PIL
    image = Image.open(image_path)
    
    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(image)
    
    return text

def process_extracted_text(text):
    # Dummy function to process the extracted text and extract the required features
    # This needs to be tailored based on how the text data is structured
    # For example, you may need regex to find specific information
    followers = 1500
    following = 300
    engagement_rate = 0.05
    is_promotional = 0
    post_frequency = 3
    sentiment = 0.1
    
    return {
        'followers': followers,
        'following': following,
        'engagement_rate': engagement_rate,
        'is_promotional': is_promotional,
        'post_frequency': post_frequency,
        'sentiment': sentiment
    }
