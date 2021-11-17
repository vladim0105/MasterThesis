from PIL import Image, ImageDraw, ImageFont
import numpy as np


def text_phantom(text, size):
    # Availability is platform dependent
    pil_font = ImageFont.load_default()

    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('1', [size*5, size], 255)

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size*5 - text_width) // 2,
              (size - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    out = np.asarray(canvas)*255
    return out
