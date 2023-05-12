# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, File
from transparent_background import Remover
from PIL import Image
import tempfile

class Predictor(BasePredictor):
    
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = Remover(device='cuda:0')

    def scale_to(self, image, size:int):
        """
        size should be 512 or 1024.
        ofc other sizes are possible
        """
        width,height = image.size
        
        # in case the image isn't square-shaped,
        # the larger side gets scaled to value "size"
        
        if width >=height:
            factor = size/width
        else:
            factor = size/height
            
        new_width = round(width*factor)
        new_height = round(height*factor)
        
        image_scaled = image.resize((new_width,new_height))
        return image_scaled

    def remove_background(self, image):
        noback = self.model.process(image)
        return Image.fromarray(noback)

    def predict(self, 
                img: File = Input(description="Image to enlarge"),
                size: int = Input(description="512,768 or 1024", choices=[512,768,1024])
                )-> Path:
            """
            Returns a scaled image (size x size) with background removed
            """
            pil_img = Image.open(img)
            scaled = self.scale_to(pil_img, size)
            scaled = scaled.convert("RGB") # RGBA breaks
            noback_scaled = self.remove_background(scaled)
            out_path = Path(tempfile.mkdtemp()) / "out.png"
            noback_scaled.save(str(out_path))
            return out_path
        
                


