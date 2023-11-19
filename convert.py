# Converts different file formats to .jpg so the detector can handle them

import os
import subprocess

directory = 'cropped_faces'

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.lower().endswith(('.jpeg', '.png', '.heic', '.webp')):
            filepath = os.path.join(root, file)
            print(f'Converting {filepath}...')
            subprocess.run(['magick', filepath, '-background', 'white', '-alpha', 'remove', '-strip', '-quality', '80%', os.path.splitext(filepath)[0] + '.jpg'])
            os.remove(filepath)
