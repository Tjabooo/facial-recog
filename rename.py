## Script for renaming every file in chosen path to respective folder name

import os

# root_directory = "known_faces/"
root_directory = "cropped_faces/"

for dirpath, dirnames, filenames in os.walk(root_directory):
    for i, filename in enumerate(filenames):
        file_ext = os.path.splitext(filename)[1]
        new_filename = f"{i+1} {os.path.basename(dirpath)}{file_ext}"
        while os.path.exists(os.path.join(dirpath, new_filename)):
            j = 1
            while os.path.exists(os.path.join(dirpath, f"{j} {os.path.basename(dirpath)}{file_ext}")):
                j += 1
            new_filename = f"{j} {os.path.basename(dirpath)}{file_ext}"
        os.rename(os.path.join(dirpath, filename), os.path.join(dirpath, new_filename))
        print(f"Successfully renamed '{filename}' to '{new_filename}'")
