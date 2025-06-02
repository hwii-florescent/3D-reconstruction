import os
import re
import shutil
from pathlib import Path

def organize_frames(source_dir):
    """
    Moves and renames image files from source_dir into subfolders
    (middle_angle, left_angle, right_angle) located in the same
    directory as this script. Filenames lose their `_cam<suffix>` part.

    - Files ending with "_cam248622301868(.ext)" → middle_angle/
    - Files ending with "_cam336522303074(.ext)" → left_angle/
    - Files ending with "_cam336522303608(.ext)" → right_angle/
    """
    src = Path(source_dir)
    if not src.is_dir():
        raise NotADirectoryError(f"Source directory '{source_dir}' not found.")

    # Destination root is where this script lives
    script_dir = Path(__file__).parent.resolve()
    dst_root = script_dir

    # Mapping from camera‐suffix → folder name under dst_root
    cam_to_folder = {
        "248622301868": "middle_angle",
        "336522303074": "left_angle",
        "336522303608": "right_angle"
    }

    # Regex to match: (rgb_frame_<digits>)_cam(<one of the IDs>)(.<extension>)?
    pattern = re.compile(
        r"^(rgb_frame_\d+)_cam(248622301868|336522303074|336522303608)(\.[A-Za-z0-9]+)?$"
    )

    # Create each angle folder inside the script's directory
    for folder_name in cam_to_folder.values():
        (dst_root / folder_name).mkdir(parents=True, exist_ok=True)

    # Iterate over every file in source_dir
    for file in src.iterdir():
        if not file.is_file():
            continue

        m = pattern.match(file.name)
        if not m:
            # Skip files that don't match the pattern
            continue

        base_name, cam_id, ext = m.groups()
        ext = ext or ""  # If no extension was captured, use empty string

        target_folder = cam_to_folder[cam_id]
        new_name = f"{base_name}{ext}"

        src_path = file
        dst_path = dst_root / target_folder / new_name

        # Move and rename
        shutil.move(str(src_path), str(dst_path))


if __name__ == "__main__":
    # === EDIT only this line if needed: path to your images folder ===
    source_folder = r"rgb_images"
    # =================================================================
    organize_frames(source_folder)
    print(f"Images moved into subfolders under '{Path(__file__).parent.resolve()}'.")
