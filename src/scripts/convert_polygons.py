import os
import numpy as np
from glob import glob
import shutil

def convert_polygon_to_bbox(polygon_coords):
    # Split x and y coordinates
    xs = np.array(polygon_coords[0::2])
    ys = np.array(polygon_coords[1::2])

    # Find min and max to build the bounding box
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)

    # Convert from (x_min, y_min, x_max, y_max) to YOLO (x_center, y_center, w, h)
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + (width / 2)
    y_center = y_min + (height / 2)

    return [x_center, y_center, width, height]

def process_label_files(input_dir, output_dir):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all label files in the input directory
    label_files = glob(os.path.join(input_dir, '*.txt'))
    
    if not label_files:
        print(f"No .txt files found in: {input_dir}")
        return

    print(f"Found {len(label_files)} files in {input_dir}. Converting...")

    for file_path in label_files:
        new_lines = []
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) > 1:
                class_id = parts[0]
                polygon_coords = [float(p) for p in parts[1:]]
                
                # Ensure there is an even number of coordinates
                if len(polygon_coords) % 2 != 0:
                    print(f"WARNING: Odd number of coordinates in file {os.path.basename(file_path)}. Skipping line.")
                    continue

                # Convert to bounding box
                bbox_coords = convert_polygon_to_bbox(polygon_coords)

                # Format the new line
                new_line = f"{class_id} {bbox_coords[0]} {bbox_coords[1]} {bbox_coords[2]} {bbox_coords[3]}"
                new_lines.append(new_line)

        # Write the new label file to the output directory
        output_file_path = os.path.join(output_dir, os.path.basename(file_path))
        with open(output_file_path, 'w') as f:
            f.write('\n'.join(new_lines))

    print(f"Conversion completed! Files saved to: {output_dir}")


# /src/scripts/convert_polygons.py

if __name__ == '__main__':
    # --- PATHS ADAPTED FOR images/train, labels/train STRUCTURE ---
    
    base_dir = os.path.join('datasets', 'my_dataset')
    
    # Input directories (where the polygons live)
    train_labels_in = os.path.join(base_dir, 'labels', 'train')
    val_labels_in = os.path.join(base_dir, 'labels', 'val')
    
    # Output directories (where the new bounding boxes will be saved)
    train_labels_out = os.path.join(base_dir, 'labels', 'train_bbox')
    val_labels_out = os.path.join(base_dir, 'labels', 'val_bbox')
    
    # --- RUN CONVERSION ---
    
    # Process training files
    process_label_files(train_labels_in, train_labels_out)
    
    # Process validation files
    process_label_files(val_labels_in, val_labels_out)
    
    print("\nReminder: The next step is to rename the original 'train' and 'val' folders")
    print("and then rename the new 'train_bbox' and 'val_bbox' folders.")
    print("\nExample for the training folder:")
    print(f"1. Rename '{train_labels_in}' to '{train_labels_in}_polygon_backup'")
    print(f"2. Rename '{train_labels_out}' to '{train_labels_in}'")