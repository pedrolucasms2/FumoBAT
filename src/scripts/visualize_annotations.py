# /src/scripts/visualize_annotations.py
import cv2
import os
import random

def draw_yolo_bboxes(image_path, label_path):
    """
    Draw YOLO label bounding boxes on an image.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return

    h, w, _ = image.shape

    # Check whether the label file exists
    if not os.path.exists(label_path):
        print(f"Label file not found for {os.path.basename(image_path)}")
        cv2.imshow("Annotation Check", image)
        cv2.waitKey(0)
        return

    # Read the label file
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)

                # Convert YOLO (normalized) coordinates to pixel coordinates
                x_center_px = x_center * w
                y_center_px = y_center * h
                width_px = width * w
                height_px = height * h

                # Compute the box corners (x_min, y_min)
                x1 = int(x_center_px - (width_px / 2))
                y1 = int(y_center_px - (height_px / 2))
                x2 = int(x_center_px + (width_px / 2))
                y2 = int(y_center_px + (height_px / 2))

                # Draw the rectangle on the image
                # Color (B, G, R) -> Green. Thickness = 2 pixels.
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Optional: draw the class ID
                label_text = f"Class: {int(class_id)}"
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the image
    # Resize the image if it is too large for the screen
    max_display_size = 800
    if h > max_display_size or w > max_display_size:
        scale = max_display_size / max(h, w)
        image_resized = cv2.resize(image, (int(w*scale), int(h*scale)))
        cv2.imshow(f"Annotation Check: {os.path.basename(image_path)}", image_resized)
    else:
        cv2.imshow(f"Annotation Check: {os.path.basename(image_path)}", image)

    print(f"Showing annotations for: {os.path.basename(image_path)}. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # --- CONFIGURE PATHS HERE ---
    base_dir = os.path.join('datasets', 'my_dataset')
    
    # Inspect the training split
    image_dir = os.path.join(base_dir, 'images', 'train')
    label_dir = os.path.join(base_dir, 'labels', 'train') # Use the folder containing the converted labels!

    # Gather the list of all images
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not all_images:
        print(f"No images found in {image_dir}")
    else:
        # Choose 5 random images to inspect
        num_images_to_check = 5
        selected_images = random.sample(all_images, min(num_images_to_check, len(all_images)))

        for image_name in selected_images:
            image_path = os.path.join(image_dir, image_name)
            
            # Build the corresponding label file path
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(label_dir, label_name)
            
            draw_yolo_bboxes(image_path, label_path)