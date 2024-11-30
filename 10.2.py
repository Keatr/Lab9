import cv2
import numpy as np
import os

def extract_barcodes(image_path):
    assert os.path.exists(image_path), f"File not found: {image_path}"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image file {image_path}")
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    barcodes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > h:  # barcodes, not vertical lines
            barcode = image[y:y+h, x:x+w]
            barcodes.append((barcode, (x, y, w, h)))
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Barcodes Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return barcodes

def analyze_barcode(barcode):
    gray = cv2.cvtColor(barcode, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    bar_positions = []
    current_position = None
    thickness = 0

    for x in range(binary.shape[1]):
        column = binary[:, x]
        if np.mean(column) < 128:  # Detect black line
            if current_position is None:
                current_position = x
                thickness = 1
            else:
                thickness += 1
        else:
            if current_position is not None:
                bar_positions.append((current_position, thickness))
                current_position = None
                thickness = 0

    return bar_positions

barcodes = extract_barcodes('test2.png')
if barcodes:
    with open('barcode_info.txt', 'w') as f:
        for idx, (barcode, (x, y, w, h)) in enumerate(barcodes):
            positions_thicknesses = analyze_barcode(barcode)
            f.write(f'Barcode {idx + 1} at position ({x}, {y}, {w}, {h}) has lines:\n')
            for position, thickness in positions_thicknesses:
                f.write(f'  Position: {position}, Thickness: {thickness}\n')
        print('Barcode information has been written to barcode_info.txt')
