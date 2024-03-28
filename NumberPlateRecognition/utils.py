import easyocr
import string
import csv

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score',
                      'license_number', 'license_number_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for frame, frame_data in results.items():
            for car_id, car_info in frame_data.items():
                if 'car' in car_info and 'license_plate' in car_info and 'text' in car_info['license_plate']:
                    writer.writerow({
                        'frame': frame,
                        'car_id': car_id,
                        'car_bbox': '[{} {} {} {}]'.format(*car_info['car']['bbox']),
                        'license_plate_bbox': '[{} {} {} {}]'.format(*car_info['license_plate']['bbox']),
                        'license_plate_bbox_score': car_info['license_plate']['bbox_score'],
                        'license_number': car_info['license_plate']['text'],
                        'license_number_score': car_info['license_plate']['text_score']
                    })


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    # Format: two uppercase letters, two digits, three uppercase letters
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in string.digits or text[2] in dict_char_to_int.keys()) and \
       (text[3] in string.digits or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting possible wrong characters read by EasyOCR at definitely incorrect
    number plate index.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate = ''
    mapping = {0: dict_int_to_char,
               1: dict_int_to_char,
               2: dict_char_to_int,
               3: dict_char_to_int,
               4: dict_int_to_char,
               5: dict_int_to_char,
               6: dict_int_to_char}

    for i in [0, 1, 2, 3, 4, 5, 6]:
        if text[i] in mapping[i].keys():
            license_plate += mapping[i][text[i]]
        else:
            license_plate += text[i]

    return license_plate


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    car_index = -1
    found = False
    for i in range(len(vehicle_track_ids)):
        x1_car, y1_car, x2_car, y2_car, car_id = vehicle_track_ids[i]

        if x1 > x1_car and y1 > y1_car and x2 < x2_car and y2 < y2_car:
            car_index = i
            found = True
            break

    if found:
        return vehicle_track_ids[car_index]

    return -1, -1, -1, -1, -1
