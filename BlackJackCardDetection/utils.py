def calculate_best_count(cards):
    card_values = {
        'A': 11,  # Ace
        'K': 10,  # King
        'Q': 10,  # Queen
        'J': 10  # Jack
    }

    count = 0
    has_ace = False

    for card in cards:
        value, suit = card[:-1], card[-1]

        # Check if the value is a special card (Ace, King, Queen, Jack)
        if value in card_values:
            count += card_values[value]
            if value == 'A':
                has_ace = True
        else:
            # Convert the value to an integer
            value = int(value)
            count += value

        # If the count exceeds 21 and there's an Ace, subtract 10 to make it 1
        if count > 21 and has_ace:
            count -= 10
            has_ace = False  # Reset the flag since we've adjusted the Ace's value

    return count


def merge_bboxes(existing_bbox, new_bbox):
    top = min(new_bbox['y'] - new_bbox['height'] / 2, existing_bbox['y'] - existing_bbox['height'] / 2)
    bottom = max(new_bbox['y'] + new_bbox['height'] / 2, existing_bbox['y'] + existing_bbox['height'] / 2)
    left = min(new_bbox['x'] - new_bbox['width'] / 2, existing_bbox['x'] - existing_bbox['width'] / 2)
    right = max(new_bbox['x'] + new_bbox['width'] / 2, existing_bbox['x'] + existing_bbox['width'] / 2)

    merged_bbox = {
        'x': (left + right) / 2,
        'y': (top + bottom) / 2,
        'width': right - left,
        'height': bottom - top
    }

    return merged_bbox
