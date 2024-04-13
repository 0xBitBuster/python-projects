import cv2 as cv

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

        if value in card_values:
            count += card_values[value]
            if value == 'A':
                has_ace = True
        else:
            value = int(value)
            count += value

        if count > 21 and has_ace:
            count -= 10
            has_ace = False

    return count


def draw_text_with_background(frame, text, position, font=cv.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=1, text_color=(255, 255, 255), background_color=(0, 0, 0)):
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    background_rect_x = position[0] - 2
    background_rect_y = position[1] - text_size[1] - 2
    cv.rectangle(frame, (background_rect_x, background_rect_y), (position[0] + text_size[0] + 2, position[1] + 2), background_color, -1)
    cv.putText(frame, text, position, font, font_scale, text_color, font_thickness)
