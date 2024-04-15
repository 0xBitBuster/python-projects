import cv2 as cv


def find_poker_hand(hand):
    if len(hand) != 5:
        return "Hand must contain exactly 5 cards."

    ranks = []
    suits = []
    possible_ranks = []
    poker_hand_ranks = {10: "Royal Flush", 9: "Straight Flush", 8: "Four of a Kind", 7: "Full House", 6: "Flush",
                        5: "Straight", 4: "Three of a Kind", 3: "Two Pair", 2: "Pair", 1: "High Card"}

    # Extract rank and suit from the card string
    for card in hand:
        if len(card) == 2:
            rank = card[0]
            suit = card[1]
        else:
            rank = card[0:2]  # For 10, J, Q, K, A
            suit = card[2]

        # Convert face cards to their numeric values
        if rank in ("J", "Q", "K", "A"):
            rank = {"J": 11, "Q": 12, "K": 13, "A": 14}[rank]
        else:
            rank = int(rank)
        ranks.append(rank)
        suits.append(suit)

    sorted_ranks = sorted(ranks)

    # Check for Flush, Royal Flush, and Straight Flush
    if suits.count(suits[0]) == 5:
        if sorted_ranks == list(range(10, 15)):  # Check for Royal Flush
            possible_ranks.append(10)
        elif all(sorted_ranks[i] == sorted_ranks[i - 1] + 1 for i in range(1, len(sorted_ranks))):
            possible_ranks.append(9)  # Straight Flush
        else:
            possible_ranks.append(6)  # Flush

    # Check for Straight
    if all(sorted_ranks[i] == sorted_ranks[i - 1] + 1 for i in range(1, len(sorted_ranks))):
        possible_ranks.append(5)

    unique_ranks = list(set(sorted_ranks))

    # Check for Four of a Kind, Full House, Three of a Kind, and Two Pair
    if len(unique_ranks) == 2:
        counts = [sorted_ranks.count(rank) for rank in unique_ranks]
        if counts == [4, 1]:
            possible_ranks.append(8)  # Four of a Kind
        elif counts == [3, 2]:
            possible_ranks.append(7)  # Full House
    elif len(unique_ranks) == 3:
        counts = [sorted_ranks.count(rank) for rank in unique_ranks]
        if counts == [3, 2, 1]:
            possible_ranks.append(4)  # Three of a Kind
        elif counts == [2, 2, 1]:
            possible_ranks.append(3)  # Two Pair
    elif len(unique_ranks) == 4:
        possible_ranks.append(2)  # Pair

    if not possible_ranks:
        possible_ranks.append(1)  # High Card

    output = poker_hand_ranks[max(possible_ranks)]
    return output
    

def draw_text_with_background(frame, text, position, font=cv.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=1, text_color=(255, 255, 255), background_color=(0, 0, 0)):
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    background_rect_x = position[0] - 2
    background_rect_y = position[1] - text_size[1] - 2
    cv.rectangle(frame, (background_rect_x, background_rect_y), (position[0] + text_size[0] + 2, position[1] + 2), background_color, -1)
    cv.putText(frame, text, position, font, font_scale, text_color, font_thickness)
