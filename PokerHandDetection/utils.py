import cv2 as cv


def find_poker_hand(hand):
    if len(hand) == 0:
        return "No cards detected"

    poker_hand_ranks = {10: "Royal Flush", 9: "Straight Flush", 8: "Four of a Kind", 7: "Full House", 6: "Flush",
                        5: "Straight", 4: "Three of a Kind", 3: "Two Pair", 2: "Pair", 1: "High Card"}
    ranks = []
    suits = []
    for card in hand:
        if len(card) == 2:
            rank = card[0]
            suit = card[1]
        else:
            rank = card[0:2]
            suit = card[2]
        if rank == "A":
            rank = 14
        elif rank == "K":
            rank = 13
        elif rank == "Q":
            rank = 12
        elif rank == "J":
            rank = 11
        ranks.append(int(rank))
        suits.append(suit)

    sorted_ranks = sorted(ranks)

    # Identify available hand types based on number of cards
    possible_ranks = []
    if len(hand) >= 5:  # Full hand
        if suits.count(suits[0]) == 5: # Check for Flush
          possible_ranks.append(6) # Flush
    
          # Check for Straight within Flush
          if all(sorted_ranks[i] == sorted_ranks[i - 1] + 1 for i in range(1, len(sorted_ranks))):
            possible_ranks.append(9) # Straight Flush
    
          # Check for Royal Flush (within Straight Flush check)
          if 14 in sorted_ranks and 13 in sorted_ranks and 12 in sorted_ranks and 11 in sorted_ranks and 10 in sorted_ranks:
            possible_ranks.append(10) # -- Royal Flush
    
        # Straight (independent of Flush)
        if all(sorted_ranks[i] == sorted_ranks[i - 1] + 1 for i in range(1, len(sorted_ranks))):
          possible_ranks.append(5)

    if len(hand) >= 3:  # Check for Full House, Two Pair, Three of a Kind, or High Card
        hand_unique_values = list(set(sorted_ranks))

        # Check for Three of a Kind (before Two Pair)
        for val in hand_unique_values:
            if sorted_ranks.count(val) == 3:  # Three of a Kind
                possible_ranks.append(4)
                break  # Exit loop after finding Three of a Kind (optional)

        # **No check for Two Pair here (removed)**

        # High Card (as a fallback)
        possible_ranks.append(1)

    if len(hand) >= 2:  # Check for Four of a Kind, Pair, or High Card
        # Check for Pair directly (more reliable)
        if len(set(ranks)) < len(ranks):  # If number of unique ranks is less than total ranks, there's a pair
            possible_ranks.append(2)

        # Check for Four of a Kind
        for val in set(ranks):
            if sorted_ranks.count(val) == 4:
                possible_ranks.append(8)
                break  # Exit the loop after finding Four of a Kind

        # High Card (as a fallback)
        if not possible_ranks:  # No Pair or Four of a Kind found
            possible_ranks.append(1)

    else:  # Hand with less than 2 cards
        possible_ranks.append(1)

    output = poker_hand_ranks[max(possible_ranks)]
    return output
    

def draw_text_with_background(frame, text, position, font=cv.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=1, text_color=(255, 255, 255), background_color=(0, 0, 0)):
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    background_rect_x = position[0] - 2
    background_rect_y = position[1] - text_size[1] - 2
    cv.rectangle(frame, (background_rect_x, background_rect_y), (position[0] + text_size[0] + 2, position[1] + 2), background_color, -1)
    cv.putText(frame, text, position, font, font_scale, text_color, font_thickness)
