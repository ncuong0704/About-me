from collections import Counter

def check_full_house(seq):
    counts = Counter(seq)
    dupes = [v for v in counts.values() if v == 2 or v == 3]
    return dupes == [2, 3] or dupes == [3, 2]
def check_three_of_a_kind(seq):
    counts = Counter(seq)
    dupes = [v for v in counts.values() if v == 3]
    return dupes == [3]
def check_two_pair(seq):
    counts = Counter(seq)
    dupes = [v for v in counts.values() if v == 2]
    return dupes == [2, 2]
def check_one_pair(seq):
    counts = Counter(seq)
    dupes = [v for v in counts.values() if v == 2]
    return dupes == [2]
rankPokers = {
    "1": "Royal Flush", 
    "2": "Straight Flush",
    "3": "Four of a Kind",
    "4": "Full House",
    "5": "Flush",
    "6": "Straight",
    "7": "Three of a Kind",
    "8": "Two Pair",
    "9": "One Pair",
    "10": "High Card",
}

rankCards = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

rankSuits = {
    "h": 1, # cơ
    "d": 2, # rô
    "c": 3, # chuồn
    "s": 4, # bích
}
def get_ranks_and_suits(hand):
    ranks = []
    suits = []
    for card in hand:
        if len(card) == 2:
            rank = card[0]
            suit = card[1]
        else:
            rank = card[0:2]
            suit = card[2]
        ranks.append(rankCards[rank])
        suits.append(rankSuits[suit])
    return ranks, suits
def findPokerHand(hand):
    hand = list(dict.fromkeys(hand))
    if len(hand) != 5:
        return "No cards"
    ranks, suits = get_ranks_and_suits(hand)
    # Kiểm tra đồng chất
    if suits.count(suits[0]) == 5:
        if sorted(ranks) == list(range(min(ranks), max(ranks) + 1)):
            if max(ranks) == 14:
                result = 1
            else:
                result = 2
        else:
            result = 5
    elif sorted(ranks) == list(range(min(ranks), max(ranks) + 1)):
        result = 6
    elif ranks.count(ranks[0]) == 4 or ranks.count(ranks[1]) == 4:
        result = 3
    elif check_full_house(ranks):
        result = 4
    elif check_three_of_a_kind(ranks):
        result = 7
    elif check_two_pair(ranks):
        result = 8
    elif check_one_pair(ranks):
        result = 9
    else:
        result = 10
    
  
    output = rankPokers[str(result)]
    print(output)
    return output

if __name__ == "__main__":
    findPokerHand(["10h", "Jh", "Qh", "Kh", "Ah"]) # Thùng phá sảnh lớn
    findPokerHand(["6c", "7c", "8c", "9c", "10c"]) # Thùng phá sảnh
    findPokerHand(["9h", "9d", "9s", "9c", "3c"]) # Tứ quý
    findPokerHand(["Qh", "Qd", "Qs", "8c", "8d"]) # Cù lũ
    findPokerHand(["2s", "7s", "9s", "Js", "Ks"]) # Đồng chất
    findPokerHand(["2d", "3c", "4h", "5s", "6h"]) # Sảnh
    findPokerHand(["5h", "5c", "5d", "Qh", "2s"]) # Sám cô
    findPokerHand(["4c", "4d", "10s", "10h", "Jc"]) # Hai đôi
    findPokerHand(["8c", "8d", "3h", "Jd", "Ks"]) # Một đôi
    findPokerHand(["2h", "5d", "7c", "9s", "Kh"]) # Mậu thầu (High card)