import random

ACTION = ["hit", "stick"]
TERMINAL = (-1, -1)


def step(s, a):
    """
    step game
    :param s: current state, (dealer's first card, the player's sum) tuple
    :param a: selected action
    :return: (state, reward) tuple
    """
    dealer, player = s
    if a == ACTION[0]:
        # if action is hit
        player += draw_one_card()
        if busted(player):
            return TERMINAL, -1
        else:
            return (dealer, player), 0
    else:
        while dealer < 17:
            dealer += draw_one_card()
            if busted(dealer):
                return TERMINAL, 1
        if dealer > player:
            return TERMINAL, -1
        elif dealer == player:
            return TERMINAL, 0
        else:
            return TERMINAL, 1


def draw_one_card():
    number = random.randint(1, 10)
    color = random.randint(1, 3)
    # if color is red
    if color == 1:
        return -number
    else:
        return number


def busted(s):
    if s > 21 or s < 1:
        return True
    else:
        return False


def new_game():
    dealer = random.randint(1, 10)
    player = random.randint(1, 10)
    return dealer, player


if __name__ == '__main__':
    s = TERMINAL
    while True:
        a = raw_input("Your action [new/hit/stick/exit]: ")
        if a == "new":
            s = new_game()
            print(s)
        elif a == "exit":
            break
        else:
            s, r = step(s, a)
            print(s, r)
