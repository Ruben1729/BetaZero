import numpy as np

BOARD_SIZE = 64

col_dict = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'g',
    7: 'h',
}

chess_dict = {
    'p': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'P': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'n': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'b': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'r': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'k': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    '.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


def encode_observation(observation):
    pgn = observation.epd()
    board = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        board_row = []
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    board_row.extend(chess_dict['.'])
            else:
                board_row.extend(chess_dict[thing])

        board.extend(board_row)

    return board


def encode_action(board, action):
    str_action = board.uci(action)
    move_i = str_action[0:2]
    move_f = str_action[2:4]

    initial_pos = list(col_dict.values()).index(move_i[0]) + ((int(move_i[1]) - 1) * 8)
    final_pos = list(col_dict.values()).index(move_f[0]) + ((int(move_f[1]) - 1) * 8)

    return final_pos + (initial_pos * 63)


def decode_action(board, num_action):
    # Get the initial position index -> floor(current_position / (BOARD_SIZE * 63))
    initial_position = np.floor(num_action / 63)

    # To translate to row and col. For row you do floor(index / 8), for col you do index % 8
    move = col_dict[initial_position % 8] + str(int(np.floor(initial_position / 8)) + 1)

    # Get the final position current_position % 63
    final_position = num_action % 63
    # To translate to row and col. For row you do floor(index / 8), for col you do index % 8
    move += col_dict[final_position % 8] + str(int(np.floor(final_position / 8)) + 1)

    try:
        action = board.parse_uci(move)
        return action
    except ValueError:
        return None
