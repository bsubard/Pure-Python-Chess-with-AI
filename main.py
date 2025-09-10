import pygame as pg
import sys
import copy
import random

# --- CONSTANTS AND GLOBALS ---
w = 1160 
h = 490
BOARD_WIDTH = 400
INFO_PANEL_WIDTH = 200
GAP = 40

BOARD_1_OFFSET_X = GAP
INFO_PANEL_OFFSET_X = BOARD_1_OFFSET_X + BOARD_WIDTH + GAP
BOARD_2_OFFSET_X = INFO_PANEL_OFFSET_X + INFO_PANEL_WIDTH + GAP


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (142, 142, 142)
SILVER = (120, 120, 120)
LIGHT = (252, 204, 116)
DARK = (87, 58, 46)
GREEN = (0, 255, 0)
RED = (215, 0, 0)
ORANGE = (255, 165, 0)
transcript, turn_number = '', 0

# --- FONT FINDER HELPER FUNCTION ---
def find_system_font():
    """Searches for a system font that supports chess characters."""
    font_preferences = ["dejavusans", "segoeuisymbol", "freeserif", "arialunicode", "symbola"]
    available_fonts = pg.font.get_fonts()
    for pref_font in font_preferences:
        if pref_font in available_fonts:
            print(f"Using system font: {pref_font}")
            return pg.font.match_font(pref_font)
    print("Warning: No suitable system font found for chess symbols. Falling back to default.")
    return None

# --- PIECE CLASS DEFINITIONS ---
class Piece:
    piece_names = ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']

    def __init__(self, colour, name, unbounded=True):
        self.colour = colour
        self.name = name
        self.unbounded = unbounded
        base_unicode = 9818 
        piece_index = self.piece_names.index(name)
        self.image = chr(base_unicode + piece_index)

    def find_moves(self, board, location, kings, check):
        x, y = location[0], location[1]
        legal_moves = []
        additional = set()
        if self.name == 'pawn':
            additional.update(self.additional_moves(board, x, y))
        for x2, y2 in self.moveset.union(additional):
            if any(i < 0 for i in (x + x2, y + y2)): continue
            try:
                coords = x + x2, y + y2
                square = board[coords[1]][coords[0]]
                if square and square.colour == self.colour:
                # pawns handle diagonal captures separately so we only do this for non-pawns
                    if self.name != 'pawn':
                        continue
                if self.name != 'pawn' and (square is None or square and square.colour != self.colour) or \
                        self.name == 'pawn' and ((x2 == 0 and square is None) or (x2, y2) in additional):
                    king = kings[int(self.colour == "black")]
                    king_pos = coords if king == (x, y) else king
                    if not board[king[1]][king[0]].in_check(board, king_pos, moved_from=location, moved_to=coords):
                        legal_moves.append(coords)
                    if square and square.colour != self.colour or coords not in legal_moves and not check: continue
                    while self.unbounded or self.name == 'pawn' and self.double_move:
                        coords = coords[0] + x2, coords[1] + y2
                        square = board[coords[1]][coords[0]]
                        if square and square.colour == self.colour:
                            break
                        if check and board[king[1]][king[0]].in_check(board, king_pos, moved_from=location, moved_to=coords): continue
                        if all(i >= 0 for i in coords) and self.name != 'pawn' and (square is None or square and square.colour != self.colour) or self.name == 'pawn' and (x2 == 0 and square is None):
                            legal_moves.append(coords)
                        elif not check: break
                        if self.name == 'pawn' or square and square.colour != self.colour: break
            except IndexError: continue
        if self.name == 'king' and not check and self.castle_rights and self.castle(board, x, y):
            legal_moves.extend(self.castle(board, x, y))
        return legal_moves

class King(Piece):
    def __init__(self, colour):
        self.back_rank = 7 if colour == 'white' else 0
        self.moveset = {(x, y) for x in range(-1, 2) for y in range(-1, 2) if x != 0 or y != 0}
        self.castle_rights = True
        super().__init__(colour, 'king', unbounded=False)
    def in_check(self, board, location, moved_from=None, moved_to=None):
        for move in self.moveset:
            coords = location
            square = board[coords[1]][coords[0]]
            while (coords != moved_to or location == moved_to) and (coords == location or coords == moved_from or square is None):
                try:
                    if any(i < 0 or i > 7 for i in (coords[0] + move[0], coords[1] + move[1])): break
                    coords = coords[0] + move[0], coords[1] + move[1]
                    square = board[coords[1]][coords[0]]
                except IndexError: break
            if square is None or square.colour == self.colour or coords == moved_to: continue
            if 0 in move and (square.name == 'rook' or square.name == 'queen') or 0 not in move and (square.name == 'bishop' or square.name == 'queen' or (square.name == 'pawn' and location[1] - coords[1] == square.direction)): return True
        for x, y in {(x, y) for x in range(-2, 3) for y in range(-2, 3) if x != 0 and y != 0 and abs(x) != abs(y)}:
            try:
                coords = location[0] + x, location[1] + y
                square = board[coords[1]][coords[0]]
                if any(i < 0 for i in (coords[0], coords[1])): continue
                if square and square.colour != self.colour and square.name == 'knight' and coords != moved_to: return True
            except IndexError: continue
        return False
    def castle(self, board, x, y):
        moves = []
        if board[self.back_rank][0] and board[self.back_rank][0].name == 'rook' and board[self.back_rank][0].castle_rights:
            squares = [(i, self.back_rank) for i in range(1, 4)]
            if all(not piece for piece in board[self.back_rank][1:4]) and all(not self.in_check(board, square) for square in squares):
                moves.append((2, self.back_rank))
        if board[self.back_rank][7] and board[self.back_rank][7].name == 'rook' and board[self.back_rank][7].castle_rights:
            squares = [(i, self.back_rank) for i in range(5, 7)]
            if all(not piece for piece in board[self.back_rank][5:7]) and all(not self.in_check(board, square) for square in squares):
                moves.append((6, self.back_rank))
        return moves

class Queen(Piece):
    def __init__(self, colour):
        self.moveset = {(x, y) for x in range(-1, 2) for y in range(-1, 2) if x != 0 or y != 0}
        super().__init__(colour, 'queen')

class Rook(Piece):
    def __init__(self, colour):
        self.moveset = {(x, y) for x in range(-1, 2) for y in range(-1, 2) if (x == 0 or y == 0) and (x != 0 or y != 0)}
        self.castle_rights = True
        super().__init__(colour, 'rook')

class Bishop(Piece):
    def __init__(self, colour):
        self.moveset = {(x, y) for x in range(-1, 2) for y in range(-1, 2) if x != 0 and y != 0}
        super().__init__(colour, 'bishop')

class Knight(Piece):
    def __init__(self, colour):
        self.moveset = {(x, y) for x in range(-2, 3) for y in range(-2, 3) if x != 0 and y != 0 and abs(x) != abs(y)}
        super().__init__(colour, 'knight', unbounded=False)

class Pawn(Piece):
    def __init__(self, colour):
        self.direction = -1 if colour == 'white' else 1
        self.moveset = {(0, y * self.direction) for y in range(1, 2)}
        self.en_passant = False
        self.double_move = True
        super().__init__(colour, 'pawn', unbounded=False)
    def additional_moves(self, board, x, y):
        valid_attacks = set()
        for n in range(-1, 2, 2):
            try:
                square = board[y + self.direction][x + n]
                if square and square.colour != self.colour:
                    valid_attacks.add((n, self.direction))
                else:
                    square = board[y][x + n]
                    if square and square.name == 'pawn' and square.en_passant:
                        valid_attacks.add((n, self.direction))
            except IndexError: pass
        return valid_attacks

# --- GAME LOGIC FUNCTIONS ---
def reset_board(with_pieces=True):
    def generate_pieces(colour):
        return [Rook(colour), Knight(colour), Bishop(colour), Queen(colour), King(colour), Bishop(colour), Knight(colour), Rook(colour)]
    board = [[None for _ in range(8)] for _ in range(8)]
    if with_pieces:
        board[0] = generate_pieces("black")
        board[7] = generate_pieces("white")
        board[1] = [Pawn("black") for _ in board[1]]
        board[6] = [Pawn("white") for _ in board[6]]
    return board

def move_piece(board, target, kings, origin, destination, captures, promotion):
    if board[destination[1]][destination[0]]:
        captures.append(board[destination[1]][destination[0]])

    promoting = False
    if target.name == 'pawn':
        if target.double_move: target.double_move = False
        if abs(origin[1] - destination[1]) == 2: target.en_passant = True
        if origin[0] != destination[0] and not board[destination[1]][destination[0]]:
            captured_pawn = board[destination[1] - target.direction][destination[0]]
            captures.append(captured_pawn)
            board[destination[1] - target.direction][destination[0]] = None
        if destination[1] == (0 if target.colour == 'white' else 7):
            promoting = True
            piece_dict = {'queen': Queen(target.colour), 'knight': Knight(target.colour), 'rook': Rook(target.colour), 'bishop': Bishop(target.colour)}
            board[destination[1]][destination[0]] = piece_dict[promotion]
    
    if not promoting:
        board[destination[1]][destination[0]] = target

    if target.name == 'king':
        kings[int(target.colour == "black")] = destination
        if target.castle_rights: target.castle_rights = False
        if destination[0] - origin[0] == 2:
            board[target.back_rank][5], board[target.back_rank][7] = board[target.back_rank][7], None
        if origin[0] - destination[0] == 2:
            board[target.back_rank][3], board[target.back_rank][0] = board[target.back_rank][0], None

    if target.name == 'rook' and target.castle_rights: target.castle_rights = False
    
    board[origin[1]][origin[0]] = None
    
    for row in board:
        for piece in row:
            if piece and piece.name == 'pawn' and piece.en_passant and piece.colour != target.colour:
                piece.en_passant = False
    
    enemy_king_idx = int(target.colour == "white")
    enemy_king_pos = kings[enemy_king_idx]
    check = board[enemy_king_pos[1]][enemy_king_pos[0]].in_check(board, enemy_king_pos)
    return board, captures, kings, check

def checkmate(board, turn, kings):
    for y, row in enumerate(board):
        for x, square in enumerate(row):
            if square and square.colour != turn:
                if square.find_moves(board, (x, y), kings, True):
                    return False
    return True

# --- DRAWING & INPUT FUNCTIONS ---
def draw_squares(screen, offset_x):
    colour_dict = {True: LIGHT, False: DARK}
    for row in range(8):
        current_colour = not (row % 2 == 0)
        for square in range(8):
            pg.draw.rect(screen, colour_dict[current_colour], (offset_x + (square * 50), 40 + (row * 50), 50, 50))
            current_colour = not current_colour

def draw_pieces(screen, font, board, flipped, offset_x):
    for row_idx, pieces_row in enumerate(board[::(-1 if flipped else 1)]):
        for col_idx, piece in enumerate(pieces_row[::(-1 if flipped else 1)]):
            if piece:
                center_pos = (offset_x + col_idx * 50 + 25, 40 + row_idx * 50 + 25)
                piece_color = (240, 240, 240) if piece.colour == 'white' else BLACK
                if piece.colour == 'white':
                    offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
                    for dx, dy in offsets:
                        border_surf = font.render(piece.image, True, BLACK)
                        screen.blit(border_surf, border_surf.get_rect(center=(center_pos[0] + dx, center_pos[1] + dy)))
                main_surf = font.render(piece.image, True, piece_color)
                screen.blit(main_surf, main_surf.get_rect(center=center_pos))

def find_square(x, y, offset_x):
    grid_x = int((x - offset_x) / 50)
    grid_y = int((y - 40) / 50)
    return grid_x, grid_y

def draw_center_divider(screen):
    pg.draw.rect(screen, SILVER, (INFO_PANEL_OFFSET_X - GAP, 0, INFO_PANEL_WIDTH + 2 * GAP, h))

def draw_captures_above_board(screen, font, captures, color_to_draw, offset_x):
    captured_pieces = [p for p in captures if p.colour == color_to_draw]
    for i, piece in enumerate(captured_pieces):
        pos = (offset_x + 5 + i * 25, 5)
        piece_color = (240, 240, 240) if piece.colour == 'white' else BLACK
        if piece.colour == 'white':
            border_surf = font.render(piece.image, True, BLACK)
            screen.blit(border_surf, (pos[0] + 1, pos[1] + 1))
        main_surf = font.render(piece.image, True, piece_color)
        screen.blit(main_surf, pos)

def draw_turn_indicator_dot(screen, turn):
    y_pos = h - 25
    radius = 6
    x_pos = (w // 2) - 90 if turn == 'white' else (w // 2) + 90
    pg.draw.circle(screen, BLACK, (x_pos, y_pos), radius)

def draw_check_highlight(screen, kings, flipped, turn, checkmate, offset_x):
    king = kings[1 if turn == 'white' else 0] if checkmate else kings[0 if turn == 'white' else 1]
    color = RED if checkmate else ORANGE
    x, y = king
    center_pos = (offset_x + (7-x if flipped else x) * 50 + 25, 40 + (7-y if flipped else y) * 50 + 25)
    pg.draw.circle(screen, color, center_pos, 25, width=3)

# --- AI: MINIMAX + ALPHA-BETA ---
piece_value = {'king': 1000, 'queen': 9, 'rook': 5, 'bishop': 3, 'knight': 3, 'pawn': 1}

def gather_all_moves_for_colour(board, colour, kings, check_flag=True):
    """Return list of (origin, dest) for colour."""
    moves = []
    for y, row in enumerate(board):
        for x, piece in enumerate(row):
            if piece and piece.colour == colour:
                legal = piece.find_moves(board, (x, y), kings, check_flag)
                for dest in legal:
                    moves.append(((x, y), dest))
    return moves

def find_kings_positions(board):
    """
    Return [white_king_pos, black_king_pos] as (x,y) tuples found on the board.
    If a king is missing for some reason, falls back to standard starting squares
    (white: (4,7), black: (4,0)) to avoid crashes.
    """
    kings = [None, None]
    for y, row in enumerate(board):
        for x, p in enumerate(row):
            if p and p.name == 'king':
                kings[int(p.colour == 'black')] = (x, y)
    if kings[0] is None:
        kings[0] = (4, 7)
    if kings[1] is None:
        kings[1] = (4, 0)
    return kings

def evaluate_material_and_mobility(board):
    """Material score from Black's perspective plus tiny mobility bonus.
       Uses real king positions so move generation doesn't crash.
    """
    score = 0
    for row in board:
        for p in row:
            if not p: continue
            val = piece_value.get(p.name, 0)
            score += val if p.colour == 'black' else -val

    # find kings on this (possibly copied) board so move generation can use them
    kings_local = find_kings_positions(board)

    # mobility bonus (small). pass the correct kings positions and check_flag=False
    black_moves = len(gather_all_moves_for_colour(board, 'black', kings_local, check_flag=False))
    white_moves = len(gather_all_moves_for_colour(board, 'white', kings_local, check_flag=False))
    score += 0.1 * (black_moves - white_moves)
    return score

def kings_adjacent(pos1, pos2):
    """Return True if kings are on adjacent squares (including diagonals)."""
    return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1


def simulate_move_and_get_state(board, kings, origin, dest, promotion_choice='queen'):
    """
    Deep-copy the board+king positions, perform origin->dest move using existing move_piece,
    and return the new board, captures list, kings, and check flag.
    Returns None if the simulated move is invalid (including if it leaves kings adjacent).
    """
    b_copy = copy.deepcopy(board)
    kings_copy = list(kings)
    captures_copy = []
    tgt = b_copy[origin[1]][origin[0]]
    if not tgt:
        return None
    try:
        b_after, caps_after, kings_after, check_after = move_piece(b_copy, tgt, kings_copy, origin, dest, captures_copy, promotion_choice)
    except Exception:
        return None

    # Reject positions where the two kings are adjacent (illegal)
    if kings_adjacent(kings_after[0], kings_after[1]):
        return None

    return b_after, caps_after, kings_after, check_after


CHECKMATE_SCORE = 10000

def minimax_ab(board, kings, depth, alpha, beta, maximizing_player, current_turn, promotion_choice='queen'):
    """
    Returns evaluation score (from Black's perspective) of the position.
    maximizing_player: True if we're maximizing for BLACK, False for WHITE.
    current_turn: whose turn it is now ('black' or 'white') in this node.
    """
    # generate moves for current_turn
    moves = gather_all_moves_for_colour(board, current_turn, kings, check_flag=True)

    # terminal conditions
    if depth == 0 or not moves:
        # if no moves -> check for checkmate/stalemate
        if not moves:
            king_idx = 0 if current_turn == 'white' else 1
            king_pos = kings[king_idx]
            # if king position seems invalid, fallback to evaluate
            try:
                in_check = board[king_pos[1]][king_pos[0]].in_check(board, king_pos)
            except Exception:
                in_check = False
            if in_check:
                # current_turn is checkmated
                return -CHECKMATE_SCORE if current_turn == 'black' else CHECKMATE_SCORE
            else:
                # stalemate -> 0 (draw)
                return 0
        return evaluate_material_and_mobility(board)

    if maximizing_player:
        value = -float('inf')
        # optional limit branching when huge
        for origin, dest in moves:
            sim = simulate_move_and_get_state(board, kings, origin, dest, promotion_choice='queen')
            if not sim: continue
            b_after, caps_after, kings_after, check_after = sim
            val = minimax_ab(b_after, kings_after, depth-1, alpha, beta, False, 'white' if current_turn == 'black' else 'black', promotion_choice)
            if val > value:
                value = val
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        # minimizing for White
        value = float('inf')
        for origin, dest in moves:
            sim = simulate_move_and_get_state(board, kings, origin, dest, promotion_choice='queen')
            if not sim: continue
            b_after, caps_after, kings_after, check_after = sim
            val = minimax_ab(b_after, kings_after, depth-1, alpha, beta, True, 'black' if current_turn == 'white' else 'white', promotion_choice)
            if val < value:
                value = val
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

def ai_pick_minimax_move(board, kings, captures, promotion, check_flag, search_depth=2):
    """
    Root that iterates black's legal moves, runs minimax with alpha-beta (depth-1 because root move consumes 1 ply),
    and returns best (origin, dest). Ties resolved by capture-value, then random.
    """
    moves = []
    for origin, dest in gather_all_moves_for_colour(board, 'black', kings, check_flag=True):
        dst_piece = board[dest[1]][dest[0]]
        is_capture = dst_piece is not None
        cap_val = piece_value[dst_piece.name] if dst_piece else 0
        moves.append((origin, dest, is_capture, cap_val))
    if not moves:
        return None

    best_move = None
    best_score = -float('inf')
    best_cap = -1

    # limit branching if insane
    MAX_ROOT = 200
    root_moves = moves if len(moves) <= MAX_ROOT else random.sample(moves, MAX_ROOT)

    for origin, dest, is_cap, cap_val in root_moves:
        sim = simulate_move_and_get_state(board, kings, origin, dest, promotion_choice='queen')
        if not sim:
            continue
        b_after, caps_after, kings_after, check_after = sim
        # Next node: it's White to move, so maximizing_player=False
        score = minimax_ab(b_after, kings_after, search_depth-1, -float('inf'), float('inf'), False, 'white', promotion_choice=promotion)
        # prefer higher score; tie-break by captured value
        if score > best_score or (score == best_score and cap_val > best_cap and is_cap):
            best_score = score
            best_move = (origin, dest)
            best_cap = cap_val

    return best_move

# --- MAIN GAME LOOP ---
def main():
    pg.init()
    pg.font.init()
    clock = pg.time.Clock()
    pg.display.set_caption('Chess - Dual View')
    screen = pg.display.set_mode((w, h))
    font_path = find_system_font() 
    piece_font = pg.font.Font(font_path, 50)
    info_font = pg.font.Font(font_path, 25)

    # AI tuning: you can increase ai_delay and ai_search_depth
    ai_delay = 3000         # milliseconds of "thinking" (increase if you raise depth)
    ai_search_depth = 2     # plies (half-moves). 3 is a cheap but effective lookahead.
                            # Try 4 if you're patient (slower), 2 if you want quicker responses.

    # --- Initial instructions ---
    print("\n--- Welcome to Chess ---")
    print("Press the number keys to select your pawn promotion piece:")
    print("1: Queen (Default)")
    print("2: Knight")
    print("3: Rook")
    print("4: Bishop")
    print("Press 'R' to reset the game at any time.")
    print("------------------------\n")

    board = reset_board()
    playing, turn, check = True, 'white', False
    kings = [(4, 7), (4, 0)]
    promotion = 'queen'
    selected_square_coords, selected_piece, captures, legal_moves = None, None, [], []

    board1_rect = pg.Rect(BOARD_1_OFFSET_X, 40, BOARD_WIDTH, BOARD_WIDTH)
    board2_rect = pg.Rect(BOARD_2_OFFSET_X, 40, BOARD_WIDTH, BOARD_WIDTH)

    # AI state
    ai_thinking = False
    ai_start_ticks = 0
    ai_precomputed_choice = None

    while True:
        screen.fill(GREY)
        draw_center_divider(screen)
        draw_squares(screen, BOARD_1_OFFSET_X)
        draw_pieces(screen, piece_font, board, False, BOARD_1_OFFSET_X)
        draw_squares(screen, BOARD_2_OFFSET_X)
        draw_pieces(screen, piece_font, board, True, BOARD_2_OFFSET_X)
        draw_captures_above_board(screen, info_font, captures, "black", BOARD_1_OFFSET_X)
        draw_captures_above_board(screen, info_font, captures, "white", BOARD_2_OFFSET_X)

        # If AI is thinking, show a small text on the center divider
        if ai_thinking:
            think_text = info_font.render("AI thinking...", True, BLACK)
            screen.blit(think_text, think_text.get_rect(center=(w//2, 20)))

        for event in pg.event.get():
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and playing:
                # ignore mouse events during AI thinking or when it's not white's turn
                if ai_thinking or turn != 'white':
                    continue

                pos = event.pos
                offset_x, flipped = (BOARD_1_OFFSET_X, False) if turn == 'white' and board1_rect.collidepoint(pos) else \
                                    (BOARD_2_OFFSET_X, True) if turn == 'black' and board2_rect.collidepoint(pos) else (None, None)
                if offset_x is not None:
                    grid_coords = find_square(pos[0], pos[1], offset_x)
                    board_coords = (7-grid_coords[0], 7-grid_coords[1]) if flipped else grid_coords
                    if selected_piece and board_coords in legal_moves:
                        board, captures, kings, check = move_piece(board, selected_piece, kings, selected_board_coords, board_coords, captures, promotion)
                        if check and checkmate(board, turn, kings):
                            playing = False
                        else:
                            turn = 'black' if turn == 'white' else 'white'

                        # start AI thinking if it became black's turn and the game is still playing
                        if turn == 'black' and playing:
                            ai_thinking = True
                            ai_start_ticks = pg.time.get_ticks()
                            # compute the minimax choice right away (may take some time)
                            ai_precomputed_choice = ai_pick_minimax_move(board, kings, captures, promotion, check, search_depth=ai_search_depth)
                        selected_piece, selected_square_coords, legal_moves = None, None, []
                    else:
                        clicked_piece = board[board_coords[1]][board_coords[0]]
                        if clicked_piece and clicked_piece.colour == turn:
                            check_flag = False if clicked_piece.name == 'king' else True
                            selected_piece, selected_square_coords, selected_board_coords, legal_moves = \
                                clicked_piece, grid_coords, board_coords, clicked_piece.find_moves(board, board_coords, kings, check_flag)
                        else:
                            selected_piece, selected_square_coords, legal_moves = None, None, []
                else: 
                    selected_piece, selected_square_coords, legal_moves = None, None, []

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_r:
                    board, kings, turn, check, captures, playing = reset_board(), [(4, 7), (4, 0)], 'white', False, [], True
                    selected_piece, selected_square_coords, legal_moves = None, None, []
                    ai_thinking = False
                    ai_precomputed_choice = None
                    print("\n--- Game Reset ---")
                
                # --- Promotion choice confirmations ---
                if event.key == pg.K_1: 
                    promotion = 'queen'
                    print("Pawn promotion chosen: Queen")
                if event.key == pg.K_2: 
                    promotion = 'knight'
                    print("Pawn promotion chosen: Knight")
                if event.key == pg.K_3: 
                    promotion = 'rook'
                    print("Pawn promotion chosen: Rook")
                if event.key == pg.K_4: 
                    promotion = 'bishop'
                    print("Pawn promotion chosen: Bishop")

            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        # If AI was thinking and enough time has passed, play precomputed move
        if ai_thinking and ai_precomputed_choice and pg.time.get_ticks() - ai_start_ticks >= ai_delay:
            origin, dest = ai_precomputed_choice
            tgt = board[origin[1]][origin[0]]
            if tgt and tgt.colour == 'black':
                board, captures, kings, check = move_piece(board, tgt, kings, origin, dest, captures, 'queen')
                if check and checkmate(board, turn, kings):
                    playing = False
                else:
                    turn = 'white'
            ai_thinking = False
            ai_precomputed_choice = None

        # If AI had no legal move
        if ai_thinking and ai_precomputed_choice is None and pg.time.get_ticks() - ai_start_ticks >= ai_delay:
            # stalemate or checkmate handling (nothing to do)
            ai_thinking = False

        if selected_piece:
            offset_x = BOARD_2_OFFSET_X if turn == 'black' else BOARD_1_OFFSET_X
            x, y = selected_square_coords
            pg.draw.rect(screen, GREEN, (offset_x + x * 50, 40 + y * 50, 50, 50), width=3)
        
        if check:
            draw_check_highlight(screen, kings, False, turn, not playing, BOARD_1_OFFSET_X)
            draw_check_highlight(screen, kings, True, turn, not playing, BOARD_2_OFFSET_X)
            
        draw_turn_indicator_dot(screen, turn)
        pg.display.update()
        clock.tick(60)

if __name__ == '__main__':
    main()
