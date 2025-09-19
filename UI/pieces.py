class Board():
    def __init__(self):
        self.squares = [[None for _ in range(8)] for _ in range(8)]
        self.curPlayer = 'W'
        self.history = []
        self.captured = {'W': [], 'B': []}
        self.pic = '/static/ChessAssets/Board.png'
        self.setupBoard()

    def stalemate(self, color):
        if self.isChecked(color):
            return False

        for x in range(8):
            for y in range(8):
                piece = self.getPiece(x, y)
                if piece and piece.color == color:
                    if self.validMoves(x, y):
                        return False

        return True

    def getResult(self):
        if self.checkmate('W'):
            return 'Black wins by checkmate'
        elif self.checkmate('B'):
            return 'White wins by checkmate'
        elif self.stalemate('W') or self.stalemate('B'):
            return 'Draw by stalemate'

        else:
            return None

    def toDict(self):
        board_dict = {
            'squares': [[None for _ in range(8)] for _ in range(8)],
            'current_player': self.curPlayer,
            'captured_white': [str(p) for p in self.captured['W']],
            'captured_black': [str(p) for p in self.captured['B']]
        }

        for y in range(8):  # Fix: iterate over y first
            for x in range(8):
                piece = self.getPiece(x, y)
                if piece:
                    board_dict['squares'][x][y] = {  # Fix: use x, y instead of y, x
                        'type': piece.__class__.__name__.lower(),
                        'color': piece.color,
                        'image': piece.pic,
                    }

        return board_dict

    def setupBoard(self):
        for i in range(8):
            self.squares[i][1] = Pawn('B', i, 1)
            self.squares[i][6] = Pawn('W', i, 6)

        pieces = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        for i, piece_class in enumerate(pieces):
            self.squares[i][0] = piece_class('B', i, 0)
            self.squares[i][7] = piece_class('W', i, 7)

    def getPiece(self, x, y):
        if 0 <= x < 8 and 0 <= y < 8:
            return self.squares[x][y]
        return None

    def copy(self):
        nboard = Board()
        nboard.squares = [[None for _ in range(8)] for _ in range(8)]

        for x in range(8):
            for y in range(8):
                piece = self.getPiece(x, y)
                if piece:
                    npiece = piece.__class__(piece.color, piece.x, piece.y)
                    npiece.isMoved = piece.isMoved
                    nboard.squares[x][y] = npiece

        nboard.curPlayer = self.curPlayer
        nboard.history = self.history.copy()
        nboard.captured = {
            'W': self.captured['W'].copy(),
            'B': self.captured['B'].copy()
        }

        return nboard

    def promote(self, x, y, piece_type=None):
        pawn = self.getPiece(x, y)
        if not isinstance(pawn, Pawn):
            return False

        if piece_type is None:
            piece_type = Queen

        if piece_type == Queen:
            new_piece = Queen(pawn.color, x, y)
        elif piece_type == Rook:
            new_piece = Rook(pawn.color, x, y)
        elif piece_type == Bishop:
            new_piece = Bishop(pawn.color, x, y)
        elif piece_type == Knight:
            new_piece = Knight(pawn.color, x, y)
        else:
            return False

        self.squares[x][y] = new_piece
        return True

    def Move(self, from_pos, to_pos):
        from_x, from_y = from_pos
        to_x, to_y = to_pos

        piece = self.getPiece(from_x, from_y)
        if piece is None or piece.color != self.curPlayer:
            return False

        if to_pos not in piece.moves(self):
            return False

        isCastling = isinstance(piece, King) and abs(from_x - to_x) == 2

        captured = self.getPiece(to_x, to_y)
        self.history.append((from_pos, to_pos, captured, isCastling))

        if captured:
            self.captured[self.curPlayer].append(captured)

        self.squares[to_x][to_y] = piece
        self.squares[from_x][from_y] = None
        piece.move(to_x, to_y)

        if isCastling:
            if to_x == 6:  # Kingside castling
                rook = self.getPiece(7, from_y)
                self.squares[5][from_y] = rook
                self.squares[7][from_y] = None
                rook.move(5, from_y)
            elif to_x == 2:  # Queenside castling
                rook = self.getPiece(0, from_y)
                self.squares[3][from_y] = rook
                self.squares[0][from_y] = None
                rook.move(3, from_y)

        self.curPlayer = 'B' if self.curPlayer == 'W' else 'W'
        return True

    def wouldCheck(self, color, from_pos, to_pos):
        temp = Board()
        temp.squares = [[None for _ in range(8)] for _ in range(8)]
        temp.curPlayer = self.curPlayer

        for x in range(8):
            for y in range(8):
                piece = self.getPiece(x, y)
                if piece:
                    new_piece = piece.__class__(piece.color, x, y)
                    new_piece.isMoved = piece.isMoved
                    temp.squares[x][y] = new_piece

        from_x, from_y = from_pos
        to_x, to_y = to_pos
        piece = temp.getPiece(from_x, from_y)

        if piece:
            temp.squares[from_x][from_y] = None
            temp.squares[to_x][to_y] = piece
            piece.x, piece.y = to_x, to_y

        return temp.isChecked(color)

    def isChecked(self, color):
        king_pos = None
        for x in range(8):
            for y in range(8):
                piece = self.getPiece(x, y)
                if piece and isinstance(piece, King) and piece.color == color:
                    king_pos = (x, y)
                    break
            if king_pos:
                break

        if not king_pos:
            return False

        oppColor = 'B' if color == 'W' else 'W'
        pawn_direction = 1 if oppColor == 'B' else -1
        for dx in [-1, 1]:
            attack_x, attack_y = king_pos[0] + dx, king_pos[1] + pawn_direction
            if 0 <= attack_x < 8 and 0 <= attack_y < 8:
                piece = self.getPiece(attack_x, attack_y)
                if piece and piece.color == oppColor and isinstance(piece, Pawn):
                    return True
        knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        for dx, dy in knight_moves:
            attack_x, attack_y = king_pos[0] + dx, king_pos[1] + dy
            if 0 <= attack_x < 8 and 0 <= attack_y < 8:
                piece = self.getPiece(attack_x, attack_y)
                if piece and piece.color == oppColor and isinstance(piece, Knight):
                    return True
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Rook and queen
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Bishop and queen
        ]

        for dx, dy in directions:
            for i in range(1, 8):
                attack_x, attack_y = king_pos[0] + i * dx, king_pos[1] + i * dy
                if not (0 <= attack_x < 8 and 0 <= attack_y < 8):
                    break

                piece = self.getPiece(attack_x, attack_y)
                if piece:
                    if piece.color == oppColor:
                        piece_type = type(piece).__name__
                        if (dx == 0 or dy == 0) and piece_type in ['Rook', 'Queen']:
                            return True
                        if (dx != 0 and dy != 0) and piece_type in ['Bishop', 'Queen']:
                            return True
                    break
        return False

    def validMoves(self, x, y):
        piece = self.getPiece(x, y)
        if not piece:
            return []
        moves = piece.moves(self)

        valid = []
        for move in moves:
            to_x, to_y = move
            if not self.wouldCheck(color= piece.color, from_pos= (x,y), to_pos= (to_x, to_y)):
                valid.append(move)

        return valid

    def checkmate(self, color):
        if not self.isChecked(color):
            return False

        for x in range(8):
            for y in range(8):
                piece = self.getPiece(x, y)
                if piece and piece.color == color:
                    for move in piece.moves(self):
                        if not self.wouldCheck(color, (x, y), move):
                            return False
        return True


class Piece():
    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y
        self.isMoved = False
        piece_type = self.__class__.__name__.lower()
        self.pic = f'/static/ChessAssets/{piece_type}{"W" if color == "W" else "B"}.png'

    def __repr__(self):
        # Standard chess piece abbreviations
        abbreviations = {
            'King': 'K',
            'Queen': 'Q',
            'Rook': 'R',
            'Bishop': 'B',
            'Knight': 'N',
            'Pawn': 'P'
        }
        class_name = self.__class__.__name__
        abbreviation = abbreviations.get(class_name, class_name[0])
        return f"{self.color}{abbreviation}"

    def move(self, x, y):
        self.x = x
        self.y = y
        self.isMoved = True

    def moves(self, board):
        raise NotImplementedError('moves not implemented')


class Queen(Piece):
    def moves(self, board):
        rook = Rook(self.color, self.x, self.y)
        bishop = Bishop(self.color, self.x, self.y)
        return rook.moves(board) + bishop.moves(board)


class King(Piece):
    def moves(self, board):
        moves = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                target = board.getPiece(new_x, new_y)
                if target is None or target.color != self.color:
                    moves.append((new_x, new_y))

        # Add castling moves
        moves.extend(self.castling(board))
        return moves

    def castling(self, board):
        moves = []
        if self.isMoved or board.isChecked(self.color):
            return moves

        # Kingside castling
        kingside_rook = board.getPiece(7, self.y)
        if (kingside_rook and isinstance(kingside_rook, Rook) and
                not kingside_rook.isMoved and
                board.getPiece(5, self.y) is None and
                board.getPiece(6, self.y) is None and
                not board.wouldCheck(self.color, (self.x, self.y), (5, self.y)) and
                not board.wouldCheck(self.color, (self.x, self.y), (6, self.y))):
            moves.append((6, self.y))

        # Queenside castling
        queenside_rook = board.getPiece(0, self.y)
        if (queenside_rook and isinstance(queenside_rook, Rook) and
                not queenside_rook.isMoved and
                board.getPiece(1, self.y) is None and
                board.getPiece(2, self.y) is None and
                board.getPiece(3, self.y) is None and
                not board.wouldCheck(self.color, (self.x, self.y), (3, self.y)) and
                not board.wouldCheck(self.color, (self.x, self.y), (2, self.y))):
            moves.append((2, self.y))

        return moves


class Bishop(Piece):
    def moves(self, board):
        moves = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dx, dy in directions:
            for i in range(1, 8):
                new_x, new_y = self.x + i * dx, self.y + i * dy
                if not (0 <= new_x < 8 and 0 <= new_y < 8):
                    break

                target = board.getPiece(new_x, new_y)
                if target is None:
                    moves.append((new_x, new_y))
                elif target.color != self.color:
                    moves.append((new_x, new_y))
                    break
                else:
                    break
        return moves


class Rook(Piece):
    def moves(self, board):
        moves = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in directions:
            for i in range(1, 8):
                new_x, new_y = self.x + i * dx, self.y + i * dy
                if not (0 <= new_x < 8 and 0 <= new_y < 8):
                    break

                target = board.getPiece(new_x, new_y)
                if target is None:
                    moves.append((new_x, new_y))
                elif target.color != self.color:
                    moves.append((new_x, new_y))
                    break
                else:
                    break
        return moves


class Knight(Piece):
    def moves(self, board):
        moves = []
        knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]

        for dx, dy in knight_moves:
            new_x, new_y = self.x + dx, self.y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                target = board.getPiece(new_x, new_y)
                if target is None or target.color != self.color:
                    moves.append((new_x, new_y))
        return moves


class Pawn(Piece):
    def moves(self, board):
        moves = []
        direction = 1 if self.color == 'B' else -1
        if 0 <= self.y + direction < 8 and board.getPiece(self.x, self.y + direction) is None:
            moves.append((self.x, self.y + direction))
            if not self.isMoved and 0 <= self.y + 2 * direction < 8 and board.getPiece(self.x,
                                                                                       self.y + 2 * direction) is None:
                moves.append((self.x, self.y + 2 * direction))
        for dx in [-1, 1]:
            if 0 <= self.x + dx < 8 and 0 <= self.y + direction < 8:
                target = board.getPiece(self.x + dx, self.y + direction)
                if target and target.color != self.color:
                    moves.append((self.x + dx, self.y + direction))

        return moves