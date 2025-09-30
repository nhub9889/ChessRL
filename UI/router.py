from flask import Flask, render_template, jsonify, request, send_from_directory
import time
import threading
import torch
from UI.pieces import Board, Queen, Pawn
from pyngrok import ngrok
from src.MCTS import MCTS
from src.model import Model
from src.pipelines import ChessState
ngrok.set_auth_token("2oEbq4cHUSHLPp6lnjVlx5lcPJT_4AvatxNjPDfVfGSJ6UmcD")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)
model.load("chessRL_100.pth")  # đường dẫn checkpoint
mcts = MCTS(model, simulations=200) 
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chess.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/board')
def getBoard():
    global currentBoard
    if currentBoard:
        return jsonify(currentBoard.toDict())


@app.route('/move', methods=['POST'])
def makeMove():
    global currentBoard, history
    data = request.json
    fr = data.get('from')
    to = data.get('to')

    frx = ord(fr[0]) - ord('a')
    fry = 8 - int(fr[1])
    tox = ord(to[0]) - ord('a')
    toy = 8 - int(to[1])

    success = currentBoard.Move((frx, fry), (tox, toy))

    if success:
        piece = currentBoard.getPiece(tox, toy)
        if isinstance(piece, Pawn) and (toy == 0 or toy == 7):
            currentBoard.promote(tox, toy, Queen)

        history.append({
            'from': fr,
            'to': to,
            'player': 'human'
        })

        response = {'success': True, 'board': currentBoard.toDict()}
        if currentBoard.checkmate('W'):
            response.update({'result': 'checkmate', 'winner': 'black'})
        elif currentBoard.checkmate('B'):
            response.update({'result': 'checkmate', 'winner': 'white'})

        return jsonify(response)
    return jsonify({'success': False})


@app.route('/new_game')
def new_game():
    global currentBoard, history
    currentBoard = Board()
    history = []

    return jsonify({'success': True, 'board': currentBoard.toDict()})

@app.route('/history')
def history():
    return jsonify(history)

@app.route('/bot_move')
def bot_move():
    global currentBoard, history

    # chạy MCTS từ trạng thái hiện tại
    state = ChessState(currentBoard)
    action_probs = mcts.run(state)

    if not action_probs:
        return jsonify({'success': False, 'message': 'No valid moves'})

    # chọn action có xác suất cao nhất
    action = max(action_probs, key=action_probs.get)
    (frx, fry), (tox, toy) = action

    success = currentBoard.Move((frx, fry), (tox, toy))
    if not success:
        return jsonify({'success': False, 'message': 'Invalid move chosen by bot'})

    piece = currentBoard.getPiece(tox, toy)
    if isinstance(piece, Pawn) and (toy == 0 or toy == 7):
        currentBoard.promote(tox, toy, Queen)

    # lưu history
    history.append({
        'from': f"{chr(ord('a')+frx)}{8-fry}",
        'to': f"{chr(ord('a')+tox)}{8-toy}",
        'player': 'bot'
    })

    response = {'success': True, 'board': currentBoard.toDict()}
    if currentBoard.checkmate('W'):
        response.update({'result': 'checkmate', 'winner': 'black'})
    elif currentBoard.checkmate('B'):
        response.update({'result': 'checkmate', 'winner': 'white'})

    return jsonify(response)


@app.route('/valid_moves', methods =['GET'])
def valid_moves():
    global currentBoard
    from_square = request.args.get('from')

    x = ord(from_square[0]) - ord('a')
    y = 8 - int(from_square[1])

    piece = currentBoard.getPiece(x, y)
    if not piece:
        return jsonify({'valid_moves': []})

    moves = currentBoard.validMoves(x, y)
    movesNotation = []
    for move in moves:
        movesNotation.append({'to': f"{chr(ord('a') + move[0])}{8 - move[1]}"})
    return jsonify({'valid_moves': movesNotation})
def run():
    app.run(debug= True, use_reloader= False, host= '0.0.0.0', port=5000)

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print("Public URL:", public_url)
    thread = threading.Thread(target= run)
    thread.daemon = True
    thread.start()

    currentBoard = Board()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down")

