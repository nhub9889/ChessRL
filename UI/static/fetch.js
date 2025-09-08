let selectedPiece = null;
let validMoves = [];

function initBoard(){
    fetch('/new_game').then(res => res.json())
        .then(data => {
            if (data.success){
                renderBoard(data.board);
            }
        })
}

function getPiece(code){
    switch(code.toLowerCase()){
        case 'p': return 'Pawn';
        case 'n': return 'Knight';
        case 'b': return 'Bishop';
        case 'r': return 'Rook';
        case 'q': return 'Queen';
        case 'k': return 'King';
        default: return null;
    }
}

function renderBoard(boardData){
    const pieces = document.getElementById('pieces-layer');
    pieces.innerHTML = '';

    let player = 'White';
    if (boardData.current_player === 'B') player = 'Black';
    const statusElement = document.getElementById('status');
    if (statusElement.textContent.includes('Checkmate') || statusElement.textContent.includes('Stalemate')) {
    } else {
        statusElement.textContent = `${player}'s turn`;
    }

    updateCaptured(boardData.captured_white, 'white');
    updateCaptured(boardData.captured_black, 'black');

    for (let y = 0; y < 8; y++){
        for (let x = 0; x < 8; x++){
            const square = document.createElement('div');
            square.className = 'square';
            square.dataset.x = x;
            square.dataset.y = y;

            const piece = boardData.squares[x][y];
            if (piece){
                const img = document.createElement('img');
                img.src = piece.image;
                img.className = 'piece';
                img.dataset.type = piece.type;
                img.dataset.color = piece.color;
                img.dataset.x = x;
                img.dataset.y = y;

                if (!statusElement.textContent.includes('Checkmate') &&
                    !statusElement.textContent.includes('Stalemate')) {
                    img.addEventListener('click', handlePieceClick);
                }
                square.appendChild(img);
            }
            if (!statusElement.textContent.includes('Checkmate') &&
                !statusElement.textContent.includes('Stalemate')) {
                square.addEventListener('click', handleSquareClick);
            }

            pieces.appendChild(square);
        }
    }
}

function updateCaptured(capturedPieces, color){
    const container = document.getElementById(`captured-${color}`);
    container.innerHTML = '';

    if (!capturedPieces) return;

    capturedPieces.forEach(pieceCode => {
        const pieceColor = pieceCode[0] === 'W' ? 'white' : 'black';
        const type = getPiece(pieceCode[1]);

        if (type){
            const img = document.createElement('img');
            img.src = `/static/ChessAssets/${type}${pieceColor === 'white' ? 'W' : 'B'}.png`;
            img.className = 'captured-piece';
            img.title = `${pieceColor} ${type}`;
            container.appendChild(img);
        }
    });
}

function getValidMoves(x, y){
    const fromNotation = `${String.fromCharCode(97 + x)}${8 - y}`;

    fetch(`/valid_moves?from=${fromNotation}`)
        .then(res => res.json())
        .then(data => {
            if (data.valid_moves){
                validMoves = data.valid_moves;
                highlightValidMoves(validMoves);
            }
        })
}

function highlightValidMoves(moves){
    document.querySelectorAll('.valid-move').forEach(el => {
        el.classList.remove('valid-move');
    });

    moves.forEach(move => {
        const to = move.to;
        const x = to.charCodeAt(0) - 97;
        const y = 8 - parseInt(to[1]);
        const square = document.querySelector(`.square[data-x="${x}"][data-y="${y}"]`);
        if (square) square.classList.add('valid-move');
    });
}

function handlePieceClick(e){
    const piece = e.target;
    const x = parseInt(piece.dataset.x);
    const y = parseInt(piece.dataset.y);
    const color = piece.dataset.color;

    // Get the current player from the board
    const currentPlayer = document.getElementById('status').textContent.includes('White') ? 'W' : 'B';

    // Only allow selecting pieces of the current player
    if (color === currentPlayer){
        document.querySelectorAll('.piece.selected').forEach(p => {
            p.classList.remove('selected');
        });

        piece.classList.add('selected');
        selectedPiece = {x, y, element: piece};

        getValidMoves(x, y);
    }
    e.stopPropagation();
}

function handleSquareClick(e){
    if (!selectedPiece) return;

    const square = e.target.classList.contains('square') ? e.target : e.target.parentElement;
    const x = parseInt(square.dataset.x);
    const y = parseInt(square.dataset.y);

    const fromNotation = `${String.fromCharCode(97 + selectedPiece.x)}${8 - selectedPiece.y}`;
    const toNotation = `${String.fromCharCode(97 + x)}${8 - y}`;

    const isValidMove = validMoves.some(move => move.to === toNotation);

    if (isValidMove) {
        fetch('/move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                from: fromNotation,
                to: toNotation
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success){
                renderBoard(data.board);

                if (data.result === 'checkmate') {
                    document.getElementById('status').textContent = `Checkmate! ${data.winner} wins!`;
                    document.querySelectorAll('.piece, .square').forEach(el => {
                        el.style.pointerEvents = 'none';
                    });
                } else if (data.result === 'stalemate') {
                    document.getElementById('status').textContent = 'Stalemate! The game is a draw.';
                    document.querySelectorAll('.piece, .square').forEach(el => {
                        el.style.pointerEvents = 'none';
                    });
                }
            }
            if (selectedPiece && selectedPiece.element) {
                selectedPiece.element.classList.remove('selected');
            }
            selectedPiece = null;
            document.querySelectorAll('.valid-move').forEach(el => {
                el.classList.remove('valid-move');
            });
        });
    } else {
        if (selectedPiece && selectedPiece.element) {
            selectedPiece.element.classList.remove('selected');
        }
        selectedPiece = null;
    }
}

document.getElementById('new-game').addEventListener('click', initBoard);
initBoard();