import streamlit as st
import chess
import chess.pgn
import io
import requests
import streamlit.components.v1 as components
from groq import Groq
import json

# --- Configuration & Intelligence ---
st.set_page_config(page_title="Chess Review AI", layout="wide")

# Securely fetch Groq API Key
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- Tactical State Management ---
if 'move_index' not in st.session_state:
    st.session_state.move_index = 0
if 'history' not in st.session_state:
    st.session_state.history = []
if 'games_list' not in st.session_state:
    st.session_state.games_list = []


def get_spatial_context(board, move):
    """
    Advanced tactical scanner. Detects threats even if the piece is pinned.
    """
    # 1. Peek at the board state AFTER the move
    board.push(move)

    new_sq = move.to_square
    piece = board.piece_at(new_sq)
    piece_name = chess.piece_name(piece.piece_type).capitalize()

    # 2. X-RAY ATTACK DETECTION
    # board.attacks(square) returns all squares the piece can see/attack
    attacked_squares = board.attacks(new_sq)

    threats = []
    for sq in attacked_squares:
        target = board.piece_at(sq)
        # Check if there is an ENEMY piece on that square
        if target and target.color != piece.color:
            threats.append(f"{chess.piece_name(target.piece_type)} on {chess.square_name(sq)}")

    # 3. DEFENSE DETECTION
    # board.attackers(color, square) finds all friendly pieces guarding this spot
    defenders = []
    friendly_units = board.attackers(piece.color, new_sq)
    for sq in friendly_units:
        # Don't count the piece as defending itself
        if sq != new_sq:
            dep_piece = board.piece_at(sq)
            defenders.append(f"{chess.piece_name(dep_piece.piece_type)} on {chess.square_name(sq)}")

    # Clean up state
    board.pop()

    # Build the Intel Report
    report = f"**{piece_name} Analysis:**\n"
    report += f"- 🎯 **Threatening:** {', '.join(threats) if threats else 'No direct pieces'}\n"
    report += f"- 🛡️ **Guarded by:** {', '.join(defenders) if defenders else 'No one (Hanging!)'}"

    return report


st.title("♟️ Chess Review AI")

def log_training_data(fen, move, spatial_info, ai_response):
    """Saves the interaction in a format ready for Fine-Tuning."""
    data_entry = {
        "instruction": f"Review this chess move: {move} at FEN {fen}",
        "context": spatial_info,
        "response": ai_response
    }
    with open("chess_training_data.jsonl", "a") as f:
        f.write(json.dumps(data_entry) + "\n")

# --- Sidebar: Mission Intelligence ---
with st.sidebar:
    st.header("Fetch Chess.com Archives")
    username = st.text_input("Username", placeholder="e.g., MagnusCarlsen")
    col_yr, col_mo = st.columns(2)
    year = col_yr.selectbox("Year", ["2026", "2025", "2024"])
    month = col_mo.selectbox("Month", [f"{i:02d}" for i in range(1, 13)])

    if st.button("Fetch Games"):
        url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month}"
        headers = {"User-Agent": "ChessReviewAI-Project"}
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            st.session_state.games_list = resp.json().get('games', [])
            st.success(f"Retrieved {len(st.session_state.games_list)} games.")

    if st.session_state.games_list:
        # 1. Create a descriptive list for the dropdown
        def game_formatter(index):
            game = st.session_state.games_list[index]
            w = game['white']
            b = game['black']
            # Return a readable string for the dropdown
            return f"{w['username']} ({w['rating']}) vs {b['username']} ({b['rating']}) | {game.get('time_class', 'Chess')}"


        # 2. Use the formatter in the selectbox
        selected_idx = st.selectbox(
            "Select Target Match",
            range(len(st.session_state.games_list)),
            format_func=game_formatter
        )

        if st.button("Initialize Board"):
            selected_game = st.session_state.games_list[selected_idx]
            pgn_io = io.StringIO(selected_game['pgn'])
            game = chess.pgn.read_game(pgn_io)

            if game:
                st.session_state.history = list(game.mainline_moves())
                st.session_state.move_index = 0
                # Store player info for the AI coach context
                st.session_state.players = {
                    "white": selected_game['white']['username'],
                    "black": selected_game['black']['username']
                }
                st.success(f"Loaded: {st.session_state.players['white']} vs {st.session_state.players['black']}")

# --- Board Logic & Rendering ---
current_board = chess.Board()
for i in range(st.session_state.move_index):
    current_board.push(st.session_state.history[i])
current_fen = current_board.fen()

col1, col2 = st.columns([1.5, 1])

with col1:
    # Neutral Board Component
    board_html = f"""
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <div id="myBoard" style="width: 450px; margin: auto; border: 1px solid #ddd;"></div>
    <script>
        var board = Chessboard('myBoard', {{
            position: '{current_fen}',
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{{piece}}.png'
        }});
    </script>
    """
    components.html(board_html, height=500)

    # Navigation Controls
    c1, c2, c3, c4 = st.columns(4)
    if c2.button("‹ Prev"):
        st.session_state.move_index = max(0, st.session_state.move_index - 1)
        st.rerun()
    if c3.button("Next ›"):
        st.session_state.move_index = min(len(st.session_state.history), st.session_state.move_index + 1)
        st.rerun()

with col2:
    st.header("Llama-3 Analysis")

    if st.session_state.history and st.session_state.move_index > 0:
        # Get historical context (last 5 moves)
        h_board = chess.Board()
        for i in range(max(0, st.session_state.move_index - 6)):
            h_board.push(st.session_state.history[i])

        recent_moves = []
        for i in range(max(0, st.session_state.move_index - 6), st.session_state.move_index - 1):
            recent_moves.append(h_board.san(st.session_state.history[i]))
            h_board.push(st.session_state.history[i])

        last_move = st.session_state.history[st.session_state.move_index - 1]
        move_san = h_board.san(last_move)
        spatial_data = get_spatial_context(h_board, last_move)

        st.subheader(f"Move Played: {move_san}")
        st.write(f"**Context:** {spatial_data}")

        analysis_report = get_spatial_context(h_board, last_move)

        # Groq Prompting
        prompt = f"""
        System: You are an elite Chess Coach. 
        Context: 
        - Recent Moves: {' '.join(recent_moves)}
        - Current FEN: {current_fen}
        - Tactical Analysis: {analysis_report}

        Task: Explain the move {move_san} in a way that helps a student understand the "Why". 
        Reference the specific threats and defenders mentioned in the Tactical Analysis.
        """

        if st.button("Ask Coach"):
            with st.spinner("Analyzing maneuvers..."):
                try:
                    # 1. Inference - Using the high-precision Llama-3 model
                    completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile",
                    )

                    # 2. Capture Response
                    ai_feedback = completion.choices[0].message.content

                    # 3. Display Result (Only ONCE)
                    st.subheader("Master's Review")
                    st.info(ai_feedback)

                    # 4. Intelligence Logging (JSONL for Fine-Tuning)
                    # We wrap this in a try-block so a file error doesn't crash the UI
                    try:
                        log_training_data(current_fen, move_san, analysis_report, ai_feedback)
                    except Exception as log_error:
                        st.warning(f"Tactical Log Failed: {log_error}")

                except Exception as e:
                    st.error(f"Inference Offline: {e}")
        else:
            # This keeps the UI clean when no move is selected
            if st.session_state.move_index == 0:
                st.write("Awaiting the first move to begin analysis.")

st.divider()

st.caption("Tactical Review Engine | Groq Llama-3")
