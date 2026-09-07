import chess
import chess.engine
import chess.pgn
import random
import sys
import os
import argparse

# ==========================================
# CONFIGURATION
# ==========================================

STOCKFISH_PATH = "/usr/local/bin/stockfish"

TARGET_PUZZLES = 100

# Standard Generation Parameters (Fixed)
STD_WHITE_ELO = 2000
STD_BLACK_ELO = 1600
MOVE_TIME = 0.2

# Opening Parameters
OPENING_PLIES = 8    
OPENING_CANDIDATES = 4
OPENING_MAX_LOSS = 20 

# Puzzle Logic
DEEP_DEPTH = 18      
MIN_ADVANTAGE = -100 
MAX_SECOND_BEST = 80 
MIN_DIFF = 180       

# Piece Values for Heuristics
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
}

# ==========================================
# HEURISTICS
# ==========================================

def is_trivial_tactic(board, move):
    """Returns True if the move is 'too obvious' (Free lunch/bad trade)."""
    if board.is_capture(move):
        attackers = board.attackers(not board.turn, move.to_square)
        if not attackers:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if not is_mate:
                return True 

        if not board.is_en_passant(move):
            attacker_piece = board.piece_at(move.from_square)
            victim_piece = board.piece_at(move.to_square)
            if attacker_piece and victim_piece:
                att_val = PIECE_VALUES.get(attacker_piece.piece_type, 0)
                vic_val = PIECE_VALUES.get(victim_piece.piece_type, 0)
                if vic_val > att_val + 1:
                    return True
    return False

def is_sacrifice(board, move):
    """Returns True if move puts piece in danger from lower value piece."""
    moving_piece = board.piece_at(move.from_square)
    if not moving_piece: return False
    val_moving = PIECE_VALUES.get(moving_piece.piece_type, 0)
    
    attackers = board.attackers(not board.turn, move.to_square)
    for sq in attackers:
        attacker_piece = board.piece_at(sq)
        if attacker_piece:
            val_attacker = PIECE_VALUES.get(attacker_piece.piece_type, 0)
            if val_attacker < val_moving:
                return True
    return False

# ==========================================
# LOGIC
# ==========================================

def get_engine():
    try:
        return chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except FileNotFoundError:
        print(f"ERROR: Stockfish not found at {STOCKFISH_PATH}")
        sys.exit(1)

def play_game(gen_engine, game_number, chaos_mode=False):
    board = chess.Board()
    game_history = []
    
    # ----------------------------------------------------
    # OPENING PHASE
    # ----------------------------------------------------
    if chaos_mode:
        # CHAOS MODE: Drunken Walk (12 plies / 6 moves)
        # Plays completely random legal moves to break all theory
        chaos_plies = 12
        for _ in range(chaos_plies):
            if board.is_game_over(): break
            legal_moves = list(board.legal_moves)
            if legal_moves:
                board.push(random.choice(legal_moves))
                game_history.append(board.copy())
    else:
        # STANDARD MODE: Semi-random solid opening
        for _ in range(OPENING_PLIES):
            if board.is_game_over(): break
            
            # Alternate start
            if len(game_history) == 0:
                move = chess.Move.from_uci("e2e4") if game_number % 2 != 0 else chess.Move.from_uci("d2d4")
                board.push(move)
                game_history.append(board.copy())
                continue

            limit = chess.engine.Limit(time=0.1) 
            info = gen_engine.analyse(board, limit, multipv=OPENING_CANDIDATES)
            candidates = []
            if info:
                best_score = info[0]["score"].relative.score(mate_score=10000)
                for line in info:
                    if "pv" not in line: continue
                    score = line["score"].relative.score(mate_score=10000)
                    if score >= best_score - OPENING_MAX_LOSS:
                        candidates.append(line["pv"][0])
            
            if not candidates and list(board.legal_moves):
                candidates = [random.choice(list(board.legal_moves))]
                
            if candidates:
                move = random.choice(candidates)
                board.push(move)
                game_history.append(board.copy())

    # ----------------------------------------------------
    # MIDGAME PHASE (Variable Elo in Chaos Mode)
    # ----------------------------------------------------
    
    # In chaos mode, randomize the strength for every game to create mismatches
    if chaos_mode:
        w_elo = random.randint(1500, 2500)
        b_elo = random.randint(1400, 2000)
    else:
        w_elo = STD_WHITE_ELO
        b_elo = STD_BLACK_ELO

    while not board.is_game_over(claim_draw=True) and len(game_history) < 150:
        current_elo = w_elo if board.turn == chess.WHITE else b_elo
        gen_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": current_elo})
        result = gen_engine.play(board, chess.engine.Limit(time=MOVE_TIME))
        
        if result.move is None: break
        board.push(result.move)
        game_history.append(board.copy())
        
    return game_history, board, w_elo, b_elo

def evaluate_missed(analyzer_engine, board, played_move):
    if board.is_game_over(): return None

    try:
        # DEEP ANALYSIS
        deep_info = analyzer_engine.analyse(board, chess.engine.Limit(depth=DEEP_DEPTH), multipv=2)
        if len(deep_info) < 2: return None
        
        best_score = deep_info[0]["score"].white().score(mate_score=10000)
        second_score = deep_info[1]["score"].white().score(mate_score=10000)
        
        if board.turn == chess.BLACK:
            best_score, second_score = -best_score, -second_score

        # Check Gap
        if best_score < MIN_ADVANTAGE: return None
        if second_score > MAX_SECOND_BEST: return None
        if (best_score - second_score) < MIN_DIFF: return None

        best_move = deep_info[0]["pv"][0]

        if played_move == best_move: return None

        if is_trivial_tactic(board, best_move):
            return None 

        tags = []
        if is_sacrifice(board, best_move): tags.append("Sacrifice")
        if not board.is_capture(best_move) and not board.gives_check(best_move): tags.append("Quiet Move")
        if board.gives_check(best_move): tags.append("Check")

        return {
            "fen": board.fen(),
            "turn": "White" if board.turn == chess.WHITE else "Black",
            "solution": best_move,
            "missed_move": played_move,
            "score": best_score,
            "tags": ", ".join(tags) if tags else "Positional"
        }

    except Exception:
        return None

# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chaos", action="store_true", help="Enable 'Drunken Walk' opening and random Elos")
    args = parser.parse_args()

    print(f"Initializing Generator...")
    gen_engine = get_engine()
    print("Initializing Analyzer...")
    eval_engine = get_engine()
    
    puzzles_found = 0
    games_played = 0
    all_puzzles_list = []
    
    if args.chaos:
        print("\n🔥 CHAOS MODE ENABLED 🔥")
        print("- Opening: 12 plies of pure random moves")
        print("- Elos: Randomized per game")
    else:
        print(f"\nStandard Mode: {STD_WHITE_ELO} vs {STD_BLACK_ELO}")

    print(f"Searching for {TARGET_PUZZLES} High-Quality Puzzles...\n")

    try:
        while puzzles_found < TARGET_PUZZLES:
            games_played += 1
            print(f"Simulating Game #{games_played}...", end="\r")
            
            _, final_board, w_elo, b_elo = play_game(gen_engine, games_played, args.chaos)
            
            # Replay and Analyze
            replay_board = chess.Board()
            for move in final_board.move_stack:
                if replay_board.fullmove_number >= 10:
                    result = evaluate_missed(eval_engine, replay_board, move)
                    
                    if result:
                        puzzles_found += 1
                        all_puzzles_list.append(result)
                        
                        print(f"\n✅ PUZZLE #{puzzles_found} ({result['tags']}) [Game Elo: {w_elo}v{b_elo}]")
                        print(f"FEN: {result['fen']}")
                        print(f"Solution: {result['solution']} (Score: {result['score']})")
                        print("-" * 50)
                        
                        if puzzles_found >= TARGET_PUZZLES: break
                
                replay_board.push(move)
                if puzzles_found >= TARGET_PUZZLES: break
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        gen_engine.quit()
        eval_engine.quit()
        
        if all_puzzles_list:
            print("\n" + "="*70)
            print(f" SUMMARY OF {len(all_puzzles_list)} PUZZLES FOUND")
            print("="*70)
            for i, p in enumerate(all_puzzles_list, 1):
                print(f"Puzzle #{i} | Turn: {p['turn']} | Type: {p['tags']}")
                print(f"FEN:      {p['fen']}")
                print(f"Solution: {p['solution']}")
                print("-" * 70)
        else:
            print("\nNo puzzles found.")

if __name__ == "__main__":
    main()