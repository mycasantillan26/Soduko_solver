import os
import cv2
import numpy as np
import tensorflow as tf
import imutils
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 1. LOAD MODEL ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model-OCR.h5')
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)

input_size = 48

# --- 2. SOLVER LOGIC ---
def valid(board, num, pos):
    for i in range(9):
        if board[pos[0]][i] == num and pos[1] != i: return False
        if board[i][pos[1]] == num and pos[0] != i: return False
    bx, by = pos[1] // 3, pos[0] // 3
    for i in range(by*3, by*3+3):
        for j in range(bx*3, bx*3+3):
            if board[i][j] == num and (i,j) != pos: return False
    return True

def solve(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for n in range(1, 10):
                    if valid(board, n, (r, c)):
                        board[r][c] = n
                        if solve(board): return True
                        board[r][c] = 0
                return False
    return True

# --- 3. THE SMART PROCESSOR ---
def process_sudoku_smart(img_path, filename):
    image = cv2.imread(img_path)
    if image is None: return None
    
    # Pre-processing for real-world photos
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find the largest 4-sided contour
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    location = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 30000: # Threshold to ensure it's a big grid
            location = approx
            break

    # REJECTION LOGIC
    if location is None:
        return "NOT_SUDOKU"

    # Perspective Warp (Flattens the image)
    def order_points(pts):
        pts = pts.reshape((4, 2))
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order_points(location)
    dst = np.array([[0,0],[900,0],[900,900],[0,900]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, M, (900, 900))
    
    # Split and Predict
    boxes = []
    rows = np.vsplit(warped, 9)
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size)) / 255.0
            boxes.append(box)
    
    rois = np.array(boxes).reshape(-1, input_size, input_size, 1)
    preds = model.predict(rois)
    grid = [np.argmax(i) for i in preds]
    board = np.array(grid).reshape(9, 9)
    
    # Check if AI read the board correctly (no duplicates in rows/cols)
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                val = board[r][c]
                board[r][c] = 0
                if not valid(board, val, (r,c)):
                    return "NOT_SUDOKU" # Misread or invalid grid
                board[r][c] = val

    solved = board.copy()
    if solve(solved):
      # Create result image
        result_mask = np.zeros((900, 900, 3), dtype="uint8")
        for r in range(9):
            for c in range(9):
                if grid[r*9+c] == 0:
                    pos = (c*100+30, r*100+70)
                    
                    # 1. Draw the "Highlight/Glow" (Light/Neon Green, Thicker)
                    # BGR: (144, 238, 144) is Light Green
                    cv2.putText(result_mask, str(solved[r][c]), pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (144, 238, 144), 10) 
                    
                    # 2. Draw the "Main Number" (Dark Green, Thinner)
                    # BGR: (0, 100, 0) is Dark Green
                    cv2.putText(result_mask, str(solved[r][c]), pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 0), 3)
                
        # Warp back to original
        M_inv = cv2.getPerspectiveTransform(dst, rect)
        inv_warp = cv2.warpPerspective(result_mask, M_inv, (image.shape[1], image.shape[0]))
        final = cv2.addWeighted(image, 0.8, inv_warp, 1, 0)
        
        out_name = "solved_" + filename
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], out_name), final)
        return out_name

    return "NOT_SUDOKU"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve_api():
    file = request.files.get('file')
    if not file: return jsonify({"error": "No file"})
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    
    result = process_sudoku_smart(path, file.filename)
    
    if result == "NOT_SUDOKU":
        return jsonify({"message": "This is not a Sudoku image. Please try a clearer photo."})
    
    return jsonify({"message": "Success", "image_url": f"/static/{result}"})

if __name__ == '__main__':
    app.run(debug=True)