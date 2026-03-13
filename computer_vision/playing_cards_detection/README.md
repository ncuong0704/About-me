# Playing Cards Detection (YOLOv8 + Poker Hand)

This is a demo project for **detecting Western playing cards** (52 classes) using **YOLOv8 (Ultralytics)** from a video or webcam stream, and then **predicting the Poker hand type** (e.g., Royal Flush, Straight Flush, etc.) from exactly **5 detected cards** in each frame.

## Overview

- **Detection**: `playingCardsDetection.py`
  - Reads video or webcam frames with OpenCV.
  - Runs the YOLO model to detect playing cards, producing a `cardList` like `["10h", "Jh", "Qh", ...]`.
  - Draws bounding boxes and confidence labels on the image.
  - Calls `findPokerHand(cardList)` to evaluate the Poker hand and display the result on the screen.
- **Poker Hand Evaluation**: `playingCardsFunc.py`
  - Normalizes the list of 5 cards (removes duplicates) and classifies the Poker hand according to basic ranking.

## Folder Structure

```text
computer_vision/Playing Cards Detection/
├─ playingCardsDetection.py
└─ playingCardsFunc.py
```

## Requirements

- Python 3.9+ (recommended)
- Libraries:
  - `ultralytics` (YOLOv8)
  - `torch`
  - `opencv-python`
  - `cvzone`

Quick install:

```bash
pip install -r requirements.txt
```

## Model & Data Preparation

The default settings in `playingCardsDetection.py` are:

- Model: `YOLO('yolov8m_synthetic.pt')`
- Video input: `cv2.VideoCapture("../videos/poker-1.mp4")`

Note: This folder **does not include** the file `yolov8m_synthetic.pt` or the `../videos/` directory by default. You should:

- **Place the weights file** `yolov8m_synthetic.pt` in your current working directory, or update the path in the code as needed.
- **Place your video file** `poker-1.mp4` inside `computer_vision/videos/` (to match the current code), or update the video path accordingly.

To use your webcam instead, change in the code like below:

```python
cap = cv2.VideoCapture(0)
```

## How to Run the Demo

From the `computer_vision/Playing Cards Detection/` directory:

```bash
python playingCardsDetection.py
```

A window named `Image` will open and display the processed frames continuously.

## Card Class Format

In `playingCardsDetection.py`, the `classnames` use this format:

- Rank: `2..10`, `J`, `Q`, `K`, `A`
- Suit:
  - `h`: hearts
  - `d`: diamonds
  - `c`: clubs
  - `s`: spades

Example: `10h` (Ten of Hearts), `As` (Ace of Spades), `Qd` (Queen of Diamonds).

## Poker Hand Inference

The function `findPokerHand(hand)`:

- **Removes duplicates** using `list(dict.fromkeys(hand))`
- If, after removing duplicates, the number of cards is **not exactly 5**, it returns `"No cards"`.
- Otherwise, it evaluates the hand as one of:
  - Flush
  - Straight
  - Four of a kind, Full house, Three of a kind, Two pair, One pair, High card

## Limitations & Notes

- **Detection quality matters**: If YOLO detects too many/few or duplicate cards in a frame, the function will return `"No cards"` because there aren't exactly 5 unique cards.
- **Low Ace straight (A-2-3-4-5)**: The current logic checks for straights in increasing order (with A=14), so “wheel straight” (A low) is **not yet supported**.
- **Relative paths**: The path `../videos/poker-1.mp4` depends on where you run the script. If you start from a different folder, update the path or use an absolute path.

## Quick Troubleshooting

- **Video won’t open / empty frames:**
  - Check the path in `cv2.VideoCapture(...)` is correct
  - Try switching to the webcam (`cv2.VideoCapture(0)`)
- **Missing weights error:**
  - Put `yolov8m_synthetic.pt` in the correct location or fix the `YOLO("path/to/weights.pt")` string in your code.
- **CPU/GPU check:**
  - The script will print your `torch` version and show whether it’s running on CPU or GPU.