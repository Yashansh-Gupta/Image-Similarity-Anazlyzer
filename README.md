# Image Similarity Analyzer

A computer vision project that compares two images using **Color Histogram Analysis**, **Edge Detection (Canny)**, and **Feature Matching (ORB)** to compute a similarity score.

---

## Project Overview

This tool takes two images as input and performs three analyses:
1. **Color Histogram Comparison** – Measures color distribution similarity using correlation.
2. **Edge Detection** – Detects structural edges using the Canny algorithm.
3. **Feature Detection & Matching** – Identifies and matches keypoints using the ORB (Oriented FAST and Rotated BRIEF) detector.

A **combined similarity score** is computed by weighting histogram similarity (50%) and feature match ratio (50%), and the result is classified as:
- **Not Similar** – Score < 30%
- **Moderately Similar** – 30% ≤ Score ≤ 65%
- **Highly Similar** – 65% < Score < 100%
- **Identical** – Score = 100%

---

## Requirements

- Python 3.8 or higher
- pip

### Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Image processing (edge detection, histogram, ORB) |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization and result display |

---

## Setup Instructions

### Step 1: Clone or Download the Project

```bash
git clone <your-repo-url>
cd <project-folder>
```

Or simply place `main.py`, `sample1.jpg`, and `sample2.jpg` in the same directory.

### Step 2: (Optional) Create a Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install opencv-python numpy matplotlib
```

---

## Running the Project

### Step 1: Prepare Your Images

Place the two images you want to compare in the **same directory** as `main.py` and name them:
- `sample1.jpg`
- `sample2.jpg`

> You can rename any `.jpg` images to `sample1.jpg` and `sample2.jpg`, or modify the filenames directly in `main.py`.

### Step 2: Run the Script

```bash
python main.py
```

### Step 3: View the Output

The script will:
- **Print** the similarity score and classification to the terminal.
- **Display** two matplotlib windows:
  - **Figure 1**: Side-by-side view of both images, their detected edges, and keypoints.
  - **Figure 2**: Feature match visualization showing matched keypoints between the two images.
- **Save** the feature match image as `matches_output.jpg` in the current directory.

#### Sample Terminal Output

```
Similarity Score is: 89.52%
Number of Matches are: 17
Final Combined Score is: 50.24 %
Images are Moderately Similar
```

---

## Project Structure

```
project/
│
├── main.py              # Main script
├── sample1.jpg          # Input Image 1
├── sample2.jpg          # Input Image 2
├── matches_output.jpg   # Output: Feature match visualization (auto-generated)
└── README.md            # This file
```

---

## Configuration

To use different images, open `main.py` and update lines 5–6:

```python
image_a = cv2.imread("your_image1.jpg")
image_b = cv2.imread("your_image2.jpg")
```

To change the number of ORB keypoints (default: 500):

```python
orb = cv2.ORB_create(nfeatures=500)  # Increase for more features
```

To change how many top matches are displayed:

```python
good_matches = matches[:10]  # Change 10 to desired number
```

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `Error: Image not found` | Ensure `sample1.jpg` and `sample2.jpg` are in the same directory as `main.py` |
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python` |
| Display window not appearing | Ensure you have a GUI environment; on headless servers, comment out `plt.show()` |
| Low match count | Images may be genuinely different, or try adjusting `nfeatures` in ORB |

---

## Notes

- The script resizes both images to **300×300 pixels** before analysis to ensure fair comparison.
- The color histogram uses an **8×8×8 bin** 3D histogram over BGR channels.
- ORB matching uses **Brute-Force Matcher** with Hamming distance.
- The output image `matches_output.jpg` is always saved regardless of similarity result.
