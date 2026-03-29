import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading images from same folder
image_a = cv2.imread("sample1.jpg")
image_b = cv2.imread("sample2.jpg")

# Checking for image errors
if image_a is None or image_b is None:
    print("Error: Image not found. Check file paths.")
    exit()

image_a = cv2.resize(image_a, (300, 300))
image_b = cv2.resize(image_b, (300, 300))


# 1. EDGE DETECTION

edges_for_image_a = cv2.Canny(image_a, 100, 200)
edges_for_image_b = cv2.Canny(image_b, 100, 200)


# 2. COLOR HISTOGRAM

histogram_for_image_a = cv2.calcHist([image_a], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
histogram_for_image_b = cv2.calcHist([image_b], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])

cv2.normalize(histogram_for_image_a, histogram_for_image_a)
cv2.normalize(histogram_for_image_b, histogram_for_image_b)


# 3. SIMILARITY (for Histogram)

similarity_score = cv2.compareHist(histogram_for_image_a, histogram_for_image_b, cv2.HISTCMP_CORREL)

print(f"Similarity Score is: {round(similarity_score*100,2)}%")


# 4. FEATURE DETECTION (using ORB)

orb = cv2.ORB_create(nfeatures=500)

kp1, des1 = orb.detectAndCompute(image_a, None)
kp2, des2 = orb.detectAndCompute(image_b, None)

image_a_kp = cv2.drawKeypoints(image_a, kp1, None, color=(0,255,0))
image_b_kp = cv2.drawKeypoints(image_b, kp2, None, color=(0,255,0))


# FEATURE MATCHING

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

# Keep only best matches
good_matches = matches[:10]

# Draw matches
match_images = cv2.drawMatches(image_a, kp1, image_b, kp2, good_matches, None, flags=2)

print("Number of Matches are:", len(matches))


match_score = len(matches) / max(len(kp1), len(kp2))
final_score = (similarity_score * 0.5) + (match_score * 0.5)
score_number= round(final_score*100,2)
print("Final Combined Score is:", score_number, "%")

if score_number < 30:
    print("Images are NOT Similar")
elif score_number >= 30 and score_number <= 65:
    print("Images are Moderately Similar")
elif score_number > 65 and score_number <= 99:
    print("Images are Highly Similar")
    
if score_number == 100:
    print("Both Images are the Same")

# Saving Output
cv2.imwrite("matches_output.jpg", match_images)

# DISPLAY RESULTS

plt.figure(figsize=(10,8))

plt.subplot(3,2,1)
plt.title("Image 1")
plt.imshow(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB))

plt.subplot(3,2,2)
plt.title("Image 2")
plt.imshow(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB))

plt.subplot(3,2,3)
plt.title("Edges 1")
plt.imshow(edges_for_image_a, cmap='gray')

plt.subplot(3,2,4)
plt.title("Edges 2")
plt.imshow(edges_for_image_b, cmap='gray')

plt.subplot(3,2,5)
plt.title("Keypoints 1")
plt.imshow(cv2.cvtColor(image_a_kp, cv2.COLOR_BGR2RGB))

plt.subplot(3,2,6)
plt.title("Keypoints 2")
plt.imshow(cv2.cvtColor(image_b_kp, cv2.COLOR_BGR2RGB))

plt.figure(figsize=(10,5))
plt.title("Feature Matches")
plt.imshow(cv2.cvtColor(match_images, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()