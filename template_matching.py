# Imports
import cv2
import  numpy as np

# Read the original image and template
messi = cv2.imread("messi5.jpg")
template = cv2.imread("template.jpg")

# Convert to grayscale
gray_messi = cv2.cvtColor(messi, cv2.COLOR_BGR2GRAY)

# shape of template
w, h, _ = template.shape[::-1]

# Template matching
res = cv2.matchTemplate(messi, template, cv2.TM_CCORR_NORMED)

# Set Threshold
threshold = 0.99

# Location of pixel value close to threshold
loc = np.where(res >= threshold)

# Draw rectangle
for pt in zip(*loc[::-1]):
    cv2.rectangle(messi, pt, (pt[0] + h, pt[1] + _), (0, 0, 255), 2)

cv2.imshow("image", messi)
cv2.imwrite("template_matched.jpg", messi)
cv2.waitKey(0)
cv2.destroyAllWindows()
