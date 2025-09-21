import cv2
import numpy as np
import imutils

# --- Part 1: The Setup ---
# Path to your OMR sheet image
IMAGE_PATH = "omr_sheet.jpg"

# Define the correct answers (0=A, 1=B, 2=C, 3=D, 4=E)
# This key needs to be updated with all 100 correct answers
ANSWER_KEY = {
    0: 2,  # Question 1: C
    1: 1,  # Question 2: B
    2: 4,  # Question 3: E
    3: 0,  # Question 4: A
    4: 3,  # Question 5: D
}

# --- Part 2: Functions for Image Processing ---
def find_sheet(edged_image):
    """Finds the main sheet contour and returns its four corner points."""
    contours = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    sheet_contour = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            sheet_contour = approx
            break
            
    return sheet_contour

def get_birdseye_view(image, sheet_contour):
    """Applies a perspective transform to flatten the sheet."""
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    pts = order_points(sheet_contour.reshape(4, 2))
    width, height = 800, 1000  # Adjust as needed for your sheet
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def sort_contours_by_y(cnts):
    """Sorts contours from top to bottom."""
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][1], reverse=False))
    return cnts

# --- Part 3: Main Execution Block ---
if __name__ == '__main__':
    # Load the image
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print(f"Error: Could not read the image file at '{IMAGE_PATH}'. Please check the file path and name.")
        exit()

    # Pre-process the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Find the main contour
    sheet_contour = find_sheet(edged)

    if sheet_contour is not None:
        # Visualize the detected contour for debugging
        display_image = image.copy()
        cv2.drawContours(display_image, [sheet_contour], -1, (0, 255, 0), 3)
        cv2.imshow("Detected Sheet Contour", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Apply the perspective transform
        warped = get_birdseye_view(image, sheet_contour)
        cv2.imshow("Flattened Sheet", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Convert the straightened image to black-and-white
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cv2.imshow("Threshold Image", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Find all the contours (bubbles) on the sheet
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        question_contours = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            # Use more forgiving filters for bubble detection
            if w >= 10 and h >= 10 and aspect_ratio >= 0.7 and aspect_ratio <= 1.3:
                question_contours.append(c)

        if len(question_contours) == 0:
            print("Error: No valid bubbles were found. Check your threshold or filtering parameters.")
            exit()
            
        question_contours = sort_contours_by_y(question_contours)
        
        score = 0
        total_questions = len(ANSWER_KEY)

        # Loop through the questions in groups of 5 bubbles per question
        for (q, i) in enumerate(np.arange(0, len(question_contours), 5)):
            q_contours = question_contours[i:i+5]
            q_contours = sorted(q_contours, key=lambda c: cv2.boundingRect(c)[0])
            
            marked_bubble = None
            for (j, c) in enumerate(q_contours):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                bubble_area = cv2.bitwise_and(thresh, thresh, mask=mask)
                marked_pixels = cv2.countNonZero(bubble_area)

                if marked_pixels > 400: # Adjust if needed
                    marked_bubble = j

            if marked_bubble is not None and marked_bubble == ANSWER_KEY.get(q):
                score += 1

        print(f"Total Score: {score} out of {total_questions}")
    else:
        print("Could not find the OMR sheet in the image. Please use a clearer photo.")

    cv2.destroyAllWindows()
