# OMR-PROJ

1. Problem Statement
Start by clearly and concisely stating the problem you are trying to solve.

Problem: Manual OMR sheet evaluation is slow and prone to human error.

Solution: An automated system to quickly and accurately score OMR sheets using computer vision.

2. Approach
Explain your technical strategy. Describe how you solved the problem without getting bogged down in code details.

Technology Stack: List the key libraries you used: Python, OpenCV, and NumPy.

Core Logic: Briefly outline the process:

Image Pre-processing: Preparing the image for analysis (e.g., converting to grayscale, removing noise).

Perspective Correction: Finding the OMR sheet and flattening it to a perfect rectangle.

Bubble Detection: Locating all the answer bubbles on the sheet.

Scoring: Counting filled bubbles and comparing them to an answer key to calculate the score.

3. Installation
Provide clear, simple instructions for anyone to get your project running.

Step 1: Clone the repository.

Bash

git clone https://github.com/your-username/your-repo-name.git
Step 2: Navigate to the project directory.

Bash

cd your-repo-name
Step 3: Install the required libraries.

Bash

pip install opencv-python numpy imutils
4. Usage
Tell the user how to use your program.

Step 1: Place your OMR sheet image (e.g., omr_sheet.jpg) in the project directory.

Step 2: Update the ANSWER_KEY in scanner.py to match your sheet's correct answers.

Step 3: Run the script from your terminal.

Bash

python scanner.py
Output: The program will print the final score in the terminal.
