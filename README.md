```markdown
# Tesseract OCR Preprocessing Tuner

This tool helps you find optimal preprocessing parameters for images before feeding them to Tesseract OCR, specifically designed for common use cases like extracting text from screenshots in applications (e.g., betting interfaces). It iterates through various combinations of image scaling, contrast, sharpness, adaptive thresholding, and blurring, scoring the OCR results against a known target text.

## Features

-   **Automated Parameter Tuning:** Finds the best `scale_factor`, `contrast_enhance`, `sharpness_enhance`, `adaptive_method`, `adaptive_block_size`, `adaptive_C`, and `median_blur_kernel` for your specific image type.
-   **Score-Based Optimization:** Uses a scoring mechanism to evaluate OCR accuracy, especially for complex patterns like handicaps and coefficients.
-   **Easy to Use:** Command-line interface for quick execution.
-   **Standalone:** A single Python script with clear dependencies.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.x**:
    Download from [python.org](https://www.python.org/downloads/).
2.  **Tesseract OCR Engine**:
    This tool *requires* Tesseract to be installed on your system.
    -   **Windows**: Download the installer from [Tesseract at UB Mannheim](https://digi.bib.uni-mannheim.de/tesseract/). Remember to add Tesseract to your system's PATH during installation, or specify its path using the `--tesseract_path` argument.
    -   **macOS**: `brew install tesseract` (if you have Homebrew).
    -   **Linux (Debian/Ubuntu)**: `sudo apt install tesseract-ocr`
    -   For other systems, refer to the [Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kislyakov84/tesseract_tuner.git
    cd tesseract_tuner
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the tuner from your terminal, providing the path to your reference image and the exact text you expect to find within that image.

```bash
python tesseract_tuner.py <path_to_your_image.png> "<Your Target Text>" [options]
```

**Arguments:**

-   `<path_to_your_image.png>`: **Required**. Path to the image file (e.g., a screenshot) that contains the text you want to optimize OCR for.
-   `<Your Target Text>`: **Required**. The *exact* text string you expect to be found in the image. This text is used to score the OCR results. **Enclose in double quotes if it contains spaces.**
    -   **Important for sports betting use cases**: Include handicap values in parentheses (e.g., `(+1.0)`) and, if applicable, the numerical coefficient at the very end (e.g., `1.097`).
    -   **Examples**:
        -   `"Форы Победа с учетом форы Реймс (+1.0) 1.097"`
        -   `"Мальме (-0.5) 1.45"`
        -   `"Some general text 1.23"`

**Options:**

-   `--max_combinations <number>`: Maximum number of parameter combinations to try. Defaults to `5000`. Lower values for faster but less thorough search.
-   `--verbose`: Enable debug logging for more detailed output during the tuning process.
-   `--tesseract_path <path>`: Optional path to the Tesseract executable. Use this if Tesseract is not in your system's PATH.

### Examples:

1.  **Basic tuning:**
    ```bash
    python tesseract_tuner.py sample_image.png "Форы Победа с учетом форы Реймс (+1.0) 1.097"
    ```

2.  **With increased verbosity and limited combinations:**
    ```bash
    python tesseract_tuner.py my_screenshot.png "Team A - Team B, 1.5 goals total (over) 1.85" --max_combinations 1000 --verbose
    ```

3.  **Specifying Tesseract path (Windows example):**
    ```bash
    python tesseract_tuner.py C:\Users\User\Desktop\game_screenshot.png "Player Name (2.5) 2.10" --tesseract_path "C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```

## Output

The script will print the "Recommended Tesseract Preprocessing Parameters" to the console upon completion. These are the parameters that resulted in the highest OCR score for your target text on the provided image, along with the achieved score. You can then integrate these parameters into your main bot or application.

## How it Works (Briefly)

The tuner performs the following steps:
1.  **Image Preprocessing**: Applies various transformations (scaling, contrast, sharpness, adaptive thresholding, blurring) to the input image.
2.  **OCR Execution**: Runs Tesseract OCR on the preprocessed image.
3.  **Scoring**: Evaluates the OCR output against your `target_text`. The scoring function specifically looks for the presence of the handicap display (e.g., `(+1.0)`) and the coefficient value, giving higher scores for matches and high OCR confidence.
4.  **Optimization**: Iterates through a predefined range of preprocessing parameters, keeping track of the combination that yields the highest OCR score.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements or bug fixes.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
```