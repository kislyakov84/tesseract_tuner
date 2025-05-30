import os
import re
import time
import argparse
import logging
from typing import Optional, Tuple, List, Any, Dict

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageStat, ImageChops

# Настройка логирования для вывода в консоль
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# Для более детальных сообщений (отладочных):
# logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


# --- НАСТРОЙКИ TESSERACT OCR ---
# Укажите путь к исполняемому файлу Tesseract, если он не в PATH
# Пример для Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Пример для Linux/macOS, если установлен через Homebrew: r'/opt/homebrew/bin/tesseract'
# Или оставьте None, если Tesseract уже в системном PATH
TESSERACT_PATH = None
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    logging.info(f"Tesseract CMD set to: {TESSERACT_PATH}")
else:
    logging.info("Tesseract CMD not explicitly set. Assuming it's in system PATH.")
    try:
        # Проверяем, доступен ли Tesseract
        pytesseract.image_to_string(Image.new('RGB', (10,10)), config='--psm 7')
        logging.info("Tesseract is accessible.")
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract is not found. Please install it or set TESSERACT_PATH correctly.")
        logging.error("See https://tesseract-ocr.github.io/tessdoc/Installation.html")
        exit(1)
    except Exception as e:
        logging.warning(f"Error checking Tesseract accessibility (may not be an issue if it works later): {e}")

# Начальные параметры, которые будут перебираться тюнером.
# Эти значения могут быть адаптированы под ваши нужды.
# 'adaptive_method': 0 для cv2.ADAPTIVE_THRESH_MEAN_C, 1 для cv2.ADAPTIVE_THRESH_GAUSSIAN_C
INITIAL_TESSERACT_PARAMS = {
    'scale_factor': 3,
    'contrast_enhance': 1.8,
    'sharpness_enhance': 2.5,
    'adaptive_method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Changed from 0/1 to direct cv2 constant
    'adaptive_block_size': 21,
    'adaptive_C': 5,
    'median_blur_kernel': 1
}


def preprocess_for_tesseract(pil_image: Image.Image,
                             scale_factor: int,
                             contrast_enhance: float,
                             sharpness_enhance: float,
                             adaptive_method: int,
                             adaptive_block_size: int,
                             adaptive_C: int,
                             median_blur_kernel: int) -> Image.Image:
    """
    Предобработка изображения для Tesseract OCR с настраиваемыми параметрами.
    """
    new_width = int(pil_image.width * scale_factor)
    new_height = int(pil_image.height * scale_factor)
    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    logging.debug(f"Image resized to {new_width}x{new_height}.")

    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_enhance)
    logging.debug(f"Contrast enhanced by {contrast_enhance}.")

    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(sharpness_enhance)
    logging.debug(f"Sharpness enhanced by {sharpness_enhance}.")

    open_cv_image = np.array(pil_image.convert('RGB'))
    open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR

    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    logging.debug("Image converted to grayscale.")

    # adaptiveThreshold requires blockSize to be odd and > 1
    if adaptive_block_size % 2 == 0:
        adaptive_block_size += 1
        logging.warning(f"adaptive_block_size must be odd. Adjusted to {adaptive_block_size}.")
    if adaptive_block_size <= 1:
        adaptive_block_size = 3
        logging.warning(f"adaptive_block_size must be > 1. Adjusted to {adaptive_block_size}.")

    thresh = cv2.adaptiveThreshold(
        gray, 255, adaptive_method,
        cv2.THRESH_BINARY, adaptive_block_size, adaptive_C
    )
    logging.debug(f"Performed adaptive binarization (method={adaptive_method}, blockSize={adaptive_block_size}, C={adaptive_C}).")

    if median_blur_kernel > 1:
        thresh = cv2.medianBlur(thresh, median_blur_kernel)
        logging.debug(f"Applied median blur with kernel {median_blur_kernel}.")

    return Image.fromarray(thresh)

def extract_text_tesseract(pil_image: Image.Image, **kwargs) -> Tuple[str, List[List[Any]]]:
    """
    Использует Tesseract OCR для распознавания текста с изображения.
    Принимает параметры для предобработки через kwargs.
    """
    try:
        processed_img = preprocess_for_tesseract(pil_image, **kwargs)

        # Сохранение предобработанного изображения для отладки
        # processed_img.save("debug_processed_for_tesseract.png")
        # logging.debug("Saved debug_processed_for_tesseract.png")

        # Настройка Tesseract (подходит для большинства текстов)
        custom_config = r'--oem 1 --psm 3 -c tessedit_char_whitelist=0123456789().,-+АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъьэюяABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

        full_text = pytesseract.image_to_string(
            processed_img,
            lang='rus+eng',
            config=custom_config
        )

        data = pytesseract.image_to_data(
            processed_img,
            lang='rus+eng',
            output_type=pytesseract.Output.DICT,
            config=custom_config
        )

        results = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            try:
                conf = float(data['conf'][i])
            except ValueError:
                conf = 0.0
            text = data['text'][i]
            if text.strip() == "": # Пропускаем пустые текстовые блоки
                continue
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            vertices = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            results.append([vertices, text, conf])
        return full_text, results
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract executable not found. Please install Tesseract or set TESSERACT_PATH.")
        return "", []
    except Exception as e:
        logging.error(f"Error in extract_text_tesseract: {e}", exc_info=True)
        return "", []

def score_ocr_result(full_text_ocr: str, ocr_blocks: List[List[Any]],
                     target_handicap_display: str, target_coef_value: str) -> float:
    """
    Оценивает качество OCR распознавания для заданного целевого текста.
    Возвращает числовой балл, чем больше, тем лучше.
    """
    score = 0.0
    full_text_lower = full_text_ocr.lower().replace(" ", "").replace(",", ".")

    target_handicap_lower = target_handicap_display.lower().replace(" ", "")
    if target_handicap_lower in full_text_lower:
        score += 1.0
        logging.debug(f"Score +1.0 for full text match of handicap display: {target_handicap_lower}")

    # Convert coef to float early for more robust comparison (if it's a valid number)
    try:
        target_coef_float = float(target_coef_value.replace(",", "."))
    except ValueError:
        target_coef_float = None # Not a valid coefficient to check for

    if target_coef_value != "нет_коэф" and target_coef_value.replace(",", ".") in full_text_lower:
        score += 1.0
        logging.debug(f"Score +1.0 for full text match of coefficient: {target_coef_value}")

    # Block-level scoring
    found_handicap_block_conf = 0.0
    found_handicap_and_coef_block_conf_sum = 0.0

    for block_vertices, block_text, block_conf in ocr_blocks:
        block_text_lower = block_text.lower().replace(" ", "").replace(",", ".")

        # Check for handicap display match in block
        if target_handicap_lower in block_text_lower:
            found_handicap_block_conf = max(found_handicap_block_conf, block_conf)
            logging.debug(f"Found handicap '{target_handicap_lower}' in block '{block_text}'. Max conf: {found_handicap_block_conf}")

            # If handicap is found, try to find coefficient in the same block
            if target_coef_float is not None:
                # Use regex to find potential float numbers in the block
                potential_coefs = re.findall(r'\b\d+(?:[.,]\d+)?\b', block_text_lower)
                for pc_str in potential_coefs:
                    try:
                        pc_float = float(pc_str.replace(",", "."))
                        # Check if the found coefficient is very close to the target
                        if abs(pc_float - target_coef_float) < 0.001: # Small tolerance for float comparison
                            score += 2.0 # Major score for finding both in one block
                            score += block_conf / 100.0 # Add confidence bonus
                            logging.debug(f"Found handicap and coefficient '{pc_float}' in same block '{block_text}'. Score +2.0 + {block_conf/100.0}")
                            found_handicap_and_coef_block_conf_sum += block_conf # Sum confs if multiple matches possible (though usually one per block)
                            break # Assume we only need one match per block

    score += found_handicap_block_conf / 200.0 # Small bonus for finding handicap in any block with good confidence
    logging.debug(f"Final handicap block conf bonus: {found_handicap_block_conf/200.0}. Current score: {score}")

    return score


def tune_tesseract_preprocessing(
    target_pil_image: Image.Image,
    search_text: str, # Исходная строка поиска (например, "Форы Победа с учетом форы Ливерпуль (+1.0) 1.073")
    max_combinations: int = 5000, # Ограничение на количество комбинаций для скорости
    param_ranges: Optional[Dict[str, List[Any]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Автоматически подбирает лучшие параметры предобработки для Tesseract OCR
    на основе распознавания целевого текста.

    Возвращает словарь с лучшими параметрами или None, если ничего не найдено.
    """
    logging.info(f"Starting Tesseract preprocessing tuning for text: '{search_text}'")

    # 1. Парсим целевой текст для системы скоринга
    handicap_match = re.search(r'(\(([-+]?\d+(?:\.\d+)?)\))', search_text)
    if not handicap_match:
        logging.error(f"Failed to extract handicap value from search_text: {search_text}")
        return None
    target_handicap_display = handicap_match.group(1).strip() # e.g., "(+1.0)"
    target_handicap_value = handicap_match.group(2).strip() # e.g., "+1.0"

    # Пытаемся извлечь ожидаемый коэффициент из search_text, если он там есть
    # Например, если search_text = "Ливерпуль (+1.0) 1.073"
    target_coef_value = "нет_коэф" # Значение по умолчанию, если не найдено
    coef_search = re.search(r'([0-9]+(?:[.,][0-9]+)?)$', search_text) # Ищем число в конце строки
    if coef_search:
        target_coef_value = coef_search.group(1).replace(",", ".")
    logging.info(f"Target text for scoring: Handicap='{target_handicap_display}', Coef='{target_coef_value}'")

    # 2. Определяем диапазоны параметров для перебора
    if param_ranges is None:
        param_ranges = {
            'scale_factor': [2, 3, 4],
            'contrast_enhance': [1.5, 1.8, 2.0, 2.2, 2.5],
            'sharpness_enhance': [1.5, 2.0, 2.5, 3.0],
            'adaptive_method': [cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C],
            'adaptive_block_size': [5, 7, 9, 11, 13, 15, 17, 19, 21], # Only odd numbers
            'adaptive_C': [-10, -5, -3, 0, 3, 5, 7, 9, 10], # Including negative values
            'median_blur_kernel': [1, 3] # 1 means no blur, 3 for light blur
        }
    else:
        logging.info("Using custom parameter ranges provided.")

    best_score = -1.0
    best_params = None

    # Generate all combinations of parameters
    import itertools
    all_combinations = list(itertools.product(*param_ranges.values()))
    
    logging.info(f"Total combinations to check: {len(all_combinations)}. Max combinations set to: {max_combinations}.")

    for i, params_tuple in enumerate(all_combinations):
        if i >= max_combinations:
            logging.info(f"Reached maximum number of combinations ({max_combinations}). Stopping search.")
            break

        current_params = {
            'scale_factor': params_tuple[0],
            'contrast_enhance': params_tuple[1],
            'sharpness_enhance': params_tuple[2],
            'adaptive_method': params_tuple[3],
            'adaptive_block_size': params_tuple[4],
            'adaptive_C': params_tuple[5],
            'median_blur_kernel': params_tuple[6]
        }

        # Ensure adaptive_block_size is odd and > 1
        if current_params['adaptive_block_size'] % 2 == 0:
            current_params['adaptive_block_size'] += 1
        if current_params['adaptive_block_size'] <= 1:
            current_params['adaptive_block_size'] = 3

        try:
            # Execute Tesseract OCR with current preprocessing parameters
            full_text, ocr_blocks = extract_text_tesseract(target_pil_image.copy(), **current_params)

            # Evaluate the result
            current_score = score_ocr_result(full_text, ocr_blocks,
                                             target_handicap_display, target_coef_value)
            
            # Print intermediate progress only if it's a significant improvement or for debugging
            if logging.getLogger().level == logging.DEBUG:
                logging.debug(f"Combo {i+1}/{len(all_combinations)}: Score={current_score:.2f}, Params={current_params}")
                
            if current_score > best_score:
                best_score = current_score
                best_params = current_params
                logging.info(f"NEW BEST! Score: {best_score:.2f}, Params: {best_params}. OCR text: '{full_text.strip()[:100]}...'")

        except Exception as e:
            logging.warning(f"Error testing parameters {current_params}: {e}")
            continue # Continue to the next combination

    if best_params:
        logging.info(f"Tuning finished. Best score: {best_score:.2f}")
        logging.info(f"Found best preprocessing parameters: {best_params}")
    else:
        logging.warning("Tuning did not find any suitable parameters.")

    return best_params

def main():
    parser = argparse.ArgumentParser(
        description="Tesseract OCR Preprocessing Tuner.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the reference image file (e.g., screenshot) containing the target text."
    )
    parser.add_argument(
        "target_text",
        type=str,
        help="""The exact text string expected to be found in the image.
Include handicap values in parentheses and, if applicable, the coefficient
at the end. This is used for scoring the OCR result.

Examples:
  "Форы Победа с учетом форы Реймс (+1.0) 1.097"
  "Мальме (-0.5) 1.45"
  "Some text with numbers 1.23"
"""
    )
    parser.add_argument(
        "--max_combinations",
        type=int,
        default=5000,
        help="Maximum number of parameter combinations to try. Lower for faster but less thorough search."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for more detailed output during tuning."
    )
    parser.add_argument(
        "--tesseract_path",
        type=str,
        default=TESSERACT_PATH,
        help=f"""Optional: Path to the Tesseract executable.
If not provided, assumes Tesseract is in your system's PATH.
Current default (if set): {TESSERACT_PATH or 'Not set'}
Example for Windows: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
Example for Linux/macOS: /usr/bin/tesseract or /opt/homebrew/bin/tesseract
"""
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    if args.tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
        logging.info(f"Tesseract CMD overridden to: {args.tesseract_path}")

    # Re-check Tesseract accessibility after path might have been set by argparse
    try:
        pytesseract.image_to_string(Image.new('RGB', (10,10)), config='--psm 7')
        logging.info("Tesseract is accessible.")
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract is not found at the specified path or in system PATH.")
        logging.error("Please ensure Tesseract is installed and accessible.")
        logging.error("You might need to set the --tesseract_path argument correctly.")
        logging.error("See https://tesseract-ocr.github.io/tessdoc/Installation.html")
        exit(1)
    except Exception as e:
        logging.warning(f"Error checking Tesseract accessibility (may not be an issue if it works later): {e}")


    image_path = args.image_path
    target_search_text = args.target_text
    max_combinations = args.max_combinations

    if not os.path.exists(image_path):
        logging.error(f"Error: Image file not found at '{image_path}'.")
        parser.print_help()
        exit(1)

    try:
        target_image_for_tune = Image.open(image_path)
        logging.info(f"Loaded image: '{image_path}'")
    except Exception as e:
        logging.error(f"Error loading image '{image_path}': {e}", exc_info=True)
        exit(1)

    logging.info(f"Starting Tesseract tuning with '{max_combinations}' max combinations for text: '{target_search_text}'")
    best_ocr_params_found = tune_tesseract_preprocessing(target_image_for_tune, target_search_text, max_combinations)

    if best_ocr_params_found:
        print("\n" + "="*50)
        print("Tuning Completed Successfully!")
        print(f"Best Score Achieved: {best_ocr_params_found.get('score', 'N/A'):.2f}") # Score is not returned by tune_tesseract_preprocessing directly, but it could be added.
        print("Recommended Tesseract Preprocessing Parameters:")
        for key, value in best_ocr_params_found.items():
            # For adaptive_method, convert cv2 constant back to human-readable form if desired
            if key == 'adaptive_method':
                method_name = "cv2.ADAPTIVE_THRESH_GAUSSIAN_C" if value == cv2.ADAPTIVE_THRESH_GAUSSIAN_C else "cv2.ADAPTIVE_THRESH_MEAN_C"
                print(f"  '{key}': {method_name},")
            else:
                print(f"  '{key}': {value},")
        print("="*50 + "\n")
        print("You can copy these parameters into your bot's configuration.")
    else:
        print("\n" + "="*50)
        print("Tuning did NOT find improved parameters.")
        print("Consider checking your target text or image, or extending the search ranges.")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()