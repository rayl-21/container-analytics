"""
OCR Module for Container Number Recognition

This module provides OCR capabilities for extracting container numbers from
detected container regions using both pytesseract and EasyOCR engines.

Features:
- Dual OCR engine support (Tesseract and EasyOCR)
- Container number format validation
- Image preprocessing for improved OCR accuracy
- Confidence scoring and result ranking
- Batch processing capabilities
"""

import logging
import re
import time
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import easyocr
import supervision as sv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContainerOCR:
    """
    OCR engine for extracting container numbers from image regions.
    
    Supports multiple OCR backends and includes preprocessing and
    validation specifically designed for container number recognition.
    """
    
    # Container number patterns (ISO 6346 standard)
    CONTAINER_NUMBER_PATTERNS = [
        r'^[A-Z]{4}[0-9]{6}[0-9]$',  # Standard: 4 letters + 7 digits
        r'^[A-Z]{3}[UJZ][0-9]{6}[0-9]$',  # With equipment category
        r'^[A-Z]{4}\s?[0-9]{6}\s?[0-9]$',  # With optional spaces
    ]
    
    # Common container owner codes (first 3 letters)
    COMMON_OWNER_CODES = {
        'APL', 'MSC', 'CMA', 'CGM', 'EVE', 'HAP', 'HMM', 'MOL', 'NYK', 'ONE',
        'PIL', 'YML', 'ZIM', 'COS', 'EMC', 'WHL', 'TRIU', 'GESU', 'TCLU', 'FCIU'
    }
    
    def __init__(
        self,
        use_easyocr: bool = True,
        use_tesseract: bool = True,
        easyocr_gpu: bool = False,
        tesseract_config: str = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    ):
        """
        Initialize the OCR engine.
        
        Args:
            use_easyocr: Whether to use EasyOCR engine
            use_tesseract: Whether to use Tesseract engine
            easyocr_gpu: Whether to use GPU for EasyOCR
            tesseract_config: Tesseract configuration string
        """
        self.use_easyocr = use_easyocr
        self.use_tesseract = use_tesseract
        self.tesseract_config = tesseract_config
        
        # Initialize OCR engines
        self.easyocr_reader = None
        if self.use_easyocr:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=easyocr_gpu)
                logger.info(f"Initialized EasyOCR (GPU: {easyocr_gpu})")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.use_easyocr = False
        
        if self.use_tesseract:
            try:
                # Test Tesseract installation
                pytesseract.get_tesseract_version()
                logger.info("Initialized Tesseract OCR")
            except Exception as e:
                logger.warning(f"Failed to initialize Tesseract: {e}")
                self.use_tesseract = False
        
        if not (self.use_easyocr or self.use_tesseract):
            raise RuntimeError("No OCR engines available. Please install pytesseract or easyocr.")
        
        # Performance tracking
        self.ocr_times = []
        
    def extract_container_numbers(
        self,
        image: Union[np.ndarray, str, Path],
        detections: Optional[sv.Detections] = None,
        min_confidence: float = 0.5
    ) -> List[Dict]:
        """
        Extract container numbers from an image or specific detection regions.
        
        Args:
            image: Input image (numpy array or path)
            detections: Optional detections to focus OCR on specific regions
            min_confidence: Minimum confidence threshold for results
            
        Returns:
            List of dictionaries containing OCR results
        """
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image: {image}")
        
        results = []
        
        if detections is not None and len(detections) > 0:
            # Process specific detection regions
            for i, bbox in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                region = image[y1:y2, x1:x2]
                
                if region.size == 0:
                    continue
                
                # Extract text from region
                region_results = self._extract_text_from_region(region, min_confidence)
                
                # Add region information
                for result in region_results:
                    result.update({
                        "detection_index": i,
                        "bbox": (x1, y1, x2, y2),
                        "region_size": region.shape[:2]
                    })
                
                results.extend(region_results)
        else:
            # Process entire image
            results = self._extract_text_from_region(image, min_confidence)
        
        # Filter and validate container numbers
        container_results = []
        for result in results:
            if self._is_valid_container_number(result["text"]):
                result["is_container_number"] = True
                result["formatted_number"] = self._format_container_number(result["text"])
                container_results.append(result)
        
        # Record processing time
        processing_time = time.time() - start_time
        self.ocr_times.append(processing_time)
        
        logger.info(f"OCR completed in {processing_time:.3f}s. "
                   f"Found {len(container_results)} container numbers from {len(results)} text regions")
        
        return container_results
    
    def _extract_text_from_region(
        self,
        region: np.ndarray,
        min_confidence: float
    ) -> List[Dict]:
        """
        Extract text from a specific image region using available OCR engines.
        
        Args:
            region: Image region as numpy array
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of OCR results
        """
        results = []
        
        # Preprocess image for better OCR
        processed_region = self._preprocess_image(region)
        
        # Try EasyOCR
        if self.use_easyocr and self.easyocr_reader is not None:
            try:
                easyocr_results = self.easyocr_reader.readtext(processed_region)
                
                for (bbox, text, confidence) in easyocr_results:
                    if confidence >= min_confidence:
                        results.append({
                            "text": text.upper().strip(),
                            "confidence": float(confidence),
                            "engine": "easyocr",
                            "bbox_relative": bbox
                        })
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        
        # Try Tesseract
        if self.use_tesseract:
            try:
                # Convert to PIL Image for Tesseract
                pil_image = Image.fromarray(processed_region)
                
                # Get detailed results from Tesseract
                data = pytesseract.image_to_data(
                    pil_image,
                    config=self.tesseract_config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Process Tesseract results
                for i, text in enumerate(data['text']):
                    confidence = float(data['conf'][i]) / 100.0  # Convert to 0-1 scale
                    
                    if confidence >= min_confidence and text.strip():
                        # Calculate bounding box
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        
                        results.append({
                            "text": text.upper().strip(),
                            "confidence": confidence,
                            "engine": "tesseract",
                            "bbox_relative": [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                        })
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize image if too small (OCR works better on larger text)
        height, width = gray.shape
        if height < 40 or width < 200:
            scale_factor = max(40 / height, 200 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def _is_valid_container_number(self, text: str) -> bool:
        """
        Validate if text matches container number format.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text appears to be a valid container number
        """
        # Clean text
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Check against patterns
        for pattern in self.CONTAINER_NUMBER_PATTERNS:
            if re.match(pattern, cleaned_text):
                return True
        
        # Additional checks for partial matches or common variations
        if len(cleaned_text) >= 10:
            # Check if starts with known owner code
            if cleaned_text[:3] in self.COMMON_OWNER_CODES:
                return True
            
            # Check basic structure (letters followed by digits)
            if re.match(r'^[A-Z]{3,4}[0-9]{6,7}$', cleaned_text):
                return True
        
        return False
    
    def _format_container_number(self, text: str) -> str:
        """
        Format container number to standard format.
        
        Args:
            text: Raw container number text
            
        Returns:
            Formatted container number
        """
        # Remove non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Standard format: ABCD1234567
        if len(cleaned) == 11 and cleaned[:4].isalpha() and cleaned[4:].isdigit():
            return cleaned
        
        # Try to fix common issues
        if len(cleaned) >= 10:
            # Extract letters and digits
            letters = ''.join(c for c in cleaned if c.isalpha())[:4]
            digits = ''.join(c for c in cleaned if c.isdigit())
            
            if len(letters) >= 3 and len(digits) >= 6:
                # Pad letters to 4 characters if needed
                if len(letters) == 3:
                    letters += 'U'  # Common equipment category identifier
                
                # Take first 7 digits
                digits = digits[:7]
                
                return letters + digits
        
        return cleaned
    
    def extract_from_detections_batch(
        self,
        images: List[np.ndarray],
        detections_list: List[sv.Detections],
        min_confidence: float = 0.5
    ) -> List[List[Dict]]:
        """
        Process multiple images and their detections in batch.
        
        Args:
            images: List of input images
            detections_list: List of detection results for each image
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of OCR results for each image
        """
        logger.info(f"Starting batch OCR processing for {len(images)} images")
        start_time = time.time()
        
        all_results = []
        
        for i, (image, detections) in enumerate(zip(images, detections_list)):
            try:
                results = self.extract_container_numbers(image, detections, min_confidence)
                all_results.append(results)
                
                logger.debug(f"Processed image {i+1}/{len(images)}: "
                           f"{len(results)} container numbers found")
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                all_results.append([])
        
        total_time = time.time() - start_time
        total_containers = sum(len(results) for results in all_results)
        
        logger.info(f"Batch OCR completed in {total_time:.2f}s. "
                   f"Total container numbers found: {total_containers}")
        
        return all_results
    
    def get_performance_stats(self) -> Dict:
        """
        Get OCR performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.ocr_times:
            return {"message": "No OCR operations performed yet"}
        
        times = np.array(self.ocr_times)
        
        return {
            "total_operations": len(times),
            "mean_time": float(np.mean(times)),
            "median_time": float(np.median(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "std_time": float(np.std(times)),
            "engines_used": {
                "easyocr": self.use_easyocr,
                "tesseract": self.use_tesseract
            }
        }


# Utility functions
def validate_container_check_digit(container_number: str) -> bool:
    """
    Validate container number check digit according to ISO 6346.
    
    Args:
        container_number: Container number to validate
        
    Returns:
        True if check digit is valid
    """
    if len(container_number) != 11:
        return False
    
    # ISO 6346 check digit calculation
    owner_equipment = container_number[:10]
    check_digit = int(container_number[10])
    
    # Character to number mapping
    char_values = {
        'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17,
        'H': 18, 'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25,
        'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32,
        'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
    }
    
    # Calculate sum
    total = 0
    for i, char in enumerate(owner_equipment):
        if char.isalpha():
            value = char_values.get(char, 0)
        else:
            value = int(char)
        
        total += value * (2 ** i)
    
    # Calculate check digit
    calculated_check = total % 11
    if calculated_check == 10:
        calculated_check = 0
    
    return calculated_check == check_digit


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Container OCR Test")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--directory", type=str, help="Path to directory with images")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum confidence")
    parser.add_argument("--tesseract-only", action="store_true", help="Use only Tesseract")
    parser.add_argument("--easyocr-only", action="store_true", help="Use only EasyOCR")
    
    args = parser.parse_args()
    
    # Initialize OCR
    use_easyocr = not args.tesseract_only
    use_tesseract = not args.easyocr_only
    
    ocr = ContainerOCR(use_easyocr=use_easyocr, use_tesseract=use_tesseract)
    
    if args.image:
        # Process single image
        results = ocr.extract_container_numbers(args.image, min_confidence=args.confidence)
        
        print(f"Found {len(results)} container numbers:")
        for result in results:
            print(f"  {result['formatted_number']} (confidence: {result['confidence']:.2f}, "
                  f"engine: {result['engine']})")
    
    elif args.directory:
        # Process directory
        image_dir = Path(args.directory)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        all_results = []
        for image_path in image_paths:
            results = ocr.extract_container_numbers(image_path, min_confidence=args.confidence)
            all_results.extend(results)
            print(f"{image_path.name}: {len(results)} container numbers")
        
        print(f"\nTotal: {len(all_results)} container numbers found")
        
        # Save results if output specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
    
    else:
        parser.print_help()
    
    # Print performance stats
    print("\nPerformance:", ocr.get_performance_stats())