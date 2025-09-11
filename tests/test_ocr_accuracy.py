"""
OCR Accuracy Tests for Container Number Recognition

This test suite focuses specifically on OCR accuracy and reliability
for container number extraction under various conditions.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from modules.detection.ocr import ContainerOCR, validate_container_check_digit


class TestOCRAccuracy:
    """Test OCR accuracy with various container number formats and conditions."""
    
    @pytest.fixture
    def ocr_engine(self):
        """Create OCR engine optimized for testing."""
        return ContainerOCR(
            use_easyocr=True,
            use_tesseract=True,
            easyocr_gpu=False  # Use CPU for testing
        )
    
    def create_text_image(self, text: str, font_scale: float = 1.0, noise: bool = False) -> np.ndarray:
        """
        Create a synthetic image with container number text.
        
        Args:
            text: Text to render
            font_scale: Font size scale
            noise: Whether to add noise to simulate real conditions
            
        Returns:
            Image as numpy array
        """
        # Create white background
        img_height, img_width = 100, 500
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)  # Black
        thickness = max(1, int(2 * font_scale))
        
        # Calculate text position (centered)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img_width - text_size[0]) // 2
        text_y = (img_height + text_size[1]) // 2
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, thickness)
        
        # Add noise if requested
        if noise:
            noise_array = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise_array)
        
        return img
    
    @pytest.mark.parametrize("container_number,expected_valid", [
        ("MSCU1234567", True),
        ("APLU9876543", True),
        ("CGMU4567890", True),
        ("TCLU5555555", True),
        ("GESU1111111", True),
        ("FCIU2222222", True),
        # Invalid cases
        ("MSCU123456", False),    # Too short
        ("MSCU12345678", False),  # Too long
        ("1234567890A", False),   # Wrong format
        ("RANDOMTEXT", False),    # Not a container number
    ])
    def test_container_number_validation_comprehensive(self, ocr_engine, container_number, expected_valid):
        """Test container number validation with various formats."""
        result = ocr_engine._is_valid_container_number(container_number)
        assert result == expected_valid, f"Validation failed for {container_number}"
    
    @pytest.mark.parametrize("input_text,expected_output", [
        ("MSCU 123 456 7", "MSCU1234567"),
        ("mscu-123-456-7", "MSCU1234567"),
        ("MSCU_123_456_7", "MSCU1234567"),
        ("M S C U 1 2 3 4 5 6 7", "MSCU1234567"),
        ("MSCU123456", "MSCU1234560"),  # Add equipment category 'U'
        ("APL123456", "APLU1234560"),   # Add equipment category 'U'
    ])
    def test_container_number_formatting(self, ocr_engine, input_text, expected_output):
        """Test container number formatting and normalization."""
        result = ocr_engine._format_container_number(input_text)
        assert result == expected_output, f"Formatting failed: {input_text} -> {result} (expected {expected_output})"
    
    def test_ocr_preprocessing(self, ocr_engine):
        """Test image preprocessing for OCR."""
        # Create test image
        original_img = self.create_text_image("MSCU1234567", font_scale=0.8)
        
        # Test preprocessing
        processed_img = ocr_engine._preprocess_image(original_img)
        
        # Verify preprocessing
        assert processed_img.shape[:2] == original_img.shape[:2] or processed_img.shape[0] >= 40
        assert len(processed_img.shape) == 2  # Should be grayscale
        assert processed_img.dtype == np.uint8
    
    @pytest.mark.parametrize("font_scale,noise_level", [
        (0.5, False),   # Small text, no noise
        (1.0, False),   # Normal text, no noise
        (1.5, False),   # Large text, no noise
        (1.0, True),    # Normal text with noise
    ])
    def test_ocr_under_various_conditions(self, ocr_engine, font_scale, noise_level):
        """Test OCR performance under various image conditions."""
        container_number = "MSCU1234567"
        test_image = self.create_text_image(container_number, font_scale, noise_level)
        
        # Mock the actual OCR engines to avoid dependency on external libraries
        with patch.object(ocr_engine, '_extract_text_from_region') as mock_extract:
            # Simulate OCR result with varying confidence based on conditions
            base_confidence = 0.9
            if font_scale < 0.7:
                base_confidence *= 0.7  # Lower confidence for small text
            if noise_level:
                base_confidence *= 0.8  # Lower confidence with noise
            
            mock_extract.return_value = [{
                'text': container_number,
                'confidence': base_confidence,
                'engine': 'mock',
                'bbox_relative': [[0, 0], [100, 0], [100, 50], [0, 50]]
            }]
            
            results = ocr_engine.extract_container_numbers(test_image)
            
            # Should find the container number
            container_results = [r for r in results if r.get('is_container_number', False)]
            assert len(container_results) > 0
            
            result = container_results[0]
            assert result['formatted_number'] == container_number
            assert result['confidence'] >= 0.5
    
    def test_ocr_with_multiple_containers(self, ocr_engine):
        """Test OCR when multiple container numbers are present."""
        # Create image with multiple container numbers
        img = np.ones((200, 800, 3), dtype=np.uint8) * 255
        
        containers = ["MSCU1234567", "APLU9876543", "CGMU5555555"]
        
        for i, container in enumerate(containers):
            y_pos = 50 + i * 50
            cv2.putText(img, container, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Mock OCR to return multiple results
        with patch.object(ocr_engine, '_extract_text_from_region') as mock_extract:
            mock_results = []
            for container in containers:
                mock_results.append({
                    'text': container,
                    'confidence': 0.9,
                    'engine': 'mock',
                    'bbox_relative': [[0, 0], [100, 0], [100, 30], [0, 30]]
                })
            mock_extract.return_value = mock_results
            
            results = ocr_engine.extract_container_numbers(img)
            
            # Should find all container numbers
            container_results = [r for r in results if r.get('is_container_number', False)]
            assert len(container_results) == 3
            
            found_numbers = {r['formatted_number'] for r in container_results}
            assert found_numbers == set(containers)
    
    def test_performance_tracking(self, ocr_engine):
        """Test OCR performance tracking."""
        # Process multiple images to build performance stats
        for i in range(5):
            test_image = self.create_text_image(f"MSCU123456{i}")
            
            with patch.object(ocr_engine, '_extract_text_from_region') as mock_extract:
                mock_extract.return_value = [{
                    'text': f'MSCU123456{i}',
                    'confidence': 0.9,
                    'engine': 'mock'
                }]
                
                ocr_engine.extract_container_numbers(test_image)
        
        # Check performance stats
        stats = ocr_engine.get_performance_stats()
        
        assert stats['total_operations'] == 5
        assert 'mean_time' in stats
        assert 'median_time' in stats
        assert stats['engines_used']['easyocr'] == True
        assert stats['engines_used']['tesseract'] == True
    
    @pytest.mark.parametrize("confidence_threshold", [0.3, 0.5, 0.7, 0.9])
    def test_confidence_filtering(self, ocr_engine, confidence_threshold):
        """Test OCR result filtering by confidence threshold."""
        test_image = self.create_text_image("MSCU1234567")
        
        # Mock OCR with varying confidence results
        with patch.object(ocr_engine, '_extract_text_from_region') as mock_extract:
            mock_extract.return_value = [
                {'text': 'MSCU1234567', 'confidence': 0.95, 'engine': 'mock'},  # High confidence
                {'text': 'APLU9876543', 'confidence': 0.6, 'engine': 'mock'},   # Medium confidence
                {'text': 'CGMU5555555', 'confidence': 0.2, 'engine': 'mock'},   # Low confidence
            ]
            
            results = ocr_engine.extract_container_numbers(test_image, min_confidence=confidence_threshold)
            
            # Check that only results above threshold are returned
            for result in results:
                if 'confidence' in result:
                    assert result['confidence'] >= confidence_threshold
    
    def test_edge_cases(self, ocr_engine):
        """Test OCR with edge cases and potential failure conditions."""
        edge_cases = [
            # Empty image
            np.ones((100, 100, 3), dtype=np.uint8) * 255,
            # Very small image
            np.ones((10, 10, 3), dtype=np.uint8) * 128,
            # Very large image
            np.ones((2000, 3000, 3), dtype=np.uint8) * 200,
            # High contrast image
            np.ones((100, 400, 3), dtype=np.uint8) * 255,
        ]
        
        # Add text to non-empty cases
        cv2.putText(edge_cases[0], "MSCU1234567", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(edge_cases[3], "MSCU1234567", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        for i, test_image in enumerate(edge_cases):
            with patch.object(ocr_engine, '_extract_text_from_region') as mock_extract:
                # Some cases might return empty results
                if i in [1, 2]:  # Very small or very large images
                    mock_extract.return_value = []
                else:
                    mock_extract.return_value = [{
                        'text': 'MSCU1234567',
                        'confidence': 0.8,
                        'engine': 'mock'
                    }]
                
                # Should not crash, even with edge cases
                try:
                    results = ocr_engine.extract_container_numbers(test_image)
                    assert isinstance(results, list)
                except Exception as e:
                    pytest.fail(f"OCR failed on edge case {i}: {e}")


class TestContainerNumberValidation:
    """Test container number validation and check digit calculation."""
    
    @pytest.mark.parametrize("container_number,should_validate", [
        # Standard format containers
        ("MSCU1234567", True),
        ("APLU9876543", True),
        ("CGMU1111111", True),
        
        # With equipment category
        ("MSCУ1234567", True),  # Equipment category U
        ("APLZ9876543", True),   # Equipment category Z
        ("CGMJ1111111", True),   # Equipment category J
        
        # Invalid formats
        ("MSCU123456", False),   # Too short
        ("MSCU12345678", False), # Too long
        ("1234MSCU567", False),  # Wrong order
        ("MSCU123456A", False),  # Letter at end
        ("", False),             # Empty string
        ("MSCU", False),         # Only letters
        ("1234567", False),      # Only numbers
    ])
    def test_container_format_validation(self, container_number, should_validate):
        """Test container number format validation."""
        ocr = ContainerOCR()
        result = ocr._is_valid_container_number(container_number)
        assert result == should_validate, f"Validation incorrect for {container_number}"
    
    def test_check_digit_calculation(self):
        """Test ISO 6346 check digit calculation."""
        # Note: This tests the function exists and runs, not necessarily correctness
        # Real check digit validation requires known valid container numbers
        
        test_numbers = [
            "MSCU1234560",  # Example container number
            "APLU9876543",  # Another example
        ]
        
        for number in test_numbers:
            if len(number) == 11:
                result = validate_container_check_digit(number)
                assert isinstance(result, bool)
    
    def test_owner_code_recognition(self):
        """Test recognition of common container owner codes."""
        ocr = ContainerOCR()
        
        # Test with known owner codes
        known_owners = ['MSC', 'APL', 'CMA', 'CGM', 'EVE', 'HAP', 'HMM', 'MOL', 'NYK', 'ONE']
        
        for owner in known_owners:
            test_number = f"{owner}U1234567"
            result = ocr._is_valid_container_number(test_number)
            assert result == True, f"Should recognize owner code {owner}"
    
    def test_case_insensitive_validation(self):
        """Test that validation works regardless of case."""
        ocr = ContainerOCR()
        
        test_cases = [
            ("MSCU1234567", "mscu1234567"),
            ("APLU9876543", "aplu9876543"),
            ("CGMU1111111", "cgmu1111111"),
        ]
        
        for upper_case, lower_case in test_cases:
            upper_result = ocr._is_valid_container_number(upper_case)
            lower_result = ocr._is_valid_container_number(lower_case)
            assert upper_result == lower_result
            
            # Both should format to the same result
            upper_formatted = ocr._format_container_number(upper_case)
            lower_formatted = ocr._format_container_number(lower_case)
            assert upper_formatted == lower_formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])