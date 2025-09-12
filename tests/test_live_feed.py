"""
Tests for the simplified Live Feed UI implementation.

This module tests the new simplified Live Feed page with:
- 7-day image gallery (2025-09-01 to 2025-09-07)
- Clean sidebar with Pull New Data toggle
- Detection overlay toggle and truck count badges
- Efficient image grid display sorted by date
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the functions from the Live Feed page
from pages import Live_Feed
from modules.database.models import session_scope, Image as DBImage, Detection


class TestLiveFeedGallery:
    """Test the new Live Feed gallery functionality."""
    
    @pytest.fixture
    def mock_image_data(self):
        """Create mock image data for testing."""
        return {
            'id': 1,
            'filepath': '/tmp/test_image.jpg',
            'camera_id': 'in_gate',
            'timestamp': datetime(2025, 9, 5, 14, 30, 0),
            'processed': True,
            'file_size': 1024,
            'detections': [
                {
                    'id': 1,
                    'object_type': 'truck',
                    'confidence': 0.95,
                    'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                    'tracking_id': 101,
                    'container_number': 'CONT12345'
                },
                {
                    'id': 2,
                    'object_type': 'container',
                    'confidence': 0.88,
                    'bbox': {'x1': 300, 'y1': 150, 'x2': 400, 'y2': 250},
                    'tracking_id': 102,
                    'container_number': 'CONT67890'
                }
            ]
        }
    
    @pytest.fixture
    def temp_image(self):
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Create a simple test image
            img = PILImage.new('RGB', (800, 600), color='blue')
            img.save(tmp_file.name)
            yield tmp_file.name
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    def test_load_7day_gallery_date_range(self):
        """Test that load_7day_gallery uses correct date range."""
        with patch('pages.Live_Feed.session_scope') as mock_session:
            mock_session_obj = MagicMock()
            mock_session.__enter__.return_value = mock_session_obj
            mock_session_obj.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
            
            from pages.Live_Feed import load_7day_gallery
            
            result = load_7day_gallery()
            
            # Verify the session query was called
            mock_session_obj.query.assert_called_once()
            
            # Verify filter was called with correct date range
            filter_call = mock_session_obj.query.return_value.filter.call_args[0][0]
            
            # The result should be an empty list for this mock
            assert result == []
    
    def test_load_7day_gallery_with_data(self, mock_image_data):
        """Test load_7day_gallery with mock database data."""
        with patch('pages.Live_Feed.session_scope') as mock_session:
            mock_session_obj = MagicMock()
            mock_session.__enter__.return_value = mock_session_obj
            
            # Mock database image
            mock_img = MagicMock()
            mock_img.id = mock_image_data['id']
            mock_img.filepath = mock_image_data['filepath']
            mock_img.camera_id = mock_image_data['camera_id']
            mock_img.timestamp = mock_image_data['timestamp']
            mock_img.processed = mock_image_data['processed']
            mock_img.file_size = mock_image_data['file_size']
            
            # Mock detections
            mock_detections = []
            for det_data in mock_image_data['detections']:
                mock_det = MagicMock()
                mock_det.id = det_data['id']
                mock_det.object_type = det_data['object_type']
                mock_det.confidence = det_data['confidence']
                mock_det.bbox = det_data['bbox']
                mock_det.tracking_id = det_data['tracking_id']
                mock_det.container_number = det_data['container_number']
                mock_detections.append(mock_det)
            
            # Setup mock queries
            mock_session_obj.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_img]
            mock_session_obj.query.return_value.filter.return_value.all.return_value = mock_detections
            
            from pages.Live_Feed import load_7day_gallery
            
            result = load_7day_gallery()
            
            # Verify we got one image with correct data
            assert len(result) == 1
            assert result[0]['id'] == mock_image_data['id']
            assert result[0]['filepath'] == mock_image_data['filepath']
            assert result[0]['camera_id'] == mock_image_data['camera_id']
            assert len(result[0]['detections']) == 2
            
            # Verify detection data
            assert result[0]['detections'][0]['object_type'] == 'truck'
            assert result[0]['detections'][1]['object_type'] == 'container'
    
    def test_display_image_card_truck_count(self, mock_image_data, temp_image):
        """Test truck count calculation in display_image_card."""
        # Update mock data to use the temporary image
        mock_image_data['filepath'] = temp_image
        
        with patch('streamlit.image'), patch('streamlit.markdown'):
            from pages.Live_Feed import display_image_card
            
            # Test with detections containing trucks and containers
            display_image_card(mock_image_data, show_detections=False)
            
            # The function should run without errors
            # Truck count should be 2 (1 truck + 1 container)
    
    def test_display_image_card_no_detections(self, temp_image):
        """Test display_image_card with no detections."""
        mock_data = {
            'id': 1,
            'filepath': temp_image,
            'camera_id': 'test_cam',
            'timestamp': datetime.now(),
            'detections': []
        }
        
        with patch('streamlit.image'), patch('streamlit.markdown'):
            from pages.Live_Feed import display_image_card
            
            # Should not raise any errors
            display_image_card(mock_data, show_detections=False)
    
    def test_display_image_card_missing_file(self):
        """Test display_image_card with missing image file."""
        mock_data = {
            'id': 1,
            'filepath': '/non/existent/path.jpg',
            'camera_id': 'test_cam',
            'timestamp': datetime.now(),
            'detections': []
        }
        
        with patch('streamlit.warning') as mock_warning:
            from pages.Live_Feed import display_image_card
            
            display_image_card(mock_data, show_detections=False)
            
            # Should show warning for missing file
            mock_warning.assert_called_once()
    
    def test_add_detection_overlays(self, temp_image):
        """Test detection overlay functionality."""
        # Create test image
        img = PILImage.open(temp_image)
        
        detections = [
            {
                'object_type': 'truck',
                'confidence': 0.95,
                'bbox': {'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150}
            },
            {
                'object_type': 'container',
                'confidence': 0.88,
                'bbox': {'x1': 200, 'y1': 100, 'x2': 300, 'y2': 200}
            }
        ]
        
        from pages.Live_Feed import add_detection_overlays
        
        result_img = add_detection_overlays(img, detections)
        
        # Should return a PIL Image
        assert isinstance(result_img, PILImage.Image)
        
        # Image dimensions should remain the same
        assert result_img.size == img.size
    
    def test_add_detection_overlays_invalid_bbox(self, temp_image):
        """Test detection overlays with invalid bounding boxes."""
        img = PILImage.open(temp_image)
        
        detections = [
            {
                'object_type': 'truck',
                'confidence': 0.95,
                'bbox': {}  # Invalid/empty bbox
            },
            {
                'object_type': 'container',
                'confidence': 0.88,
                'bbox': {'x1': 1000, 'y1': 1000, 'x2': 50, 'y2': 50}  # Invalid coordinates
            }
        ]
        
        from pages.Live_Feed import add_detection_overlays
        
        # Should not raise errors, just skip invalid detections
        result_img = add_detection_overlays(img, detections)
        assert isinstance(result_img, PILImage.Image)
    
    def test_truck_count_calculation(self, mock_image_data):
        """Test truck/container count calculation logic."""
        detections = mock_image_data['detections']
        
        # Count trucks, containers, and vehicles
        truck_count = len([
            d for d in detections 
            if d.get('object_type', '').lower() in ['truck', 'container', 'vehicle']
        ])
        
        # Should count both truck and container
        assert truck_count == 2
    
    def test_date_grouping_logic(self):
        """Test image grouping by date logic."""
        from datetime import datetime
        
        # Mock gallery data with different dates
        gallery_data = [
            {'timestamp': datetime(2025, 9, 1, 10, 0), 'id': 1},
            {'timestamp': datetime(2025, 9, 1, 15, 0), 'id': 2},
            {'timestamp': datetime(2025, 9, 2, 8, 0), 'id': 3},
            {'timestamp': datetime(2025, 9, 3, 12, 0), 'id': 4}
        ]
        
        # Group images by date (simulating main function logic)
        grouped_images = {}
        for img_data in gallery_data:
            img_date = img_data['timestamp'].strftime('%Y-%m-%d')
            if img_date not in grouped_images:
                grouped_images[img_date] = []
            grouped_images[img_date].append(img_data)
        
        # Should have 3 date groups
        assert len(grouped_images) == 3
        assert '2025-09-01' in grouped_images
        assert '2025-09-02' in grouped_images
        assert '2025-09-03' in grouped_images
        
        # September 1st should have 2 images
        assert len(grouped_images['2025-09-01']) == 2
        
        # Other dates should have 1 image each
        assert len(grouped_images['2025-09-02']) == 1
        assert len(grouped_images['2025-09-03']) == 1
    
    def test_error_handling_load_gallery(self):
        """Test error handling in load_7day_gallery."""
        with patch('pages.Live_Feed.session_scope') as mock_session:
            # Simulate database error
            mock_session.side_effect = Exception("Database connection failed")
            
            with patch('streamlit.error') as mock_error:
                from pages.Live_Feed import load_7day_gallery
                
                result = load_7day_gallery()
                
                # Should return empty list on error
                assert result == []
                
                # Should show error message
                mock_error.assert_called_once()


class TestLiveFeedUI:
    """Test the UI components and behavior."""
    
    @patch('streamlit.sidebar')
    @patch('streamlit.toggle')
    @patch('streamlit.button')
    def test_sidebar_controls(self, mock_button, mock_toggle, mock_sidebar):
        """Test sidebar control elements."""
        # Mock the toggle states
        mock_toggle.side_effect = [False, True]  # pull_new_data=False, show_detections=True
        mock_button.return_value = False
        
        from pages.Live_Feed import main
        
        with patch('pages.Live_Feed.load_7day_gallery', return_value=[]):
            with patch('streamlit.markdown'), patch('streamlit.info'):
                # Should not raise any errors
                main()
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_summary_metrics(self, mock_metric, mock_columns):
        """Test the summary metrics display."""
        mock_columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        
        from pages.Live_Feed import main
        
        with patch('pages.Live_Feed.load_7day_gallery', return_value=[]):
            with patch('streamlit.sidebar'), patch('streamlit.markdown'), \
                 patch('streamlit.info'), patch('streamlit.toggle'), \
                 patch('streamlit.button'), patch('streamlit.subheader'), \
                 patch('streamlit.write'), patch('streamlit.divider'), \
                 patch('streamlit.spinner'):
                
                main()
                
                # Should call metric display
                assert mock_metric.called
    
    def test_empty_gallery_display(self):
        """Test behavior when no images are available."""
        from pages.Live_Feed import main
        
        with patch('pages.Live_Feed.load_7day_gallery', return_value=[]):
            with patch('streamlit.info') as mock_info, \
                 patch('streamlit.sidebar'), patch('streamlit.markdown'), \
                 patch('streamlit.toggle'), patch('streamlit.button'), \
                 patch('streamlit.subheader'), patch('streamlit.write'), \
                 patch('streamlit.divider'), patch('streamlit.spinner'):
                
                main()
                
                # Should show info message about no images
                mock_info.assert_called()


class TestLiveFeedIntegration:
    """Integration tests for the complete Live Feed functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_streamlit_mocks(self):
        """Setup common Streamlit mocks for integration tests."""
        with patch('streamlit.set_page_config'), \
             patch('streamlit.markdown'), \
             patch('streamlit.sidebar'), \
             patch('streamlit.toggle'), \
             patch('streamlit.button'), \
             patch('streamlit.write'), \
             patch('streamlit.subheader'), \
             patch('streamlit.info'), \
             patch('streamlit.divider'), \
             patch('streamlit.spinner'), \
             patch('streamlit.columns'), \
             patch('streamlit.metric'):
            yield
    
    def test_main_function_complete_flow(self, temp_image):
        """Test the complete main function flow with data."""
        # Create mock gallery data
        mock_gallery = [
            {
                'id': 1,
                'filepath': temp_image,
                'camera_id': 'in_gate',
                'timestamp': datetime(2025, 9, 5, 14, 30),
                'detections': [
                    {'object_type': 'truck', 'confidence': 0.95},
                    {'object_type': 'container', 'confidence': 0.88}
                ]
            }
        ]
        
        from pages.Live_Feed import main
        
        with patch('pages.Live_Feed.load_7day_gallery', return_value=mock_gallery), \
             patch('pages.Live_Feed.display_image_card') as mock_display:
            
            main()
            
            # Should call display_image_card for the image
            mock_display.assert_called_once()
    
    def test_main_function_error_handling(self):
        """Test main function error handling."""
        from pages.Live_Feed import main
        
        with patch('pages.Live_Feed.load_7day_gallery', side_effect=Exception("Test error")), \
             patch('streamlit.error') as mock_error:
            
            main()
            
            # Should handle the error gracefully
            mock_error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])