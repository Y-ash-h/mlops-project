import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../mlops-pipeline")))

from src.utils.data_router import detect_data_type

class TestDagLogic(unittest.TestCase):
    
    @patch("src.utils.data_router.Path")
    def test_detect_data_type_tabular_multiple_csv(self, mock_path):
        # Mock a directory with 2 CSVs (current logic requires > 1 CSV for tabular)
        mock_dir = MagicMock()
        mock_path.return_value = mock_dir
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = True
        
        # Mock meta.json missing
        mock_dir.__truediv__.return_value.exists.return_value = False
        
        # Mock file globbing
        file1 = MagicMock()
        file1.is_file.return_value = True
        file1.suffix = ".csv"
        file2 = MagicMock()
        file2.is_file.return_value = True
        file2.suffix = ".csv"
        
        mock_dir.rglob.return_value = [file1, file2]
        
        dtype = detect_data_type("/tmp/fake_data")
        self.assertEqual(dtype, "tabular")

    @patch("src.utils.data_router.Path")
    def test_detect_data_type_single_csv_is_text(self, mock_path):
        # Current logic: 1 CSV -> text
        mock_dir = MagicMock()
        mock_path.return_value = mock_dir
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = True
        
        mock_dir.__truediv__.return_value.exists.return_value = False
        
        file1 = MagicMock()
        file1.is_file.return_value = True
        file1.suffix = ".csv"
        
        mock_dir.rglob.return_value = [file1]
        
        dtype = detect_data_type("/tmp/fake_data")
        self.assertEqual(dtype, "text")

    @patch("src.utils.data_router.Path")
    def test_detect_data_type_text(self, mock_path):
        # Mock a directory with TXT
        mock_dir = MagicMock()
        mock_path.return_value = mock_dir
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = True
        
        # Mock meta.json missing
        mock_dir.__truediv__.return_value.exists.return_value = False
        
        # Mock file globbing
        file1 = MagicMock()
        file1.is_file.return_value = True
        file1.suffix = ".txt"
        
        mock_dir.rglob.return_value = [file1]
        
        dtype = detect_data_type("/tmp/fake_data")
        self.assertEqual(dtype, "text")

if __name__ == "__main__":
    unittest.main()
