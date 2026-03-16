"""
Script test preprocessing: load Preprocessor.pkl và transform 1 mẫu hợp lệ.
Chạy từ thư mục gốc project: python test_preprocessing.py
"""
import os
import sys
import pandas as pd

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.Airbnb.utils.utils import load_object


def get_sample_row():
    """1 mẫu hợp lệ: tất cả giá trị nằm trong categories đã train."""
    return pd.DataFrame([{
        # Numerical (đúng thứ tự như num_pipeline)
        'amenities': 10,
        'accommodates': 4,
        'bathrooms': 2,
        'latitude': 40.7128,
        'longitude': -74.0060,
        'host_response_rate': 98,
        'number_of_reviews': 50,
        'review_scores_rating': 95,
        'bedrooms': 2,
        'beds': 3,
        # Categorical (đúng thứ tự như cat_pipeline, giá trị phải có trong OrdinalEncoder)
        'property_type': 'Apartment',
        'room_type': 'Entire home/apt',
        'bed_type': 'Real Bed',
        'cancellation_policy': 'flexible',
        'cleaning_fee': 'True',
        'city': 'NYC',
        'host_identity_verified': 't',
        'instant_bookable': 'f',
        'host_has_profile_pic': 't',
    }])


if __name__ == "__main__":
    preprocessor_path = os.path.join("Artifacts", "Preprocessor.pkl")
    if not os.path.exists(preprocessor_path):
        print(f"Không tìm thấy {preprocessor_path}. Chạy Training_pipeline trước.")
        sys.exit(1)

    preprocessor = load_object(preprocessor_path)
    X = get_sample_row()

    print("Input (1 row, 19 features):")
    print(X.T.to_string())
    print()

    scaled = preprocessor.transform(X)
    print("Output after preprocessing (shape):", scaled.shape)
    print("Scaled array:\n", scaled)
