"""
Download Alzheimer's Dataset from Kaggle
Requires kaggle.json API credentials
"""

import os
import zipfile
import shutil

def download_kaggle_dataset():
    """Download and extract Alzheimer's dataset from Kaggle"""
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle package not installed")
        print("Installing kaggle...")
        os.system("pip install kaggle")
        import kaggle
    
    print("=" * 60)
    print("Downloading Alzheimer's Dataset from Kaggle")
    print("=" * 60)
    
    # Common Alzheimer's datasets on Kaggle
    datasets = [
        "tourist55/alzheimers-dataset-4-class-of-images",
        "uraninjo/augmented-alzheimer-mri-dataset"
    ]
    
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")
    
    choice = input(f"\nSelect dataset (1-{len(datasets)}) or enter custom dataset name: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(datasets):
        dataset_name = datasets[int(choice) - 1]
    else:
        dataset_name = choice
    
    print(f"\n📥 Downloading: {dataset_name}")
    
    # Create dataset directory
    os.makedirs('./dataset', exist_ok=True)
    
    try:
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path='./dataset',
            unzip=True
        )
        print("✅ Dataset downloaded successfully!")
        
        # List downloaded contents
        print("\n📂 Downloaded contents:")
        for item in os.listdir('./dataset'):
            path = os.path.join('./dataset', item)
            if os.path.isdir(path):
                count = len(os.listdir(path))
                print(f"  📁 {item}/ ({count} items)")
            else:
                print(f"  📄 {item}")
        
        # Check for expected structure
        expected_path = './dataset/AugmentedAlzheimerDataset'
        if os.path.exists(expected_path):
            print(f"\n✅ Found AugmentedAlzheimerDataset!")
            classes = os.listdir(expected_path)
            print(f"Classes: {classes}")
            for cls in classes:
                cls_path = os.path.join(expected_path, cls)
                if os.path.isdir(cls_path):
                    img_count = len(os.listdir(cls_path))
                    print(f"  - {cls}: {img_count} images")
        else:
            print(f"\n⚠️ AugmentedAlzheimerDataset folder not found")
            print("You may need to reorganize the dataset structure")
            print("\nExpected structure:")
            print("dataset/")
            print("  AugmentedAlzheimerDataset/")
            print("    MildDemented/")
            print("    ModerateDemented/")
            print("    NonDemented/")
            print("    VeryMildDemented/")
        
        print("\n" + "=" * 60)
        print("✅ Dataset setup complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your kaggle.json is in the right location")
        print("2. Verify the dataset name is correct")
        print("3. Make sure you've accepted the dataset terms on Kaggle")

if __name__ == '__main__':
    download_kaggle_dataset()
