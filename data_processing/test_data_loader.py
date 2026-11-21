from data_quality import analyze_data_quality
from data_loader import load_dataset


if __name__ == "__main__":
    print("Testing Data Loader...")

    dataset_path = input("Enter the CSV file path: ").strip().strip('"')

    try:
        df = load_dataset(dataset_path)
        analyze_data_quality(df)

        print("\n--- DATASET PREVIEW ---")
        print(df.head())

        print("\n--- DATASET INFORMATION ---")
        print(df.info())

        print("\n--- MISSING VALUES ---")
        print(df.isnull().sum())

        print("\n--- DUPLICATE ROWS ---")
        print(df.duplicated().sum())

        print("\nDATA LOADER TEST SUCCESSFUL!")

    except Exception as e:
        print("\nDATA LOADER TEST FAILED!")
        print("Error:", e)