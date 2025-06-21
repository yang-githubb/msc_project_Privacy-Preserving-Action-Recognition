import torch

def main():
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA is NOT available.")

if __name__ == "__main__":
    main()
