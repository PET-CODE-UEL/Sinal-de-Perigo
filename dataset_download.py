import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "ahmedkhanak1995/sign-language-gesture-images-dataset"
)

print("Path to dataset files:", path)
