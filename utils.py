def clean_up_files(files):
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error removing {file}: {str(e)}")
