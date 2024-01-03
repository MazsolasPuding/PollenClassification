from pathlib import Path
import shutil

def image_folder_separation(path: str):
    root_path = Path.cwd()
    database_path = Path(path)

    pictures = [item.name for item in database_path.iterdir()]  # Some pictures have spaces instead of underscores
    classes = set([pic.replace(' ', '_').split('_')[0] for pic in pictures])
    
    new_data_path = root_path / "KaggleDB_Structured"
    new_data_path.mkdir(parents=True, exist_ok=True)

    for _class in classes:
        new_sub = new_data_path / _class
        new_sub.mkdir(parents=True, exist_ok=True)
        pic_counter = 0

        for pic in pictures:
            if pic.replace(' ', '_').split('_')[0] == _class:
                shutil.copy(database_path / pic, (new_sub / str(pic_counter)).with_suffix(".jpg"))
                pic_counter += 1
                

if __name__ == "__main__":
    image_folder_separation(path="./KaggleDB")

