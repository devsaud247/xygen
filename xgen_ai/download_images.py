import urllib
import requests
import os
import pandas as pd


class UrlToImage:
    def __init__(self) -> None:
        os.makedirs(f"{os.getcwd()}/images", exist_ok=True)
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Chrome/51.0.2704.103")]
        urllib.request.install_opener(opener)
        self.read_links_file("images_url.csv")

    def read_links_file(self, path) -> None:
        df = pd.read_csv(path)
        images_links = df.get("image", [])
        self.url_to_image(images_links)

    def url_to_image(self, imgs) -> None:

        count=1

        for img in imgs:
            image_name = f"data_{count}.jpg"
            count += 1
            file_exists = os.path.exists(f"images/{image_name}")
            if not file_exists:
                try:
                    urllib.request.urlretrieve(img, f"images/{image_name}")
                    print(image_name, file_exists)
                except:
                    pass
            


if __name__ == "__main__":
    UrlToImage()
