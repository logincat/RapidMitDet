import os
import cv2
from tqdm import tqdm
from torchvahadane import TorchVahadaneNormalizer

if __name__ == '__main__':
    image_dir = "./dataset/MIDOG21/tiff/"
    output_dir = "./dataset/MIDOG21/tiff_sa/"
    os.makedirs(output_dir, exist_ok=True)
    domain_dir = "./dataset/MIDOG21/domain/"
    for domain_file in os.listdir(domain_dir):
        domain_name = domain_file.split(".")[0]
        im_path = os.path.join(domain_dir, domain_file)
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        normalizer = TorchVahadaneNormalizer()
        normalizer.fit(im)
        for input_file in tqdm(os.listdir(image_dir)):
            image_id = int(input_file[:3])
            img_name = input_file.split(".")[0]
            file_path = os.path.join(image_dir, input_file)
            file_name = img_name + "_" + domain_name
            output_path = output_dir + file_name + ".tiff"
            if os.path.exists(output_path):
                continue
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transformed_img = normalizer.transform(img).cpu().numpy()
            transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, transformed_img)
