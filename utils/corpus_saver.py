import glob
import os


def GET_PROJECT_ROOT():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(current_dir, "requirements.txt")):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not locate project root (requirements.txt not found)")
        current_dir = parent_dir
    return current_dir


PROJECT_ROOT = GET_PROJECT_ROOT()


class corpus_saver:
    def __init__(
        self,
        clean_data_path={
            "bbox": "dict/context_encoded/bboxes_encoded/",
            "class": "dict/context_encoded/classes_encoded/",
            "color": "dict/context_encoded/colors_encoded/",
            "tag": "dict/context_encoded/tags_encoded/",
        },
        save_corpus_path="dict/tag/tag_list.txt",
    ):
        self.clean_data_path = clean_data_path
        self.save_corpus_path = save_corpus_path

    def save_corpus(self, data_type="tag"):
        clean_data_paths = os.path.join(PROJECT_ROOT, self.clean_data_path[data_type])

        context = []
        # Lặp qua các thư mục con (L21_a, L22_a, L22_b, ...)
        sub_dirs = glob.glob(os.path.join(clean_data_paths, "*"))
        sub_dirs = [d for d in sub_dirs if os.path.isdir(d)]
        sub_dirs.sort()

        for sub_dir in sub_dirs:
            # Lặp qua các file txt trong mỗi thư mục con (V001.txt, V002.txt, ...)
            txt_files = glob.glob(os.path.join(sub_dir, "*.txt"))
            txt_files.sort()

            for path in txt_files:
                with open(path, "r", encoding="utf-8") as f:
                    data = f.readlines()
                    data = [item.strip().split() for item in data]
                    context += data

        corpus = []
        for data in context:
            corpus += data

        # Sắp xếp theo thứ tự alphabet
        unique_words = sorted(set(corpus))
        tag_corpus = "\n".join([word.replace("_", " ") for word in unique_words]) + "\n"

        with open(os.path.join(PROJECT_ROOT, self.save_corpus_path), "w") as f:
            f.write(tag_corpus)


if __name__ == "__main__":
    saver = corpus_saver()
    saver.save_corpus("tag")
    print("Corpus saved successfully")
