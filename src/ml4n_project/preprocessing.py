from logging import error
import pandas as pd
class Preprocesser:
    def __init__(self, path:str = "../../res/ddos_dataset.csv"):
        self.path: str = path
        self.dataset: pd.DataFrame = pd.read_csv(self.path)
        try:
            self.dataset: pd.DataFrame = pd.read_csv(self.path)
        except:
            error(f"Unable to find dataset {path.split("/")[-1]} at {"".join(path.split("/")[0:-1])}")

        
    def visualize(self):
        print(self.dataset.head(5))

if __name__ == "__main__":
    preprocess = Preprocesser()
    preprocess.visualize()
