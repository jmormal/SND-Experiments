from classes_pydantic import Edge, Hub, Mode, Network, Plant, Product, RTI, Zone


class Model:
    def __init__(self, path: str) -> None:
        self.N = Network.load_json(path)
        pass


if __name__ == "__main__":
    N = Model("./fsd.json/SND_RTI_small_1.json")
