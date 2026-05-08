from __future__ import annotations
from numpy import full
from pydantic import BaseModel, field_serializer, field_validator, computed_field
import json


# ============================================================
# BASE ENTITIES
# ============================================================


class Product(BaseModel, frozen=True):
    name: str
    category: str

    def __hash__(self):
        return hash((self.name, self.category))


class Mode(BaseModel, frozen=True):
    id: int
    max_vol: float
    min_vol: float
    min_q: float
    max_q: float

    def __hash__(self):
        return hash(self.id)


class RTI(BaseModel):
    id: int
    name: str = ""
    type: str = ""
    average_useful_life: float
    current_stock: int
    product_capacity: dict[Product, float] = {}
    return_inlay_fraction: dict[Product, float] = {}
    length: float
    width: float
    full_height: float
    folded_height: float

    @computed_field
    @property
    def volume_full(self) -> float:
        return self.length * self.width * self.full_height

    @computed_field
    @property
    def volume_folded(self) -> float:
        return self.length * self.width * self.folded_height

    @field_serializer("product_capacity", "return_inlay_fraction")
    @classmethod
    def ser_product_dict(cls, v: dict[Product, float]) -> dict[str, float]:
        return {p.name: val for p, val in v.items()}


class Plant(BaseModel):
    id: int
    x: float
    y: float
    is_large: bool


class Hub(BaseModel):
    id: int
    x: float
    y: float
    zone_id: int
    fixed_cost: float
    variable_cost: float


class Zone(BaseModel):
    id: int
    x: float
    y: float
    plants: list[Plant] = []
    hubs: list[Hub] = []


class Edge(BaseModel):
    id: int
    source: Plant | Hub
    target: Plant | Hub
    zone_id: int
    demand: dict[Product, float] = {}
    allowed_empties: list[RTI] = []
    allowed_modes: list[Mode] = []
    fixed_economic_cost_per_volume: dict[Mode, float] = {}
    fixed_co2_cost_per_volume: dict[Mode, float] = {}
    variable_economic_cost_per_volume: dict[Mode, float] = {}
    variable_co2_cost_per_volume: dict[Mode, float] = {}

    @field_serializer("demand")
    @classmethod
    def ser_demand(cls, v: dict[Product, float]) -> dict[str, float]:
        return {p.name: val for p, val in v.items()}

    @field_serializer(
        "fixed_economic_cost_per_volume",
        "fixed_co2_cost_per_volume",
        "variable_economic_cost_per_volume",
        "variable_co2_cost_per_volume",
    )
    @classmethod
    def ser_mode_dict(cls, v: dict[Mode, float]) -> dict[str, float]:
        return {str(m.id): val for m, val in v.items()}

    @field_serializer("source", "target")
    @classmethod
    def ser_node(cls, v: Plant | Hub) -> dict:
        return {"type": type(v).__name__, **v.model_dump()}

    @field_serializer("allowed_empties")
    @classmethod
    def ser_empties(cls, v: list[RTI]) -> list[int]:
        return [r.id for r in v]


# ============================================================
# NETWORK
# ============================================================


class Network(BaseModel):
    products: dict[str, Product] = {}
    rtis: dict[int, RTI] = {}
    plants: dict[int, Plant] = {}
    hubs: dict[int, Hub] = {}
    zones: dict[int, Zone] = {}
    modes: dict[int, Mode] = {}
    edges: dict[int, Edge] = {}

    @field_serializer("rtis", "plants", "hubs", "zones", "modes", "edges")
    @classmethod
    def ser_int_keyed(cls, v: dict) -> dict[str, dict]:
        return {str(k): val.model_dump() for k, val in v.items()}

    @field_validator("rtis", "plants", "hubs", "zones", "modes", "edges", mode="before")
    @classmethod
    def deser_int_keyed(cls, v):
        if not isinstance(v, dict):
            return v
        return {int(k) if isinstance(k, str) else k: val for k, val in v.items()}

    @classmethod
    def load_json(cls, path: str) -> Network:
        """Load from JSON, rebuilding Product/Mode dict keys from registries."""
        with open(path) as f:
            raw = json.load(f)

        # 1. Build simple lookups first
        products = {
            name: Product(**pdata) for name, pdata in raw.get("products", {}).items()
        }
        modes = {int(k): Mode(**mdata) for k, mdata in raw.get("modes", {}).items()}
        plants = {int(k): Plant(**pdata) for k, pdata in raw.get("plants", {}).items()}
        hubs = {int(k): Hub(**hdata) for k, hdata in raw.get("hubs", {}).items()}
        zones = {int(k): Zone(**zdata) for k, zdata in raw.get("zones", {}).items()}

        # 2. Rebuild RTIs with Product keys
        def rebuild_product_dict(d: dict[str, float]) -> dict[Product, float]:
            return {products[name]: val for name, val in d.items()}

        def rebuild_mode_dict(d: dict[str, float]) -> dict[Mode, float]:
            return {modes[int(k)]: val for k, val in d.items()}

        rtis = {}
        for k, rdata in raw.get("rtis", {}).items():
            processed_data = {
                **rdata,
                "product_capacity": rebuild_product_dict(
                    rdata.get("product_capacity", {})
                ),
                "return_inlay_fraction": rebuild_product_dict(
                    rdata.get("return_inlay_fraction", {})
                ),
            }
            rtis[int(k)] = RTI(**processed_data)

        # 3. Rebuild Edges with Product and Mode keys
        def parse_node(d: dict) -> Plant | Hub:
            data = {k: v for k, v in d.items() if k != "type"}
            return Plant(**data) if d["type"] == "Plant" else Hub(**data)

        edges = {}
        for k, edata in raw.get("edges", {}).items():
            edges[int(k)] = Edge(
                id=edata["id"],
                source=parse_node(edata["source"]),
                target=parse_node(edata["target"]),
                zone_id=edata["zone_id"],
                demand=rebuild_product_dict(edata.get("demand", {})),
                allowed_empties=[rtis[eid] for eid in edata.get("allowed_empties", [])],
                fixed_economic_cost_per_volume=rebuild_mode_dict(
                    edata.get("fixed_economic_cost_per_volume", {})
                ),
                fixed_co2_cost_per_volume=rebuild_mode_dict(
                    edata.get("fixed_co2_cost_per_volume", {})
                ),
                variable_economic_cost_per_volume=rebuild_mode_dict(
                    edata.get("variable_economic_cost_per_volume", {})
                ),
                variable_co2_cost_per_volume=rebuild_mode_dict(
                    edata.get("variable_co2_cost_per_volume", {})
                ),
            )

        return Network(
            products=products,
            rtis=rtis,
            plants=plants,
            hubs=hubs,
            zones=zones,
            modes=modes,
            edges=edges,
        )

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Products
    beer = Product(name="Beer", category="Beverage")
    juice = Product(name="Juice", category="Beverage")

    # RTIs
    keg = RTI(
        id=1,
        folded_height=40,
        full_height=60,
        length=40,
        width=40,
        average_useful_life=5.0,
        current_stock=1000,
        product_capacity={beer: 45.0, juice: 40.0},
        return_inlay_fraction={beer: 0.85, juice: 0.90},
    )
    crate = RTI(
        id=2,
        folded_height=40,
        full_height=60,
        length=40,
        width=40,
        average_useful_life=3.0,
        current_stock=500,
        product_capacity={juice: 25.0},
        return_inlay_fraction={juice: 0.80},
    )

    # Plants & Hubs
    plant_a = Plant(id=1, x=40.0, y=50.0, is_large=True)
    plant_b = Plant(id=2, x=60.0, y=70.0, is_large=False)
    hub_1 = Hub(id=1, x=45.0, y=55.0, zone_id=1, fixed_cost=10000.0, variable_cost=2.5)
    hub_2 = Hub(id=2, x=65.0, y=75.0, zone_id=2, fixed_cost=8000.0, variable_cost=3.0)

    # Zones
    zone_1 = Zone(id=1, x=42.0, y=52.0, plants=[plant_a], hubs=[hub_1])
    zone_2 = Zone(id=2, x=62.0, y=72.0, plants=[plant_b], hubs=[hub_2])

    # Modes
    truck = Mode(id=1, max_vol=100.0, min_vol=10.0, min_q=1.0, max_q=20.0)
    rail = Mode(id=2, max_vol=500.0, min_vol=50.0, min_q=5.0, max_q=100.0)

    # Edges
    edge_1 = Edge(
        id=1,
        source=plant_a,
        target=hub_1,
        zone_id=1,
        demand={beer: 200.0, juice: 150.0},
        allowed_empties=[keg, crate],
        fixed_economic_cost_per_volume={truck: 5.0, rail: 3.0},
        fixed_co2_cost_per_volume={truck: 1.2, rail: 0.4},
        variable_economic_cost_per_volume={truck: 0.8, rail: 0.3},
        variable_co2_cost_per_volume={truck: 0.2, rail: 0.05},
    )
    edge_2 = Edge(
        id=2,
        source=hub_1,
        target=hub_2,
        zone_id=2,
        demand={beer: 100.0},
        allowed_empties=[keg],
        fixed_economic_cost_per_volume={truck: 6.0},
        fixed_co2_cost_per_volume={truck: 1.5},
        variable_economic_cost_per_volume={truck: 1.0},
        variable_co2_cost_per_volume={truck: 0.25},
    )

    # Build network
    network = Network(
        products={"Beer": beer, "Juice": juice},
        rtis={1: keg, 2: crate},
        plants={1: plant_a, 2: plant_b},
        hubs={1: hub_1, 2: hub_2},
        zones={1: zone_1, 2: zone_2},
        modes={1: truck, 2: rail},
        edges={1: edge_1, 2: edge_2},
    )

    # Save
    json_path = "./outputs/network.json"
    network.save_json(json_path)
    print("=== SAVED JSON ===")
    print(network.model_dump_json(indent=2))

    # Round-trip
    loaded = Network.load_json(json_path)
    print("\n=== ROUND-TRIP CHECK ===")
    print(f"Products:       {list(loaded.products.keys())}")
    print(f"Edge 1 demand:  {loaded.edges[1].demand}")
    print(f"Edge 1 costs:   {loaded.edges[1].fixed_economic_cost_per_volume}")
    print(f"RTI 1 capacity: {loaded.rtis[1].product_capacity}")

    # Verify keys are actual objects, not strings
    demand_key = list(loaded.edges[1].demand.keys())[0]
    cost_key = list(loaded.edges[1].fixed_economic_cost_per_volume.keys())[0]
    print(f"\ndemand key type: {type(demand_key).__name__} -> {demand_key}")
    print(f"cost key type:   {type(cost_key).__name__} -> {cost_key}")
