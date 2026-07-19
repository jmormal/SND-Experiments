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
    volume: float = 0

    def __hash__(self):
        return hash((self.name, self.category))


class Mode(BaseModel, frozen=True):
    id: int
    max_vol: float
    min_vol: float
    min_q: float
    max_q: float

    fixed_environmental_cost: float = 1
    fixed_economical_cost: float = 1
    economic_cost_per_km: float = 1
    environmental_cost_per_km: float = 1

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

    # --- NEW: purchase cost and embodied emissions ---
    purchase_cost: float = 0.0  # EUR per unit
    embodied_co2: float = 0.0  # kg CO2 per unit

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

    def add_product(
        self, product: Product, capacity: float, return_inlay_fraction: float
    ) -> None:
        self.product_capacity[product] = capacity
        self.return_inlay_fraction[product] = return_inlay_fraction


class Plant(BaseModel):
    id: int
    x: float
    y: float
    is_large: bool
    zone_id: int = 0
    incoming_demand: dict[Product, float] = {}
    outgoing_demand: dict[Product, float] = {}
    possible_incoming_empties: list[RTI] = []
    possible_outgoing_empties: list[RTI] = []

    @field_serializer("incoming_demand", "outgoing_demand")
    @classmethod
    def ser_demand(cls, v: dict[Product, float]) -> dict[str, float]:
        return {p.name: val for p, val in v.items()}

    @field_validator("incoming_demand", "outgoing_demand", mode="before")
    @classmethod
    def deser_demand(cls, v):
        if not isinstance(v, dict):
            return v
        # If keys are strings (from JSON), just return empty — edges will repopulate
        first_key = next(iter(v), None)
        if isinstance(first_key, str):
            return {}
        return v

    @field_serializer("possible_incoming_empties", "possible_outgoing_empties")
    @classmethod
    def ser_rtis(cls, v: list[RTI]) -> list[int]:
        return [r.id for r in v]

    @field_validator(
        "possible_incoming_empties", "possible_outgoing_empties", mode="before"
    )
    @classmethod
    def deser_rtis(cls, v):
        if not isinstance(v, list):
            return v
        if v and isinstance(v[0], int):
            return []
        return v

    def calculate_possible_empties(self, rtis: dict[int, RTI]):
        # Outgoing products need empty RTIs to arrive (incoming empties)
        outgoing_products = set(self.outgoing_demand.keys())
        self.possible_incoming_empties = [
            rti
            for rti in rtis.values()
            if outgoing_products & set(rti.product_capacity.keys())
        ]

        # Incoming products arrive in RTIs that then leave empty (outgoing empties)
        incoming_products = set(self.incoming_demand.keys())
        self.possible_outgoing_empties = [
            rti
            for rti in rtis.values()
            if incoming_products & set(rti.product_capacity.keys())
        ]


class Hub(BaseModel):
    id: int
    x: float
    y: float
    fixed_economic_cost_per_volume: float = 0
    fixed_co2_cost_per_volume: float = 0
    variable_economic_cost_per_volume: float = 0
    variable_co2_cost_per_volume: float = 0
    zone_id: int = 0


class Zone(BaseModel):
    id: int
    x: float
    y: float
    importance: float = 0
    weight_in: float = 0
    weight_out: float = 0
    plants: list[Plant] = []
    hubs: list[Hub] = []


class Edge(BaseModel):
    id: int
    source: Plant | Hub
    target: Plant | Hub
    zone_id: int

    # Production Costs
    demand: dict[Product, float] = {}
    safety_stock_empties: dict[
        Product, float
    ] = {}  # Safety Stocks needed for production at source
    safety_stocks_fulls: dict[
        Product, float
    ] = {}  # Safety Stocks of fulls at destination

    # Allowd Modes
    allowed_empties: list[RTI] = []
    allowed_modes: list[Mode] = []

    lead_time: dict[Mode, float] = {}

    # Costs
    fixed_economic_cost_per_volume: dict[Mode, float] = {}
    fixed_co2_cost_per_volume: dict[Mode, float] = {}
    variable_economic_cost_per_volume: dict[Mode, float] = {}
    variable_co2_cost_per_volume: dict[Mode, float] = {}

    @field_serializer("demand", "safety_stock_empties", "safety_stocks_fulls")
    @classmethod
    def ser_demand(cls, v: dict[Product, float]) -> dict[str, float]:
        return {p.name: val for p, val in v.items()}

    @field_serializer(
        "fixed_economic_cost_per_volume",
        "fixed_co2_cost_per_volume",
        "variable_economic_cost_per_volume",
        "variable_co2_cost_per_volume",
        "lead_time",
    )
    @classmethod
    def ser_mode_dict(cls, v: dict[Mode, float]) -> dict[str, float]:
        return {str(m.id): val for m, val in v.items()}

    @field_serializer("source", "target")
    @classmethod
    def ser_node(cls, v: Plant | Hub) -> dict:
        return {"type": type(v).__name__, "id": v.id}

    @field_serializer("allowed_empties")
    @classmethod
    def ser_empties(cls, v: list[RTI]) -> list[int]:
        return [r.id for r in v]

    @field_serializer("allowed_modes")
    @classmethod
    def ser_modes(cls, v: list[Mode]) -> list[int]:
        return [m.id for m in v]

    def model_post_init(self, __context):
        if isinstance(self.source, Plant):
            for product, qty in self.demand.items():
                self.source.outgoing_demand[product] = (
                    self.source.outgoing_demand.get(product, 0.0) + qty
                )
        if isinstance(self.target, Plant):
            for product, qty in self.demand.items():
                self.target.incoming_demand[product] = (
                    self.target.incoming_demand.get(product, 0.0) + qty
                )


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
        plants = {
            int(k): Plant(
                **{
                    key: val
                    for key, val in pdata.items()
                    if key
                    not in (
                        "incoming_demand",
                        "outgoing_demand",
                        "possible_incoming_empties",
                        "possible_outgoing_empties",
                    )
                }
            )
            for k, pdata in raw.get("plants", {}).items()
        }
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
        # def parse_node(d: dict) -> Plant | Hub:
        #     data = {k: v for k, v in d.items() if k != "type"}
        #     return Plant(**data) if d["type"] == "Plant" else Hub(**data)
        def parse_node(d: dict) -> Plant | Hub:
            if d["type"] == "Plant":
                return plants[d["id"]]
            else:
                return hubs[d["id"]]

        edges = {}
        for k, edata in raw.get("edges", {}).items():
            edges[int(k)] = Edge(
                id=edata["id"],
                source=parse_node(edata["source"]),
                target=parse_node(edata["target"]),
                zone_id=edata["zone_id"],
                demand=rebuild_product_dict(edata.get("demand", {})),
                safety_stocks_fulls=rebuild_product_dict(
                    edata.get("safety_stocks_fulls", {})
                ),
                safety_stock_empties=rebuild_product_dict(
                    edata.get("safety_stock_empties", {})
                ),
                allowed_empties=[rtis[eid] for eid in edata.get("allowed_empties", [])],
                allowed_modes=[modes[mid] for mid in edata.get("allowed_modes", [])],
                lead_time=rebuild_mode_dict(edata.get("lead_time", {})),
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

    def add_edges_model(self):
        # Write now we have the defined the fulls, but we still need to add the edges of empties.
        num_edges = len(self.edges)
        # Hubs
        for plant in self.plants.values():
            plant.calculate_possible_empties(self.rtis)
        for hub in self.hubs.values():
            # Hub To Hubs all type
            for plant in self.plants.values():
                # Needed for plant:
                # We check the product. If
                if plant.possible_incoming_empties:
                    self.edges[num_edges] = Edge(
                        id=num_edges,
                        source=hub,
                        target=plant,
                        zone_id=2,
                        allowed_empties=plant.possible_incoming_empties,
                    )
                    num_edges += 1

                if plant.possible_outgoing_empties:
                    self.edges[num_edges] = Edge(
                        id=num_edges,
                        source=hub,
                        target=plant,
                        zone_id=2,
                        allowed_empties=plant.possible_outgoing_empties,
                    )
                    num_edges += 1
            # Excee from plant:

            # Hub To Hubs all type
            for hub1 in self.hubs.values():
                if hub1.id == hub.id:
                    continue
                list_rtis = list(self.rtis.values())
                self.edges[num_edges] = Edge(
                    id=num_edges,
                    source=hub,
                    target=hub1,
                    zone_id=2,
                    allowed_empties=list_rtis,
                )

        # We make available only to those who needed by those how have spare

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))
