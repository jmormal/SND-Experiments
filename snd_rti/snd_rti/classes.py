from dataclasses import dataclass, field


@dataclass(frozen=True)  # frozen=True makes it hashable, so it can be a dict key
class Product:
    name: str
    category: str


@dataclass
class RTI:
    id: int
    volume_full: float
    volume_empty: float
    average_useful_life: float
    current_stock: int
    product_capacity: dict[Product, float] = field(default_factory=dict)
    return_inlay_fraction: dict[Product, float] = field(default_factory=dict)


@dataclass
class Plant:
    id: int
    x: float
    y: float
    is_large: bool


@dataclass
class Hub:
    id: int
    x: float
    y: float
    zone_id: int
    fixed_cost: float
    variable_cost: float


@dataclass
class Zone:
    id: int
    x: float
    y: float
    plants: list[Plant] = field(default_factory=list)
    hubs: list[Hub] = field(default_factory=list)


@dataclass(frozen=True)  # frozen=True makes it hashable, so it can be a dict key
class Mode:
    id: int
    max_vol: float
    min_vol: float
    min_q: float
    max_q: float


@dataclass
class Edge:
    id: int
    source: Plant | Hub
    target: Plant | Hub
    zone_id: int
    demand: dict[Product, float] = field(default_factory=dict)
    allowed_empties: list[RTI] = field(default_factory=list)
    fixed_economic_cost_per_volume: dict[Mode, float] = field(default_factory=dict)
    fixed_co2_cost_per_volume: dict[Mode, float] = field(default_factory=dict)
    variable_economic_cost_per_volume: dict[Mode, float] = field(default_factory=dict)
    variable_co2_cost_per_volume: dict[Mode, float] = field(default_factory=dict)


@dataclass
class Network:
    products: dict[str, Product] = field(default_factory=dict)
    rtis: dict[int, RTI] = field(default_factory=dict)
    plants: dict[int, Plant] = field(default_factory=dict)
    hubs: dict[int, Hub] = field(default_factory=dict)
    zones: dict[int, Zone] = field(default_factory=dict)
    modes: dict[int, Mode] = field(default_factory=dict)
    edges: dict[int, Edge] = field(default_factory=dict)
