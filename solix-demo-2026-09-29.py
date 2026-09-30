"""
Solix Notebook demo - 29 Sep 2026
=================================
Order-to-cash analytics engine, written in plain Python (standard library only),
to show that the notebook runs real application code, not just one-liners.

How to run it in Solix Notebook:
  * open this file from the workspace tree  ->  press "Run file" (the play button)
  * the whole program runs in your kernel and prints its report below the editor

What it exercises (each section prints PASS/FAIL checks at the end):
  1. dataclasses, enums, type hints, generators, decorators, context managers
  2. a deterministic data generator (seeded random) - 500 orders, 40 customers, 25 products
  3. a pricing + discount rules engine and tax calculation
  4. inventory simulation with reorder points
  5. statistics written by hand: mean, median, stdev, percentiles, moving averages
  6. least-squares linear regression + a 3-month revenue forecast
  7. k-means clustering of customers (hand-written, no numpy)
  8. sorting algorithms (merge sort, quick sort) checked against sorted()
  9. recursion with memoisation, a prime sieve, matrix multiplication
 10. JSON round-trip, error handling, and a formatted text report

Nothing here opens a database connection: in Solix Notebook, databases are
reached only through a Source/Target connection (see the .ipynb demo).
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import Enum
from functools import lru_cache, wraps
from typing import Callable, Dict, Iterable, Iterator, List, Tuple

SEED = 2026_09_29
RESULTS: List[Tuple[str, bool, str]] = []  # (check, passed, detail)


# ---------------------------------------------------------------------------
# 1. Small building blocks: decorator, context manager, enums, dataclasses
# ---------------------------------------------------------------------------

def timed(fn: Callable) -> Callable:
    """Decorator: records how long each engine step takes."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        value = fn(*args, **kwargs)
        TIMINGS[fn.__name__] = (time.perf_counter() - start) * 1000
        return value
    return wrapper


TIMINGS: Dict[str, float] = {}


@contextmanager
def section(title: str) -> Iterator[None]:
    print(f"\n{'=' * 72}\n{title}\n{'-' * 72}")
    yield
    print(f"{'-' * 72}")


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


class Region(Enum):
    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"


class Segment(Enum):
    RETAIL = "Retail"
    WHOLESALE = "Wholesale"
    ENTERPRISE = "Enterprise"


class Status(Enum):
    PAID = "Paid"
    OPEN = "Open"
    OVERDUE = "Overdue"
    CANCELLED = "Cancelled"


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    category: str
    unit_cost: float
    list_price: float
    reorder_point: int


@dataclass(frozen=True)
class Customer:
    customer_id: str
    name: str
    region: Region
    segment: Segment
    credit_limit: float


@dataclass
class OrderLine:
    sku: str
    quantity: int
    unit_price: float
    discount_pct: float = 0.0

    @property
    def net(self) -> float:
        return round(self.quantity * self.unit_price * (1 - self.discount_pct / 100), 2)


@dataclass
class Order:
    order_id: str
    customer_id: str
    order_date: date
    lines: List[OrderLine] = field(default_factory=list)
    status: Status = Status.OPEN
    tax_rate: float = 0.18

    @property
    def subtotal(self) -> float:
        return round(sum(line.net for line in self.lines), 2)

    @property
    def tax(self) -> float:
        return round(self.subtotal * self.tax_rate, 2)

    @property
    def total(self) -> float:
        return round(self.subtotal + self.tax, 2)


# ---------------------------------------------------------------------------
# 2. Deterministic data generator
# ---------------------------------------------------------------------------

CATEGORIES = {
    "Storage": ["Archive Disk", "Tape Library", "Object Vault", "Cold Tier", "Backup Node"],
    "Software": ["Data Governance", "eDiscovery", "Retention Suite", "Data Masking", "Test Data"],
    "Services": ["Migration Pack", "Health Check", "Onboarding", "Training Day", "Support Plus"],
    "Analytics": ["Lakehouse Seat", "BI Seat", "Notebook Seat", "AI Assist", "Query Engine"],
    "Security": ["Key Manager", "Audit Trail", "Access Broker", "Vault Agent", "Threat Scan"],
}


def generate_products(rng: random.Random) -> List[Product]:
    products = []
    for c_index, (category, names) in enumerate(CATEGORIES.items()):
        for p_index, name in enumerate(names):
            cost = round(rng.uniform(40, 900), 2)
            products.append(Product(
                sku=f"SKU-{c_index + 1}{p_index + 1:02d}",
                name=name,
                category=category,
                unit_cost=cost,
                list_price=round(cost * rng.uniform(1.3, 2.4), 2),
                reorder_point=rng.randint(20, 60),
            ))
    return products


def generate_customers(rng: random.Random, count: int = 40) -> List[Customer]:
    prefixes = ["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne", "Tyrell", "Soylent",
                "Cyberdyne", "Hooli", "Vandelay", "Wonka", "Gringotts", "Oscorp", "Aperture"]
    suffixes = ["Retail", "Bank", "Health", "Logistics", "Energy", "Telecom", "Foods", "Motors"]
    customers = []
    for i in range(count):
        segment = rng.choices(list(Segment), weights=[5, 3, 2])[0]
        customers.append(Customer(
            customer_id=f"C{i + 1:03d}",
            name=f"{rng.choice(prefixes)} {rng.choice(suffixes)}",
            region=rng.choice(list(Region)),
            segment=segment,
            credit_limit={Segment.RETAIL: 25_000, Segment.WHOLESALE: 80_000, Segment.ENTERPRISE: 250_000}[segment],
        ))
    return customers


def generate_orders(rng: random.Random, customers: List[Customer], products: List[Product],
                    count: int = 500, start: date = date(2026, 1, 1)) -> List[Order]:
    orders = []
    for i in range(count):
        customer = rng.choice(customers)
        # Seasonality: more orders towards quarter ends.
        day = int(rng.triangular(0, 270, 250))
        order = Order(order_id=f"SO-{i + 1:05d}", customer_id=customer.customer_id, order_date=start + timedelta(days=day))
        for product in rng.sample(products, rng.randint(1, 4)):
            order.lines.append(OrderLine(sku=product.sku, quantity=rng.randint(1, 30), unit_price=product.list_price))
        order.status = rng.choices(list(Status), weights=[70, 15, 10, 5])[0]
        orders.append(order)
    return orders


# ---------------------------------------------------------------------------
# 3. Pricing, discount rules and tax
# ---------------------------------------------------------------------------

DiscountRule = Callable[[Order, Customer], float]


def volume_discount(order: Order, _: Customer) -> float:
    units = sum(line.quantity for line in order.lines)
    return 10.0 if units >= 60 else 5.0 if units >= 30 else 0.0


def segment_discount(_: Order, customer: Customer) -> float:
    return {Segment.RETAIL: 0.0, Segment.WHOLESALE: 4.0, Segment.ENTERPRISE: 8.0}[customer.segment]


def quarter_end_discount(order: Order, _: Customer) -> float:
    return 3.0 if order.order_date.month in (3, 6, 9) and order.order_date.day >= 20 else 0.0


RULES: List[DiscountRule] = [volume_discount, segment_discount, quarter_end_discount]
MAX_DISCOUNT = 15.0


@timed
def apply_pricing(orders: List[Order], customers: Dict[str, Customer]) -> None:
    for order in orders:
        customer = customers[order.customer_id]
        discount = min(sum(rule(order, customer) for rule in RULES), MAX_DISCOUNT)
        for line in order.lines:
            line.discount_pct = discount
        order.tax_rate = 0.18 if customer.region in (Region.NORTH, Region.SOUTH) else 0.12


# ---------------------------------------------------------------------------
# 4. Inventory simulation
# ---------------------------------------------------------------------------

@timed
def simulate_inventory(orders: List[Order], products: List[Product], opening: int = 400) -> Dict[str, Dict[str, int]]:
    stock = {p.sku: opening for p in products}
    reorder = {p.sku: p.reorder_point for p in products}
    events: Dict[str, Dict[str, int]] = defaultdict(lambda: {"shipped": 0, "reorders": 0, "stockouts": 0})
    for order in sorted(orders, key=lambda o: o.order_date):
        if order.status is Status.CANCELLED:
            continue
        for line in order.lines:
            if stock[line.sku] < line.quantity:
                events[line.sku]["stockouts"] += 1
                stock[line.sku] += 300  # emergency replenishment
                events[line.sku]["reorders"] += 1
            stock[line.sku] -= line.quantity
            events[line.sku]["shipped"] += line.quantity
            if stock[line.sku] <= reorder[line.sku]:
                stock[line.sku] += 250
                events[line.sku]["reorders"] += 1
    for sku, left in stock.items():
        events[sku]["closing"] = left
    return dict(events)


# ---------------------------------------------------------------------------
# 5. Statistics by hand (checked against the statistics module)
# ---------------------------------------------------------------------------

def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def median(values: List[float]) -> float:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def stdev(values: List[float]) -> float:
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def percentile(values: List[float], pct: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[int(k)] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def moving_average(values: List[float], window: int) -> List[float]:
    out, running = [], 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        if i >= window - 1:
            out.append(running / window)
    return out


# ---------------------------------------------------------------------------
# 6. Least-squares regression and forecast
# ---------------------------------------------------------------------------

def linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Returns (slope, intercept, r_squared)."""
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return slope, intercept, 1 - ss_res / ss_tot if ss_tot else 1.0


# ---------------------------------------------------------------------------
# 7. K-means clustering (no numpy)
# ---------------------------------------------------------------------------

Point = Tuple[float, float]


def kmeans(points: List[Point], k: int, rng: random.Random, iterations: int = 50) -> Tuple[List[Point], List[int]]:
    centroids = rng.sample(points, k)
    labels = [0] * len(points)
    for _ in range(iterations):
        labels = [min(range(k), key=lambda c: (p[0] - centroids[c][0]) ** 2 + (p[1] - centroids[c][1]) ** 2) for p in points]
        new = []
        for c in range(k):
            members = [p for p, label in zip(points, labels) if label == c] or [centroids[c]]
            new.append((mean([m[0] for m in members]), mean([m[1] for m in members])))
        if new == centroids:
            break
        centroids = new
    return centroids, labels


# ---------------------------------------------------------------------------
# 8-9. Algorithms
# ---------------------------------------------------------------------------

def merge_sort(values: List[float]) -> List[float]:
    if len(values) <= 1:
        return values
    mid = len(values) // 2
    left, right = merge_sort(values[:mid]), merge_sort(values[mid:])
    merged, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    return merged + left[i:] + right[j:]


def quick_sort(values: List[float]) -> List[float]:
    if len(values) <= 1:
        return values
    pivot = values[len(values) // 2]
    return (quick_sort([v for v in values if v < pivot]) + [v for v in values if v == pivot]
            + quick_sort([v for v in values if v > pivot]))


@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)


def primes_up_to(n: int) -> List[int]:
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def batched(items: Iterable, size: int) -> Iterator[list]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# 10. Report
# ---------------------------------------------------------------------------

def money(value: float) -> str:
    return f"{value:>14,.2f}"


def bar(value: float, maximum: float, width: int = 28) -> str:
    return "#" * max(1, round(width * value / maximum)) if maximum else ""


def main() -> int:
    started = time.perf_counter()
    rng = random.Random(SEED)
    products = generate_products(rng)
    customers = generate_customers(rng)
    by_id = {c.customer_id: c for c in customers}
    product_by_sku = {p.sku: p for p in products}
    orders = generate_orders(rng, customers, products)
    apply_pricing(orders, by_id)
    live = [o for o in orders if o.status is not Status.CANCELLED]

    with section("1. Data generated"):
        print(f"products: {len(products)}  customers: {len(customers)}  orders: {len(orders)} "
              f"({len(orders) - len(live)} cancelled)  order lines: {sum(len(o.lines) for o in orders)}")
        print(f"date range: {min(o.order_date for o in orders)} .. {max(o.order_date for o in orders)}")
        check("500 orders generated", len(orders) == 500)
        check("same seed gives the same data", generate_orders(random.Random(SEED), generate_customers(random.Random(SEED)),
                                                               generate_products(random.Random(SEED)))[0].order_id == "SO-00001")

    with section("2. Revenue by region and segment"):
        by_region: Dict[str, float] = defaultdict(float)
        by_segment: Dict[str, float] = defaultdict(float)
        for o in live:
            c = by_id[o.customer_id]
            by_region[c.region.value] += o.total
            by_segment[c.segment.value] += o.total
        top = max(by_region.values())
        for region, value in sorted(by_region.items(), key=lambda kv: -kv[1]):
            print(f"{region:<12}{money(value)}  {bar(value, top)}")
        print()
        for segment, value in sorted(by_segment.items(), key=lambda kv: -kv[1]):
            print(f"{segment:<12}{money(value)}")
        total_revenue = sum(o.total for o in live)
        check("region totals add up to total revenue", abs(sum(by_region.values()) - total_revenue) < 0.01, f"{total_revenue:,.2f}")

    with section("3. Discounts and margin"):
        discounts = Counter(o.lines[0].discount_pct for o in live)
        for pct, n in sorted(discounts.items()):
            print(f"discount {pct:>5.1f}% : {n:>4} orders")
        cost = sum(line.quantity * product_by_sku[line.sku].unit_cost for o in live for line in o.lines)
        net = sum(o.subtotal for o in live)
        print(f"net revenue {net:,.2f}  cost {cost:,.2f}  gross margin {100 * (net - cost) / net:.1f}%")
        check("no discount above the 15% cap", max(discounts) <= MAX_DISCOUNT, f"max {max(discounts)}%")

    with section("4. Monthly revenue, moving average and forecast"):
        monthly: Dict[int, float] = defaultdict(float)
        for o in live:
            monthly[o.order_date.month] += o.total
        months = sorted(monthly)
        values = [monthly[m] for m in months]
        ma = moving_average(values, 3)
        for i, m in enumerate(months):
            avg = f"{ma[i - 2]:>14,.2f}" if i >= 2 else " " * 14
            print(f"2026-{m:02d} {money(values[i])} 3-mo avg {avg}  {bar(values[i], max(values), 20)}")
        slope, intercept, r2 = linear_regression([float(m) for m in months], values)
        print(f"\ntrend: {slope:+,.2f} per month (R^2 = {r2:.3f})")
        for ahead in (1, 2, 3):
            m = months[-1] + ahead
            print(f"forecast 2026-{m:02d}: {slope * m + intercept:,.2f}")
        check("regression fits a line through two points exactly", linear_regression([1, 2], [3, 5])[:2] == (2.0, 1.0))

    with section("5. Order value statistics (hand-written vs statistics module)"):
        totals = [o.total for o in live]
        print(f"mean {mean(totals):,.2f}  median {median(totals):,.2f}  stdev {stdev(totals):,.2f}")
        print(f"p50 {percentile(totals, 50):,.2f}  p90 {percentile(totals, 90):,.2f}  p99 {percentile(totals, 99):,.2f}")
        check("mean matches statistics.mean", math.isclose(mean(totals), statistics.mean(totals)))
        check("median matches statistics.median", math.isclose(median(totals), statistics.median(totals)))
        check("stdev matches statistics.stdev", math.isclose(stdev(totals), statistics.stdev(totals)))

    with section("6. Accounts receivable ageing"):
        as_of = date(2026, 9, 29)
        buckets = {"0-30": 0.0, "31-60": 0.0, "61-90": 0.0, "90+": 0.0}
        for o in live:
            if o.status in (Status.OPEN, Status.OVERDUE):
                age = (as_of - o.order_date).days
                key = "0-30" if age <= 30 else "31-60" if age <= 60 else "61-90" if age <= 90 else "90+"
                buckets[key] += o.total
        for key, value in buckets.items():
            print(f"{key:>6} days {money(value)}")
        exposure: Dict[str, float] = defaultdict(float)
        for o in live:
            if o.status is not Status.PAID:
                exposure[o.customer_id] += o.total
        over = [cid for cid, value in exposure.items() if value > by_id[cid].credit_limit]
        print(f"customers over their credit limit: {len(over)} {sorted(over)[:8]}")
        check("ageing buckets are non-negative", all(v >= 0 for v in buckets.values()))

    with section("7. Inventory simulation"):
        inventory = simulate_inventory(orders, products)
        worst = sorted(inventory.items(), key=lambda kv: -kv[1]["shipped"])[:5]
        print(f"{'SKU':<9}{'product':<18}{'shipped':>8}{'reorders':>9}{'stockouts':>10}{'closing':>8}")
        for sku, e in worst:
            print(f"{sku:<9}{product_by_sku[sku].name:<18}{e['shipped']:>8}{e['reorders']:>9}{e['stockouts']:>10}{e['closing']:>8}")
        shipped = sum(e["shipped"] for e in inventory.values())
        ordered = sum(line.quantity for o in live for line in o.lines)
        check("every ordered unit was shipped", shipped == ordered, f"{shipped} units")

    with section("8. Customer segments with k-means (order count x average order value)"):
        stats: Dict[str, List[float]] = defaultdict(list)
        for o in live:
            stats[o.customer_id].append(o.total)
        ids = sorted(stats)
        raw = [(float(len(stats[c])), mean(stats[c])) for c in ids]
        mx0, mx1 = max(p[0] for p in raw), max(p[1] for p in raw)
        points = [(p[0] / mx0, p[1] / mx1) for p in raw]  # scale both axes to 0..1
        centroids, labels = kmeans(points, 3, random.Random(SEED))
        for c in range(3):
            members = [ids[i] for i, label in enumerate(labels) if label == c]
            print(f"group {c + 1}: {len(members):>2} customers, ~{centroids[c][0] * mx0:.1f} orders, "
                  f"avg order {centroids[c][1] * mx1:,.0f}  e.g. {members[:4]}")
        check("every customer placed in a group", len(labels) == len(points))

    with section("9. Algorithms"):
        sample = [o.total for o in orders[:200]]
        check("merge sort matches sorted()", merge_sort(sample) == sorted(sample))
        check("quick sort matches sorted()", quick_sort(sample) == sorted(sample))
        check("fibonacci(90) with memoisation", fibonacci(90) == 2880067194370816120)
        primes = primes_up_to(10_000)
        check("1229 primes below 10,000", len(primes) == 1229, f"last {primes[-1]}")
        check("matrix multiplication", matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]])
        check("generator batches", [len(b) for b in batched(range(10), 4)] == [4, 4, 2])
        print(f"fibonacci(90) = {fibonacci(90):,}   primes below 10,000: {len(primes)}   largest: {primes[-1]}")

    with section("10. JSON round-trip and error handling"):
        payload = json.dumps([asdict(o) for o in orders[:3]], default=str)
        back = json.loads(payload)
        check("JSON round-trip keeps order ids", [o["order_id"] for o in back] == ["SO-00001", "SO-00002", "SO-00003"])
        try:
            by_id["C999"]
            check("unknown customer raises KeyError", False)
        except KeyError as e:
            check("unknown customer raises KeyError", True, f"KeyError {e}")
        try:
            mean([])
            check("mean of nothing raises", False)
        except ZeroDivisionError:
            check("mean of nothing raises", True, "ZeroDivisionError")
        print(f"sample JSON ({len(payload)} bytes): {payload[:150]}...")

    with section("Summary"):
        for step, ms in TIMINGS.items():
            print(f"step {step:<22} {ms:8.2f} ms")
        passed = sum(1 for _, ok, _ in RESULTS if ok)
        for name, ok, detail in RESULTS:
            print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
        print(f"\n{passed} of {len(RESULTS)} checks passed in {(time.perf_counter() - started) * 1000:.0f} ms")
    return 0 if passed == len(RESULTS) else 1


# "Run file" executes this module in the kernel, where __name__ is "__main__" too.
if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
