"""
TPC-H Benchmark Queries implemented using Ray Data APIs.
All 22 queries follow the pattern of using expressions and Arrow kernels.
"""

import ray
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from ray.data.aggregate import Sum, Mean, Count, Max, Min
from ray.data.expressions import col, udf
from ray.data.datatype import DataType
from time import perf_counter

# S3 bucket path for TPC-H data
S3_BUCKET = "s3://goutam-test-data/tpch-10"

# Default number of partitions for joins (adjust based on data size)
JOIN_PARTITIONS = 100


# ============================================================================
# Helper UDFs for common operations
# ============================================================================


@udf(return_dtype=DataType.float64())
def to_f64(arr: pa.Array) -> pa.Array:
    """Cast any numeric type to float64."""
    return pc.cast(arr, pa.float64())


@udf(return_dtype=DataType.int64())
def extract_year(arr: pa.Array) -> pa.Array:
    """Extract year from date."""
    return pc.year(arr)


@udf(return_dtype=DataType.string())
def substring(arr: pa.Array, start: int, length: int) -> pa.Array:
    """Extract substring from string."""
    return pc.utf8_slice_codeunits(arr, start, start + length)


# String operation UDFs (since col().str methods don't exist in Ray Data expressions)
@udf(return_dtype=DataType.bool())
def str_endswith(arr: pa.Array, suffix: str) -> pa.Array:
    """Check if string ends with suffix."""
    return pc.ends_with(arr, suffix)


@udf(return_dtype=DataType.bool())
def str_startswith(arr: pa.Array, prefix: str) -> pa.Array:
    """Check if string starts with prefix."""
    return pc.starts_with(arr, prefix)


@udf(return_dtype=DataType.bool())
def str_contains(arr: pa.Array, pattern: str) -> pa.Array:
    """Check if string contains pattern."""
    return pc.match_substring(arr, pattern)


@udf(return_dtype=DataType.bool())
def str_match(arr: pa.Array, pattern: str) -> pa.Array:
    """Check if string matches regex pattern."""
    return pc.match_substring_regex(arr, pattern)


# ============================================================================
# Query Implementations
# ============================================================================


def q1():
    """
    Q1: Pricing Summary Report
    Aggregates lineitem by return flag and line status.
    """
    ds = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")
    cutoff = np.datetime64("1998-09-24")  # date '1998-12-01' - 68 days
    ds = ds.filter(expr=col("l_shipdate") <= cutoff)

    # Build float views + derived columns
    ds = (
        ds.with_column("l_quantity_f", to_f64(col("l_quantity")))
        .with_column("l_extendedprice_f", to_f64(col("l_extendedprice")))
        .with_column("l_discount_f", to_f64(col("l_discount")))
        .with_column("l_tax_f", to_f64(col("l_tax")))
        .with_column(
            "disc_price",
            to_f64(col("l_extendedprice")) * (1 - to_f64(col("l_discount"))),
        )
        .with_column("charge", col("disc_price") * (1 + to_f64(col("l_tax"))))
    )

    # Drop original DECIMALs
    ds = ds.select_columns(
        [
            "l_returnflag",
            "l_linestatus",
            "l_quantity_f",
            "l_extendedprice_f",
            "l_discount_f",
            "disc_price",
            "charge",
        ]
    )

    result = (
        ds.groupby(["l_returnflag", "l_linestatus"])
        .aggregate(
            Sum(on="l_quantity_f", alias_name="sum_qty"),
            Sum(on="l_extendedprice_f", alias_name="sum_base_price"),
            Sum(on="disc_price", alias_name="sum_disc_price"),
            Sum(on="charge", alias_name="sum_charge"),
            Mean(on="l_quantity_f", alias_name="avg_qty"),
            Mean(on="l_extendedprice_f", alias_name="avg_price"),
            Mean(on="l_discount_f", alias_name="avg_disc"),
            Count(alias_name="count_order"),
        )
        .sort(key=["l_returnflag", "l_linestatus"])
        .select_columns(
            [
                "l_returnflag",
                "l_linestatus",
                "sum_qty",
                "sum_base_price",
                "sum_disc_price",
                "sum_charge",
                "avg_qty",
                "avg_price",
                "avg_disc",
                "count_order",
            ]
        )
    )

    return result


def q2():
    """
    Q2: Minimum Cost Supplier
    Finds suppliers with minimum supply cost for a specific part type and size in a region.
    """
    # Read all required tables
    part = ray.data.read_parquet(f"{S3_BUCKET}/part.parquet")
    supplier = ray.data.read_parquet(f"{S3_BUCKET}/supplier.parquet")
    partsupp = ray.data.read_parquet(f"{S3_BUCKET}/partsupp.parquet")
    nation = ray.data.read_parquet(f"{S3_BUCKET}/nation.parquet")
    region = ray.data.read_parquet(f"{S3_BUCKET}/region.parquet")

    # Filter part
    part = part.filter(expr=(col("p_size") == 48) & str_endswith(col("p_type"), "TIN"))

    # Filter region
    region = region.filter(expr=col("r_name") == "ASIA")

    # Join nation with region
    nation = nation.join(
        region, "inner", JOIN_PARTITIONS, on=("n_regionkey",), right_on=("r_regionkey",)
    )

    # Join supplier with nation
    supplier = supplier.join(
        nation, "inner", JOIN_PARTITIONS, on=("s_nationkey",), right_on=("n_nationkey",)
    )

    # Join partsupp with supplier
    ps_supp = partsupp.join(
        supplier, "inner", JOIN_PARTITIONS, on=("ps_suppkey",), right_on=("s_suppkey",)
    )

    # Join with part
    result = part.join(
        ps_supp, "inner", JOIN_PARTITIONS, on=("p_partkey",), right_on=("ps_partkey",)
    )

    # Calculate minimum cost per part (for filtering)
    # First, compute min cost for each partkey
    min_cost = result.groupby("p_partkey").aggregate(
        Min(on="ps_supplycost", alias_name="min_cost")
    )

    # Join back to get only rows with minimum cost
    result = result.join(
        min_cost, "inner", JOIN_PARTITIONS, on=("p_partkey",), right_on=("p_partkey",)
    )
    result = result.filter(expr=col("ps_supplycost") == col("min_cost"))

    # Select and sort
    result = (
        result.select_columns(
            [
                "s_acctbal",
                "s_name",
                "n_name",
                "p_partkey",
                "p_mfgr",
                "s_address",
                "s_phone",
                "s_comment",
            ]
        )
        .sort(
            key=["s_acctbal", "n_name", "s_name", "p_partkey"],
            descending=[True, False, False, False],
        )
        .limit(100)
    )

    return result


def q3():
    """
    Q3: Shipping Priority
    Retrieves top unshipped orders with highest revenue for a market segment.
    """
    customer = ray.data.read_parquet(f"{S3_BUCKET}/customer.parquet")
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")

    # Filter customer
    customer = customer.filter(expr=col("c_mktsegment") == "BUILDING")

    # Filter orders
    orders = orders.filter(expr=col("o_orderdate") < np.datetime64("1995-03-15"))

    # Filter lineitem
    lineitem = lineitem.filter(expr=col("l_shipdate") > np.datetime64("1995-03-15"))

    # Convert decimals to float
    lineitem = (
        lineitem.with_column("l_extendedprice_f", to_f64(col("l_extendedprice")))
        .with_column("l_discount_f", to_f64(col("l_discount")))
        .with_column(
            "revenue", to_f64(col("l_extendedprice")) * (1 - to_f64(col("l_discount")))
        )
    )

    # Join customer with orders
    co = customer.join(
        orders, "inner", JOIN_PARTITIONS, on=("c_custkey",), right_on=("o_custkey",)
    )

    # Join with lineitem
    result = co.join(
        lineitem, "inner", JOIN_PARTITIONS, on=("o_orderkey",), right_on=("l_orderkey",)
    )

    # Select needed columns before groupby
    result = result.select_columns(
        ["l_orderkey", "o_orderdate", "o_shippriority", "revenue"]
    )

    # Group by and aggregate
    result = (
        result.groupby(["l_orderkey", "o_orderdate", "o_shippriority"])
        .aggregate(Sum(on="revenue", alias_name="revenue"))
        .sort(key=["revenue", "o_orderdate"], descending=[True, False])
        .limit(10)
        .select_columns(["l_orderkey", "revenue", "o_orderdate", "o_shippriority"])
    )

    return result


def q4():
    """
    Q4: Order Priority Checking
    Counts orders by priority where lineitem was committed late.
    """
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")

    # Filter orders
    orders = orders.filter(
        expr=(col("o_orderdate") >= np.datetime64("1995-04-01"))
        & (col("o_orderdate") < np.datetime64("1995-07-01"))  # +3 months
    )

    # Filter lineitem (where commit date < receipt date)
    lineitem = lineitem.filter(expr=col("l_commitdate") < col("l_receiptdate"))

    # Get unique order keys from filtered lineitem
    lineitem_orders = (
        lineitem.select_columns(["l_orderkey"])
        .groupby("l_orderkey")
        .count()
        .drop_columns(["count()"])
    )

    # Semi-join: keep only orders that exist in lineitem_orders
    result = orders.join(
        lineitem_orders,
        "inner",
        JOIN_PARTITIONS,
        on=("o_orderkey",),
        right_on=("l_orderkey",),
    )

    # Group by priority and count
    result = (
        result.groupby("o_orderpriority")
        .aggregate(Count(alias_name="order_count"))
        .sort(key="o_orderpriority")
    )

    return result


def q5():
    """
    Q5: Local Supplier Volume
    Lists revenue from lineitem supplied by local suppliers in a region for a given year.
    """
    customer = ray.data.read_parquet(f"{S3_BUCKET}/customer.parquet")
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")
    supplier = ray.data.read_parquet(f"{S3_BUCKET}/supplier.parquet")
    nation = ray.data.read_parquet(f"{S3_BUCKET}/nation.parquet")
    region = ray.data.read_parquet(f"{S3_BUCKET}/region.parquet")

    # Filter orders by date
    orders = orders.filter(
        expr=(col("o_orderdate") >= np.datetime64("1994-01-01"))
        & (col("o_orderdate") < np.datetime64("1995-01-01"))
    )

    # Filter region
    region = region.filter(expr=col("r_name") == "AFRICA")

    # Join nation with region
    nation = nation.join(
        region, "inner", JOIN_PARTITIONS, on=("n_regionkey",), right_on=("r_regionkey",)
    )

    # Join customer with nation
    customer = customer.join(
        nation, "inner", JOIN_PARTITIONS, on=("c_nationkey",), right_on=("n_nationkey",)
    )

    # Join supplier with nation (need to read nation again since we modified it)
    nation2 = ray.data.read_parquet(f"{S3_BUCKET}/nation.parquet")
    nation2 = nation2.join(
        region.select_columns(["r_regionkey", "r_name"]),
        "inner",
        JOIN_PARTITIONS,
        on=("n_regionkey",),
        right_on=("r_regionkey",),
    )
    supplier = supplier.join(
        nation2,
        "inner",
        JOIN_PARTITIONS,
        on=("s_nationkey",),
        right_on=("n_nationkey",),
    )

    # Join customer with orders
    co = customer.join(
        orders, "inner", JOIN_PARTITIONS, on=("c_custkey",), right_on=("o_custkey",)
    )

    # Join with lineitem
    col_result = co.join(
        lineitem, "inner", JOIN_PARTITIONS, on=("o_orderkey",), right_on=("l_orderkey",)
    )

    # Join with supplier (on both suppkey and nationkey)
    result = col_result.join(
        supplier,
        "inner",
        JOIN_PARTITIONS,
        on=["l_suppkey", "c_nationkey"],
        right_on=["s_suppkey", "s_nationkey"],
    )

    # Calculate revenue
    result = result.with_column(
        "revenue", to_f64(col("l_extendedprice")) * (1 - to_f64(col("l_discount")))
    )

    # Group and aggregate
    result = (
        result.groupby("n_name")
        .aggregate(Sum(on="revenue", alias_name="revenue"))
        .sort(key="revenue", descending=True)
    )

    return result


def q6():
    """
    Q6: Forecasting Revenue Change
    Computes revenue increase from eliminating certain discounts.
    """
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")

    # Apply filters
    lineitem = lineitem.filter(
        expr=(col("l_shipdate") >= np.datetime64("1994-01-01"))
        & (col("l_shipdate") < np.datetime64("1995-01-01"))
        & (to_f64(col("l_discount")) >= 0.03)
        & (to_f64(col("l_discount")) <= 0.05)
        & (to_f64(col("l_quantity")) < 24)
    )

    # Calculate revenue
    lineitem = lineitem.with_column(
        "revenue", to_f64(col("l_extendedprice")) * to_f64(col("l_discount"))
    )

    # Aggregate and return sum
    result = lineitem.select_columns(["revenue"])
    revenue_sum = result.sum(on="revenue")

    return ray.data.from_items([revenue_sum])


def q7():
    """
    Q7: Volume Shipping
    Determines shipping volume between two nations.
    """
    supplier = ray.data.read_parquet(f"{S3_BUCKET}/supplier.parquet")
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")
    customer = ray.data.read_parquet(f"{S3_BUCKET}/customer.parquet")
    nation = ray.data.read_parquet(f"{S3_BUCKET}/nation.parquet")

    # Filter lineitem by date
    lineitem = lineitem.filter(
        expr=(col("l_shipdate") >= np.datetime64("1995-01-01"))
        & (col("l_shipdate") <= np.datetime64("1996-12-31"))
    )

    # Filter nations
    nation1 = nation.filter(
        expr=(col("n_name") == "GERMANY") | (col("n_name") == "IRAQ")
    ).select_columns(["n_nationkey", "n_name"])

    nation2 = nation.filter(
        expr=(col("n_name") == "GERMANY") | (col("n_name") == "IRAQ")
    ).select_columns(["n_nationkey", "n_name"])

    # Join supplier with nation1
    supplier = supplier.join(
        nation1,
        "inner",
        JOIN_PARTITIONS,
        on=("s_nationkey",),
        right_on=("n_nationkey",),
    )
    supplier = supplier.rename_columns({"n_name": "supp_nation"})

    # Join customer with nation2
    customer = customer.join(
        nation2,
        "inner",
        JOIN_PARTITIONS,
        on=("c_nationkey",),
        right_on=("n_nationkey",),
    )
    customer = customer.rename_columns({"n_name": "cust_nation"})

    # Join supplier with lineitem
    sl = supplier.join(
        lineitem, "inner", JOIN_PARTITIONS, on=("s_suppkey",), right_on=("l_suppkey",)
    )

    # Join with orders
    slo = sl.join(
        orders, "inner", JOIN_PARTITIONS, on=("l_orderkey",), right_on=("o_orderkey",)
    )

    # Join with customer
    result = slo.join(
        customer, "inner", JOIN_PARTITIONS, on=("o_custkey",), right_on=("c_custkey",)
    )

    # Filter for the specific nation pairs
    @udf(return_dtype=DataType.bool())
    def is_valid_nation_pair(supp_nation: pa.Array, cust_nation: pa.Array) -> pa.Array:
        valid = pc.or_(
            pc.and_(pc.equal(supp_nation, "GERMANY"), pc.equal(cust_nation, "IRAQ")),
            pc.and_(pc.equal(supp_nation, "IRAQ"), pc.equal(cust_nation, "GERMANY")),
        )
        return valid

    result = result.filter(
        expr=is_valid_nation_pair(col("supp_nation"), col("cust_nation"))
    )

    # Extract year and calculate volume
    result = result.with_column("l_year", extract_year(col("l_shipdate"))).with_column(
        "volume", to_f64(col("l_extendedprice")) * (1 - to_f64(col("l_discount")))
    )

    # Select columns and group
    result = result.select_columns(["supp_nation", "cust_nation", "l_year", "volume"])

    result = (
        result.groupby(["supp_nation", "cust_nation", "l_year"])
        .aggregate(Sum(on="volume", alias_name="revenue"))
        .sort(key=["supp_nation", "cust_nation", "l_year"])
    )

    return result


def q8():
    """
    Q8: National Market Share
    Determines market share of a specific nation for a part type in a region.
    """
    part = ray.data.read_parquet(f"{S3_BUCKET}/part.parquet")
    supplier = ray.data.read_parquet(f"{S3_BUCKET}/supplier.parquet")
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")
    customer = ray.data.read_parquet(f"{S3_BUCKET}/customer.parquet")
    nation = ray.data.read_parquet(f"{S3_BUCKET}/nation.parquet")
    region = ray.data.read_parquet(f"{S3_BUCKET}/region.parquet")

    # Filter part
    part = part.filter(expr=col("p_type") == "LARGE PLATED STEEL")

    # Filter orders
    orders = orders.filter(
        expr=(col("o_orderdate") >= np.datetime64("1995-01-01"))
        & (col("o_orderdate") <= np.datetime64("1996-12-31"))
    )

    # Filter region
    region = region.filter(expr=col("r_name") == "MIDDLE EAST")

    # Join nation with region (for customer)
    nation1 = nation.join(
        region, "inner", JOIN_PARTITIONS, on=("n_regionkey",), right_on=("r_regionkey",)
    )

    # Join customer with nation1
    customer = customer.join(
        nation1,
        "inner",
        JOIN_PARTITIONS,
        on=("c_nationkey",),
        right_on=("n_nationkey",),
    )

    # Join supplier with nation (keeping nation name as nation for supplier)
    nation2 = ray.data.read_parquet(f"{S3_BUCKET}/nation.parquet")
    supplier = supplier.join(
        nation2,
        "inner",
        JOIN_PARTITIONS,
        on=("s_nationkey",),
        right_on=("n_nationkey",),
    )
    supplier = supplier.rename_columns({"n_name": "nation"})

    # Build joins
    pl = part.join(
        lineitem, "inner", JOIN_PARTITIONS, on=("p_partkey",), right_on=("l_partkey",)
    )
    pls = pl.join(
        supplier, "inner", JOIN_PARTITIONS, on=("l_suppkey",), right_on=("s_suppkey",)
    )
    plso = pls.join(
        orders, "inner", JOIN_PARTITIONS, on=("l_orderkey",), right_on=("o_orderkey",)
    )
    result = plso.join(
        customer, "inner", JOIN_PARTITIONS, on=("o_custkey",), right_on=("c_custkey",)
    )

    # Extract year and calculate volume
    result = result.with_column("o_year", extract_year(col("o_orderdate"))).with_column(
        "volume", to_f64(col("l_extendedprice")) * (1 - to_f64(col("l_discount")))
    )

    # Create iraq_volume column (volume if nation is IRAQ, else 0)
    @udf(return_dtype=DataType.float64())
    def case_iraq(nation: pa.Array, volume: pa.Array) -> pa.Array:
        return pc.if_else(pc.equal(nation, "IRAQ"), volume, 0.0)

    result = result.with_column("iraq_volume", case_iraq(col("nation"), col("volume")))

    # Select columns for aggregation
    result = result.select_columns(["o_year", "volume", "iraq_volume"])

    # Group by year and compute market share
    result = (
        result.groupby("o_year")
        .aggregate(
            Sum(on="iraq_volume", alias_name="iraq_sum"),
            Sum(on="volume", alias_name="total_sum"),
        )
        .with_column("mkt_share", col("iraq_sum") / col("total_sum"))
        .sort(key="o_year")
        .select_columns(["o_year", "mkt_share"])
    )

    return result


def q9():
    """
    Q9: Product Type Profit Measure
    Determines profit for lineitems of a particular product type.
    """
    part = ray.data.read_parquet(f"{S3_BUCKET}/part.parquet")
    supplier = ray.data.read_parquet(f"{S3_BUCKET}/supplier.parquet")
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")
    partsupp = ray.data.read_parquet(f"{S3_BUCKET}/partsupp.parquet")
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")
    nation = ray.data.read_parquet(f"{S3_BUCKET}/nation.parquet")

    # Filter part
    part = part.filter(expr=str_contains(col("p_name"), "moccasin"))

    # Join supplier with nation
    supplier = supplier.join(
        nation, "inner", JOIN_PARTITIONS, on=("s_nationkey",), right_on=("n_nationkey",)
    )

    # Join part with lineitem
    pl = part.join(
        lineitem, "inner", JOIN_PARTITIONS, on=("p_partkey",), right_on=("l_partkey",)
    )

    # Join with supplier
    pls = pl.join(
        supplier, "inner", JOIN_PARTITIONS, on=("l_suppkey",), right_on=("s_suppkey",)
    )

    # Join with partsupp
    plsps = pls.join(
        partsupp,
        "inner",
        JOIN_PARTITIONS,
        on=["l_suppkey", "l_partkey"],
        right_on=["ps_suppkey", "ps_partkey"],
    )

    # Join with orders
    result = plsps.join(
        orders, "inner", JOIN_PARTITIONS, on=("l_orderkey",), right_on=("o_orderkey",)
    )

    # Calculate amount and extract year
    result = result.with_column("o_year", extract_year(col("o_orderdate"))).with_column(
        "amount",
        to_f64(col("l_extendedprice")) * (1 - to_f64(col("l_discount")))
        - to_f64(col("ps_supplycost")) * to_f64(col("l_quantity")),
    )

    # Group and aggregate
    result = (
        result.groupby(["n_name", "o_year"])
        .aggregate(Sum(on="amount", alias_name="sum_profit"))
        .sort(key=["n_name", "o_year"], descending=[False, True])
        .rename_columns({"n_name": "nation"})
    )

    return result


def q10():
    """
    Q10: Returned Item Reporting
    Identifies customers who returned items and their lost revenue.
    """
    customer = ray.data.read_parquet(f"{S3_BUCKET}/customer.parquet")
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")
    nation = ray.data.read_parquet(f"{S3_BUCKET}/nation.parquet")

    # Filter orders
    orders = orders.filter(
        expr=(col("o_orderdate") >= np.datetime64("1993-07-01"))
        & (col("o_orderdate") < np.datetime64("1993-10-01"))
    )

    # Filter lineitem
    lineitem = lineitem.filter(expr=col("l_returnflag") == "R")

    # Calculate revenue
    lineitem = lineitem.with_column(
        "revenue", to_f64(col("l_extendedprice")) * (1 - to_f64(col("l_discount")))
    )

    # Join customer with orders
    co = customer.join(
        orders, "inner", JOIN_PARTITIONS, on=("c_custkey",), right_on=("o_custkey",)
    )

    # Join with lineitem
    col_result = co.join(
        lineitem, "inner", JOIN_PARTITIONS, on=("o_orderkey",), right_on=("l_orderkey",)
    )

    # Join with nation
    result = col_result.join(
        nation, "inner", JOIN_PARTITIONS, on=("c_nationkey",), right_on=("n_nationkey",)
    )

    # Group and aggregate
    result = (
        result.groupby(
            [
                "c_custkey",
                "c_name",
                "c_acctbal",
                "c_phone",
                "n_name",
                "c_address",
                "c_comment",
            ]
        )
        .aggregate(Sum(on="revenue", alias_name="revenue"))
        .sort(key="revenue", descending=True)
        .limit(20)
    )

    return result


def q11():
    """
    Q11: Important Stock Identification
    Finds most important parts with significant stock value.
    """
    partsupp = ray.data.read_parquet(f"{S3_BUCKET}/partsupp.parquet")
    supplier = ray.data.read_parquet(f"{S3_BUCKET}/supplier.parquet")
    nation = ray.data.read_parquet(f"{S3_BUCKET}/nation.parquet")

    # Filter nation
    nation = nation.filter(expr=col("n_name") == "ALGERIA")

    # Join supplier with nation
    supplier = supplier.join(
        nation, "inner", JOIN_PARTITIONS, on=("s_nationkey",), right_on=("n_nationkey",)
    )

    # Join partsupp with supplier
    ps = partsupp.join(
        supplier, "inner", JOIN_PARTITIONS, on=("ps_suppkey",), right_on=("s_suppkey",)
    )

    # Calculate value
    ps = ps.with_column(
        "value", to_f64(col("ps_supplycost")) * to_f64(col("ps_availqty"))
    )

    # Group by partkey and compute sum
    result = ps.groupby("ps_partkey").aggregate(Sum(on="value", alias_name="value"))

    # To filter by having clause, we need the threshold
    # We'll compute total first, then filter
    # This requires materializing data which is expensive
    # For production, consider alternative approaches
    ps_agg = result.take_all()
    total = sum(row["value"] for row in ps_agg)
    threshold = total * 0.0001

    # Filter and sort
    result_data = [row for row in ps_agg if row["value"] > threshold]
    result_data.sort(key=lambda x: x["value"], reverse=True)

    # Convert back to dataset (not ideal, but necessary for this query pattern)
    return ray.data.from_items(result_data)


def q12():
    """
    Q12: Shipping Modes and Order Priority
    Determines shipping mode preference for high priority orders.
    """
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")

    # Filter lineitem
    lineitem = lineitem.filter(
        expr=col("l_shipmode").is_in(["FOB", "SHIP"])
        & (col("l_commitdate") < col("l_receiptdate"))
        & (col("l_shipdate") < col("l_commitdate"))
        & (col("l_receiptdate") >= np.datetime64("1995-01-01"))
        & (col("l_receiptdate") < np.datetime64("1996-01-01"))
    )

    # Join with orders
    result = lineitem.join(
        orders, "inner", JOIN_PARTITIONS, on=("l_orderkey",), right_on=("o_orderkey",)
    )

    # Create conditional columns
    @udf(return_dtype=DataType.int64())
    def is_high_priority(priority: pa.Array) -> pa.Array:
        return pc.if_else(
            pc.or_(pc.equal(priority, "1-URGENT"), pc.equal(priority, "2-HIGH")), 1, 0
        )

    @udf(return_dtype=DataType.int64())
    def is_low_priority(priority: pa.Array) -> pa.Array:
        return pc.if_else(
            pc.and_(
                pc.not_equal(priority, "1-URGENT"), pc.not_equal(priority, "2-HIGH")
            ),
            1,
            0,
        )

    result = result.with_column(
        "high_line", is_high_priority(col("o_orderpriority"))
    ).with_column("low_line", is_low_priority(col("o_orderpriority")))

    # Group and aggregate
    result = (
        result.groupby("l_shipmode")
        .aggregate(
            Sum(on="high_line", alias_name="high_line_count"),
            Sum(on="low_line", alias_name="low_line_count"),
        )
        .sort(key="l_shipmode")
    )

    return result


def q13():
    """
    Q13: Customer Distribution
    Shows distribution of customers by order count.
    """
    customer = ray.data.read_parquet(f"{S3_BUCKET}/customer.parquet")
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")

    # Filter orders (exclude those with specific comment pattern)
    orders = orders.filter(expr=~str_match(col("o_comment"), ".*express.*requests.*"))

    # Left outer join customer with orders
    co = customer.join(
        orders,
        "left_outer",
        JOIN_PARTITIONS,
        on=("c_custkey",),
        right_on=("o_custkey",),
    )

    # Count orders per customer
    c_orders = co.groupby("c_custkey").aggregate(
        Count(on="o_orderkey", alias_name="c_count")
    )

    # Count customers per order count
    result = (
        c_orders.groupby("c_count")
        .aggregate(Count(alias_name="custdist"))
        .sort(key=["custdist", "c_count"], descending=[True, True])
    )

    return result


def q14():
    """
    Q14: Promotion Effect
    Measures percentage of revenue from promotional parts.
    """
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")
    part = ray.data.read_parquet(f"{S3_BUCKET}/part.parquet")

    # Filter lineitem
    lineitem = lineitem.filter(
        expr=(col("l_shipdate") >= np.datetime64("1995-02-01"))
        & (col("l_shipdate") < np.datetime64("1995-03-01"))
    )

    # Join with part
    result = lineitem.join(
        part, "inner", JOIN_PARTITIONS, on=("l_partkey",), right_on=("p_partkey",)
    )

    # Calculate revenue components
    result = result.with_column(
        "revenue", to_f64(col("l_extendedprice")) * (1 - to_f64(col("l_discount")))
    )

    # Calculate promo revenue
    @udf(return_dtype=DataType.float64())
    def promo_revenue(p_type: pa.Array, revenue: pa.Array) -> pa.Array:
        return pc.if_else(pc.starts_with(p_type, "PROMO"), revenue, 0.0)

    result = result.with_column(
        "promo_rev", promo_revenue(col("p_type"), col("revenue"))
    )

    # Aggregate
    result = result.select_columns(["revenue", "promo_rev"])
    agg = result.aggregate(
        Sum(on="promo_rev", alias_name="promo_sum"),
        Sum(on="revenue", alias_name="total_sum"),
    )

    # Calculate percentage (aggregate returns a dict)
    promo_percentage = 100.0 * agg["promo_sum"] / agg["total_sum"]

    return ray.data.from_items([{"promo_revenue": promo_percentage}])


def q15():
    """
    Q15: Top Supplier
    Determines top supplier based on revenue in a time period.
    """
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")
    supplier = ray.data.read_parquet(f"{S3_BUCKET}/supplier.parquet")

    # Filter lineitem
    lineitem = lineitem.filter(
        expr=(col("l_shipdate") >= np.datetime64("1996-08-01"))
        & (col("l_shipdate") < np.datetime64("1996-11-01"))
    )

    # Calculate revenue
    lineitem = lineitem.with_column(
        "revenue", to_f64(col("l_extendedprice")) * (1 - to_f64(col("l_discount")))
    )

    # Group by supplier and compute total revenue
    revenue_ds = (
        lineitem.groupby("l_suppkey")
        .aggregate(Sum(on="revenue", alias_name="total_revenue"))
        .rename_columns({"l_suppkey": "supplier_no"})
    )

    # Find max revenue
    max_revenue_data = revenue_ds.aggregate(
        Max(on="total_revenue", alias_name="max_rev")
    )
    max_revenue = max_revenue_data["max_rev"]

    # Filter to suppliers with max revenue
    revenue_ds = revenue_ds.filter(expr=col("total_revenue") == max_revenue)

    # Join with supplier
    result = supplier.join(
        revenue_ds,
        "inner",
        JOIN_PARTITIONS,
        on=("s_suppkey",),
        right_on=("supplier_no",),
    )

    # Select and sort
    result = result.select_columns(
        ["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"]
    ).sort(key="s_suppkey")

    return result


def q16():
    """
    Q16: Parts/Supplier Relationship
    Counts suppliers who can supply parts meeting certain criteria.
    """
    partsupp = ray.data.read_parquet(f"{S3_BUCKET}/partsupp.parquet")
    part = ray.data.read_parquet(f"{S3_BUCKET}/part.parquet")
    supplier = ray.data.read_parquet(f"{S3_BUCKET}/supplier.parquet")

    # Filter part
    part = part.filter(
        expr=(col("p_brand") != "Brand#14")
        & (~str_startswith(col("p_type"), "SMALL PLATED"))
        & col("p_size").is_in([14, 6, 5, 31, 49, 15, 41, 47])
    )

    # Filter supplier (get suppliers to exclude)
    excluded_suppliers = supplier.filter(
        expr=str_match(col("s_comment"), ".*Customer.*Complaints.*")
    ).select_columns(["s_suppkey"])

    # Get excluded supplier keys as a set (need to materialize)
    excluded_keys = set(row["s_suppkey"] for row in excluded_suppliers.take_all())

    # Filter partsupp to exclude those suppliers
    @udf(return_dtype=DataType.bool())
    def not_in_excluded(suppkey: pa.Array) -> pa.Array:
        # Create a mask for suppkeys not in excluded set
        mask = pc.invert(pc.is_in(suppkey, pa.array(list(excluded_keys))))
        return mask

    partsupp = partsupp.filter(expr=not_in_excluded(col("ps_suppkey")))

    # Join with part
    result = partsupp.join(
        part, "inner", JOIN_PARTITIONS, on=("ps_partkey",), right_on=("p_partkey",)
    )

    # Group by brand, type, size and count distinct suppliers
    # Ray Data doesn't have count distinct in aggregate, so we need to work around
    result = (
        result.select_columns(["p_brand", "p_type", "p_size", "ps_suppkey"])
        .groupby(["p_brand", "p_type", "p_size", "ps_suppkey"])
        .count()
        .drop_columns(["count()"])
    )

    result = (
        result.groupby(["p_brand", "p_type", "p_size"])
        .aggregate(Count(alias_name="supplier_cnt"))
        .sort(
            key=["supplier_cnt", "p_brand", "p_type", "p_size"],
            descending=[True, False, False, False],
        )
    )

    return result


def q17():
    """
    Q17: Small-Quantity-Order Revenue
    Determines average yearly revenue for a specific part.
    """
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")
    part = ray.data.read_parquet(f"{S3_BUCKET}/part.parquet")

    # Filter part
    part = part.filter(
        expr=(col("p_brand") == "Brand#42") & (col("p_container") == "LG BAG")
    )

    # Calculate average quantity per part
    avg_qty = (
        lineitem.groupby("l_partkey")
        .aggregate(Mean(on="l_quantity", alias_name="avg_qty"))
        .with_column("threshold", col("avg_qty") * 0.2)
    )

    # Join part with lineitem
    pl = part.join(
        lineitem, "inner", JOIN_PARTITIONS, on=("p_partkey",), right_on=("l_partkey",)
    )

    # Join with average quantities
    result = pl.join(
        avg_qty, "inner", JOIN_PARTITIONS, on=("l_partkey",), right_on=("l_partkey",)
    )

    # Filter by quantity threshold
    result = result.filter(expr=to_f64(col("l_quantity")) < col("threshold"))

    # Calculate sum and divide by 7
    result = result.with_column("price", to_f64(col("l_extendedprice")))
    total = result.sum(on="price")

    # Materialize and compute avg_yearly (sum returns a dict)
    avg_yearly = total["sum(price)"] / 7.0

    return ray.data.from_items([{"avg_yearly": avg_yearly}])


def q18():
    """
    Q18: Large Volume Customer
    Lists customers with large orders.
    """
    customer = ray.data.read_parquet(f"{S3_BUCKET}/customer.parquet")
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")

    # Find large orders (sum of quantity > 313)
    large_orders = (
        lineitem.groupby("l_orderkey")
        .aggregate(Sum(on="l_quantity", alias_name="total_qty"))
        .filter(expr=col("total_qty") > 313)
        .select_columns(["l_orderkey"])
    )

    # Filter orders
    orders = orders.join(
        large_orders,
        "inner",
        JOIN_PARTITIONS,
        on=("o_orderkey",),
        right_on=("l_orderkey",),
    )

    # Join customer with orders
    co = customer.join(
        orders, "inner", JOIN_PARTITIONS, on=("c_custkey",), right_on=("o_custkey",)
    )

    # Join with lineitem
    result = co.join(
        lineitem, "inner", JOIN_PARTITIONS, on=("o_orderkey",), right_on=("l_orderkey",)
    )

    # Group and aggregate
    result = (
        result.groupby(
            ["c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"]
        )
        .aggregate(Sum(on="l_quantity", alias_name="sum_quantity"))
        .sort(key=["o_totalprice", "o_orderdate"], descending=[True, False])
        .limit(100)
    )

    return result


def q19():
    """
    Q19: Discounted Revenue
    Reports revenue from specific parts and shipping modes.
    """
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")
    part = ray.data.read_parquet(f"{S3_BUCKET}/part.parquet")

    # Filter lineitem
    lineitem = lineitem.filter(
        expr=col("l_shipmode").is_in(["AIR", "AIR REG"])
        & (col("l_shipinstruct") == "DELIVER IN PERSON")
    )

    # Join with part
    result = lineitem.join(
        part, "inner", JOIN_PARTITIONS, on=("l_partkey",), right_on=("p_partkey",)
    )

    # Create complex filter conditions
    @udf(return_dtype=DataType.bool())
    def matches_criteria(
        p_brand: pa.Array, p_container: pa.Array, p_size: pa.Array, l_quantity: pa.Array
    ) -> pa.Array:
        # Convert l_quantity to float for comparison
        l_qty_f = pc.cast(l_quantity, pa.float64())

        # Condition 1
        cond1 = pc.and_(
            pc.and_(
                pc.equal(p_brand, "Brand#21"),
                pc.is_in(
                    p_container, pa.array(["SM CASE", "SM BOX", "SM PACK", "SM PKG"])
                ),
            ),
            pc.and_(
                pc.and_(pc.greater_equal(l_qty_f, 8.0), pc.less_equal(l_qty_f, 18.0)),
                pc.and_(pc.greater_equal(p_size, 1), pc.less_equal(p_size, 5)),
            ),
        )

        # Condition 2
        cond2 = pc.and_(
            pc.and_(
                pc.equal(p_brand, "Brand#13"),
                pc.is_in(
                    p_container, pa.array(["MED BAG", "MED BOX", "MED PKG", "MED PACK"])
                ),
            ),
            pc.and_(
                pc.and_(pc.greater_equal(l_qty_f, 20.0), pc.less_equal(l_qty_f, 30.0)),
                pc.and_(pc.greater_equal(p_size, 1), pc.less_equal(p_size, 10)),
            ),
        )

        # Condition 3
        cond3 = pc.and_(
            pc.and_(
                pc.equal(p_brand, "Brand#52"),
                pc.is_in(
                    p_container, pa.array(["LG CASE", "LG BOX", "LG PACK", "LG PKG"])
                ),
            ),
            pc.and_(
                pc.and_(pc.greater_equal(l_qty_f, 30.0), pc.less_equal(l_qty_f, 40.0)),
                pc.and_(pc.greater_equal(p_size, 1), pc.less_equal(p_size, 15)),
            ),
        )

        return pc.or_(pc.or_(cond1, cond2), cond3)

    result = result.filter(
        expr=matches_criteria(
            col("p_brand"), col("p_container"), col("p_size"), col("l_quantity")
        )
    )

    # Calculate revenue
    result = result.with_column(
        "revenue", to_f64(col("l_extendedprice")) * (1 - to_f64(col("l_discount")))
    )

    # Sum revenue
    total = result.sum(on="revenue")

    return total


def q20():
    """
    Q20: Potential Part Promotion
    Identifies suppliers with excess inventory for a specific part.
    """
    supplier = ray.data.read_parquet(f"{S3_BUCKET}/supplier.parquet")
    nation = ray.data.read_parquet(f"{S3_BUCKET}/nation.parquet")
    partsupp = ray.data.read_parquet(f"{S3_BUCKET}/partsupp.parquet")
    part = ray.data.read_parquet(f"{S3_BUCKET}/part.parquet")
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")

    # Filter nation
    nation = nation.filter(expr=col("n_name") == "KENYA")

    # Filter part
    part = part.filter(expr=str_startswith(col("p_name"), "blanched"))

    # Filter lineitem
    lineitem = lineitem.filter(
        expr=(col("l_shipdate") >= np.datetime64("1993-01-01"))
        & (col("l_shipdate") < np.datetime64("1994-01-01"))
    )

    # Calculate 0.5 * sum(l_quantity) per part/supplier
    lineitem_agg = (
        lineitem.groupby(["l_partkey", "l_suppkey"])
        .aggregate(Sum(on="l_quantity", alias_name="sum_qty"))
        .with_column("threshold", col("sum_qty") * 0.5)
    )

    # Join part with partsupp
    ps = part.join(
        partsupp, "inner", JOIN_PARTITIONS, on=("p_partkey",), right_on=("ps_partkey",)
    )

    # Join with lineitem aggregates
    ps = ps.join(
        lineitem_agg,
        "inner",
        JOIN_PARTITIONS,
        on=["ps_partkey", "ps_suppkey"],
        right_on=["l_partkey", "l_suppkey"],
    )

    # Filter by availability threshold
    ps = ps.filter(expr=to_f64(col("ps_availqty")) > col("threshold"))

    # Get unique supplier keys
    valid_suppliers = (
        ps.select_columns(["ps_suppkey"])
        .groupby("ps_suppkey")
        .count()
        .drop_columns(["count()"])
    )

    # Join supplier with nation
    supplier = supplier.join(
        nation, "inner", JOIN_PARTITIONS, on=("s_nationkey",), right_on=("n_nationkey",)
    )

    # Filter to valid suppliers
    result = supplier.join(
        valid_suppliers,
        "inner",
        JOIN_PARTITIONS,
        on=("s_suppkey",),
        right_on=("ps_suppkey",),
    )

    # Select and sort
    result = result.select_columns(["s_name", "s_address"]).sort(key="s_name")

    return result


def q21():
    """
    Q21: Suppliers Who Kept Orders Waiting
    Identifies suppliers whose parts caused order delays.
    """
    supplier = ray.data.read_parquet(f"{S3_BUCKET}/supplier.parquet")
    lineitem = ray.data.read_parquet(f"{S3_BUCKET}/lineitem.parquet")
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")
    nation = ray.data.read_parquet(f"{S3_BUCKET}/nation.parquet")

    # Filter nation
    nation = nation.filter(expr=col("n_name") == "ARGENTINA")

    # Filter orders
    orders = orders.filter(expr=col("o_orderstatus") == "F")

    # Filter lineitem (l1: where receipt > commit)
    l1 = lineitem.filter(expr=col("l_receiptdate") > col("l_commitdate"))

    # For EXISTS condition: find orders with multiple suppliers
    l2 = (
        lineitem.select_columns(["l_orderkey", "l_suppkey"])
        .groupby(["l_orderkey", "l_suppkey"])
        .count()
        .drop_columns(["count()"])
    )
    orders_multi_supp = (
        l2.groupby("l_orderkey")
        .aggregate(Count(alias_name="supp_count"))
        .filter(expr=col("supp_count") > 1)
        .select_columns(["l_orderkey"])
    )

    # For NOT EXISTS condition: find orders where other supplier also failed
    l3 = lineitem.filter(expr=col("l_receiptdate") > col("l_commitdate"))
    l3 = l3.select_columns(["l_orderkey", "l_suppkey"]).rename_columns(
        {"l_orderkey": "l3_orderkey", "l_suppkey": "l3_suppkey"}
    )

    # Join l1 with l3 to find orderkeys where another supplier also failed
    l1_l3 = l1.join(
        l3, "inner", JOIN_PARTITIONS, on=("l_orderkey",), right_on=("l3_orderkey",)
    )
    l1_l3 = l1_l3.filter(expr=col("l_suppkey") != col("l3_suppkey"))
    orders_multi_fail = (
        l1_l3.select_columns(["l_orderkey"])
        .groupby("l_orderkey")
        .count()
        .drop_columns(["count()"])
    )

    # Get order keys to exclude
    exclude_keys = set(row["l_orderkey"] for row in orders_multi_fail.take_all())

    # Filter l1 to exclude these orders
    @udf(return_dtype=DataType.bool())
    def not_in_exclude(orderkey: pa.Array) -> pa.Array:
        mask = pc.invert(pc.is_in(orderkey, pa.array(list(exclude_keys))))
        return mask

    l1 = l1.filter(expr=not_in_exclude(col("l_orderkey")))

    # Keep only orders with multiple suppliers
    l1 = l1.join(
        orders_multi_supp,
        "inner",
        JOIN_PARTITIONS,
        on=("l_orderkey",),
        right_on=("l_orderkey",),
    )

    # Join with supplier
    sl = supplier.join(
        l1, "inner", JOIN_PARTITIONS, on=("s_suppkey",), right_on=("l_suppkey",)
    )

    # Join with orders
    slo = sl.join(
        orders, "inner", JOIN_PARTITIONS, on=("l_orderkey",), right_on=("o_orderkey",)
    )

    # Join with nation
    result = slo.join(
        nation, "inner", JOIN_PARTITIONS, on=("s_nationkey",), right_on=("n_nationkey",)
    )

    # Group and count
    result = (
        result.groupby("s_name")
        .aggregate(Count(alias_name="numwait"))
        .sort(key=["numwait", "s_name"], descending=[True, False])
        .limit(100)
    )

    return result


def q22():
    """
    Q22: Global Sales Opportunity
    Identifies potential customers in specific countries.
    """
    customer = ray.data.read_parquet(f"{S3_BUCKET}/customer.parquet")
    orders = ray.data.read_parquet(f"{S3_BUCKET}/orders.parquet")

    # Define country codes
    country_codes = ["24", "34", "16", "30", "33", "14", "13"]

    # Extract country code from phone
    @udf(return_dtype=DataType.string())
    def get_country_code(phone: pa.Array) -> pa.Array:
        return pc.utf8_slice_codeunits(phone, 0, 2)

    customer = customer.with_column("cntrycode", get_country_code(col("c_phone")))

    # Filter by country codes
    customer = customer.filter(expr=col("cntrycode").is_in(country_codes))

    # Calculate average account balance for positive balances
    avg_acctbal_ds = customer.filter(expr=to_f64(col("c_acctbal")) > 0.0)
    avg_data = avg_acctbal_ds.aggregate(Mean(on="c_acctbal", alias_name="avg_bal"))
    avg_acctbal = avg_data["avg_bal"]

    # Filter customers with balance > average
    customer = customer.filter(expr=to_f64(col("c_acctbal")) > avg_acctbal)

    # Find customers with no orders (NOT EXISTS)
    customer_with_orders = (
        orders.select_columns(["o_custkey"])
        .groupby("o_custkey")
        .count()
        .drop_columns(["count()"])
    )
    customers_no_orders_keys = set(row["c_custkey"] for row in customer.take_all())
    customers_with_orders_keys = set(
        row["o_custkey"] for row in customer_with_orders.take_all()
    )
    valid_customers = customers_no_orders_keys - customers_with_orders_keys

    # Filter to valid customers
    @udf(return_dtype=DataType.bool())
    def is_valid_customer(custkey: pa.Array) -> pa.Array:
        return pc.is_in(custkey, pa.array(list(valid_customers)))

    customer = customer.filter(expr=is_valid_customer(col("c_custkey")))

    # Select columns
    customer = customer.select_columns(["cntrycode", "c_acctbal"])

    # Group and aggregate
    result = (
        customer.groupby("cntrycode")
        .aggregate(
            Count(alias_name="numcust"), Sum(on="c_acctbal", alias_name="totacctbal")
        )
        .sort(key="cntrycode")
    )

    return result


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Execute all 22 TPC-H queries and measure execution times."""

    print("=" * 80)
    print("TPC-H Benchmark - Ray Data Implementation")
    print("=" * 80)
    print()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    queries = [
        ("Q1", q1),
        ("Q2", q2),
        ("Q3", q3),
        ("Q4", q4),
        ("Q5", q5),
        ("Q6", q6),
        ("Q7", q7),
        ("Q8", q8),
        ("Q9", q9),
        ("Q10", q10),
        ("Q11", q11),
        ("Q12", q12),
        ("Q13", q13),
        ("Q14", q14),
        ("Q15", q15),
        ("Q16", q16),
        ("Q17", q17),
        ("Q18", q18),
        ("Q19", q19),
        ("Q20", q20),
        ("Q21", q21),
        ("Q22", q22),
    ]

    results = {}

    for query_name, query_func in queries:
        print(f"Executing {query_name}...")
        start = perf_counter()
        try:
            result = query_func()
            # Materialize results to measure actual execution time
            result.take_all()
            end = perf_counter()
            elapsed = end - start
            results[query_name] = elapsed
            print(f"  {query_name} completed in {elapsed:.2f} seconds")
        except Exception as e:
            end = perf_counter()
            elapsed = end - start
            results[query_name] = None
            print(f"  {query_name} FAILED after {elapsed:.2f} seconds: {str(e)}")
        print()

    # Print summary
    print("=" * 80)
    print("Execution Summary")
    print("=" * 80)
    print(f"{'Query':<10} {'Time (seconds)':<20} {'Status':<10}")
    print("-" * 80)

    total_time = 0
    successful = 0

    for query_name in sorted(results.keys()):
        elapsed = results[query_name]
        if elapsed is not None:
            print(f"{query_name:<10} {elapsed:>18.2f}   {'SUCCESS':<10}")
            total_time += elapsed
            successful += 1
        else:
            print(f"{query_name:<10} {'N/A':>18}   {'FAILED':<10}")

    print("-" * 80)
    print(f"{'Total':<10} {total_time:>18.2f}")
    print(f"\nSuccessful: {successful}/22 queries")
    print("=" * 80)


if __name__ == "__main__":
    main()
