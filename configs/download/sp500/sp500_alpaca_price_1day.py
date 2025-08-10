workdir = "workdir"
assets_name = "sp500"
source = "alpaca"
data_type = "price"
level = "1day"
tag = f"{assets_name}_{source}_{data_type}_{level}"
exp_path = f"{workdir}/{tag}"
log_path = "finworld.log"

downloader = dict(
    type = "PriceDownloader",
    source = source,
    assets_path = f"configs/_asset_list_/{assets_name}.json",
    start_date = "2015-05-01",
    end_date = "2025-05-01",
    level=level,
    format="%Y-%m-%d",
    max_concurrent = 6,
)