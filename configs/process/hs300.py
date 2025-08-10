workdir = "workdir"
assets_name = "hs300"
source = None
data_type = None
level = None
tag = f"{assets_name}"
exp_path = f"{workdir}/{tag}"
log_path = "finworld.log"

processor = dict(
    type = "AggProcessor",
    procs_config = [
        dict(
            type="Processor",
            assets_name=assets_name,
            data_path=f"workdir/{assets_name}_fmp_price_1day/price",
            data_type="price",
            assets_path = f"configs/_asset_list_/{assets_name}.json",
            source="fmp",
            start_date="1995-05-01",
            end_date="2025-05-01",
            level="1day",
            format="%Y-%m-%d",
            feature_type = None,
            max_concurrent = 6,
        ),
        dict(
            type="Processor",
            assets_name=assets_name,
            data_path=f"workdir/{assets_name}_fmp_price_1min/price",
            data_type="price",
            assets_path = f"configs/_asset_list_/{assets_name}.json",
            source="fmp",
            start_date="1995-05-01",
            end_date="2025-05-01",
            level="1min",
            format="%Y-%m-%d",
            feature_type = None,
            max_concurrent = 6,
        ),
        dict(
            type="Processor",
            assets_name=assets_name,
            data_path=f"workdir/{assets_name}_akshare_price_1day/price",
            data_type="price",
            assets_path = f"configs/_asset_list_/{assets_name}.json",
            source="akshare",
            start_date="1995-05-01",
            end_date="2025-05-01",
            level="1day",
            format="%Y-%m-%d",
            feature_type = None,
            max_concurrent = 6,
        ),
        dict(
            type="Processor",
            assets_name=assets_name,
            data_path=f"workdir/{assets_name}_akshare_price_1day/price",
            data_type="feature",
            assets_path=f"configs/_asset_list_/{assets_name}.json",
            source="akshare",
            start_date="1995-05-01",
            end_date="2025-05-01",
            level="1day",
            format="%Y-%m-%d",
            feature_type="Alpha158",
            max_concurrent=6,
        ),
        dict(
            type="Processor",
            assets_name=assets_name,
            data_path=f"workdir/{assets_name}_tushare_price_1day/price",
            data_type="price",
            assets_path = f"configs/_asset_list_/{assets_name}.json",
            source="tushare",
            start_date="1995-05-01",
            end_date="2025-05-01",
            level="1day",
            format="%Y-%m-%d",
            feature_type = None,
            max_concurrent = 6,
        )
    ],
    assets_path = f"configs/_asset_list_/{assets_name}.json",
    max_concurrent = 6,
    repo_id = tag,
    repo_type = "dataset",
)