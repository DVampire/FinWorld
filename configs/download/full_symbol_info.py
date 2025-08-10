workdir = "workdir"
source = "fmp"
tag = "full86k"
exp_path = f"{workdir}/{tag}"
log_path = "finworld.log"

downloader = dict(
    type = "SymbolInfoDownloader",
    source = source,
    save_name = tag,
    max_concurrent = 2000,
)