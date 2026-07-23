from intelligence import data_catalog


def test_catalog_keeps_paid_domains_explicitly_unavailable() -> None:
    domains = {
        "derivatives": {"status": "LIVE", "source": "Kraken"},
        "options": {"status": "LIVE", "source": "Deribit"},
        "etf_flows": {"status": "PROVIDER_REQUIRED", "source": "Licensed ETF provider"},
        "stablecoins": {"status": "LIVE", "source": "DefiLlama"},
        "onchain_flows": {
            "status": "PROVIDER_REQUIRED",
            "source": "Licensed on-chain provider",
        },
        "macro": {"status": "LIMITED", "source": "FRED"},
        "network": {"status": "LIVE", "source": "mempool.space"},
        "sentiment": {"status": "LIVE", "source": "Alternative.me"},
        "information_velocity": {
            "status": "PROVIDER_REQUIRED",
            "source": "Licensed social/news firehose",
        },
        "coinbase_premium": {"status": "LIVE", "source": "Coinbase vs Kraken"},
    }
    catalog = data_catalog(domains, prediction_market_status="LIVE")
    status_by_domain = {row["domain"]: row["status"] for row in catalog}
    assert status_by_domain["ETF flows"] == "PROVIDER_REQUIRED"
    assert status_by_domain["On-chain flows"] == "PROVIDER_REQUIRED"
    assert status_by_domain["Information velocity"] == "PROVIDER_REQUIRED"
    assert status_by_domain["Prediction markets"] == "LIVE"
