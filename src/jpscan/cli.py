"""命令行入口：jpscan <group> <command>。"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from . import __version__
from .config import Settings, load_settings
from .ledger import Ledger
from .paths import DataPaths
from .store import Store
from .util.common import setup_logging, to_date

app = typer.Typer(help="日本上市股票全量扫描、分析与组合构建工具", no_args_is_help=True)
ingest_app = typer.Typer(help="数据接入", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")


class Ctx:
    def __init__(self, params: Optional[Path] = None):
        self.settings: Settings = load_settings(params_path=params)
        self.paths = DataPaths(self.settings.data_dir).ensure()
        self.ledger = Ledger(self.paths.ledger_sqlite)
        self.store = Store(self.paths)
        setup_logging(logging.INFO, self.paths.root / "logs" / "jpscan.log")

    def jquants(self):
        from .connectors.jquants import JQuantsClient
        plan = str(((self.settings.params.get("data_sources") or {}).get("jquants") or {}).get("plan", "Free"))
        return JQuantsClient(self.settings.jquants_api_key or "", self.settings.jquants_rpm, self.paths, self.ledger, plan)


def _print(obj) -> None:
    typer.echo(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def init(params: Optional[Path] = typer.Option(None, help="参数文件路径")) -> None:
    """创建数据目录与账本。"""
    ctx = Ctx(params)
    _print({"data_dir": str(ctx.paths.root), "ledger": str(ctx.paths.ledger_sqlite), "params_version": ctx.settings.params_version})


@app.command()
def status() -> None:
    """数据完整性与账本摘要。"""
    ctx = Ctx()
    _print({"completeness": ctx.ledger.completeness_summary(), "jquants_window": ctx.ledger.get_kv("jquants.window"),
            "tables": sorted(ctx.store.all_views().keys())})


@ingest_app.command("jquants")
def ingest_jquants(
    mode: str = typer.Argument("daily", help="daily | backfill | rebuild（用原始层重建 staging）"),
    start: Optional[str] = typer.Option(None, "--from", help="回填起始日 YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, "--to", help="回填结束日 YYYY-MM-DD"),
    datasets: Optional[str] = typer.Option(None, help="逗号分隔的数据集名"),
    no_resume: bool = typer.Option(False, help="忽略已完成记录，强制重拉"),
) -> None:
    """J-Quants 采集：daily（增量）或 backfill（全量回填，可断点续传）。"""
    from .jobs.ingest_jquants import JQuantsIngest
    ctx = Ctx()
    job = JQuantsIngest(ctx.settings, ctx.paths, ctx.ledger, ctx.store, ctx.jquants())
    ds = [s.strip() for s in datasets.split(",")] if datasets else None
    if mode == "backfill":
        s = to_date(start) or date(2008, 5, 7)
        e = to_date(end) or date.today()
        _print(job.backfill_roundrobin(s, e, ds or ["master", "bars_daily", "fins_summary", "earnings_date", "valuation"]))
    elif mode == "rebuild":
        _print(job.rebuild_from_raw(ds))
    else:
        _print(job.daily(ds))


features_app = typer.Typer(help="特征计算", no_args_is_help=True)
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build(
    start: Optional[str] = typer.Option(None, "--from", help="起始日 YYYY-MM-DD（缺省：全部）"),
    end: Optional[str] = typer.Option(None, "--to", help="结束日 YYYY-MM-DD"),
) -> None:
    """由 staging 表构建 mart.features_daily。"""
    from .features.build import build_features
    ctx = Ctx()
    run_id = ctx.ledger.start_run("features", __version__, ctx.settings.params_version)
    try:
        df = build_features(ctx.store, ctx.settings.params, to_date(start), to_date(end))
        ctx.ledger.finish_run(run_id, "ok", note=f"rows={len(df)}")
        _print({"run_id": run_id, "rows": int(len(df)), "dates": [str(df['Date'].min()), str(df['Date'].max())] if len(df) else None})
    except Exception as ex:  # noqa: BLE001
        ctx.ledger.finish_run(run_id, "error", note=str(ex)[:300])
        raise


signals_app = typer.Typer(help="事件分、错杀分与环境标签", no_args_is_help=True)
app.add_typer(signals_app, name="signals")


@signals_app.command("build")
def signals_build(
    start: Optional[str] = typer.Option(None, "--from"), end: Optional[str] = typer.Option(None, "--to"),
) -> None:
    """由 features_daily + disclosures + news_items 构建 oversold_daily、event_scores_daily、regime_daily。"""
    from .events.news_score import build_event_scores
    from .events.oversold import build_oversold
    from .regime.regime import build_regime
    ctx = Ctx()
    s, e = to_date(start), to_date(end)
    where = " AND ".join([w for w in [f"Date >= DATE '{s}'" if s else "", f"Date <= DATE '{e}'" if e else ""] if w]) or None
    feat = ctx.store.read("mart", "features_daily", where=where)
    if feat.empty:
        _print({"error": "features_daily 为空，请先运行 features build"})
        raise typer.Exit(1)
    feat["Date"] = feat["Date"].map(to_date)
    disc = ctx.store.read("staging", "disclosures") if ctx.store.exists("staging", "disclosures") else pd.DataFrame()
    news = ctx.store.read("staging", "news_items") if ctx.store.exists("staging", "news_items") else pd.DataFrame()
    macro = ctx.store.read("staging", "macro_series") if ctx.store.exists("staging", "macro_series") else pd.DataFrame()
    ov = build_oversold(feat, disc if not disc.empty else None, ctx.settings.params)
    for p, sub in ov.groupby(pd.to_datetime(ov["Date"]).dt.strftime("%Y-%m")):
        ctx.store.write_merge("mart", "oversold_daily", sub, str(p), ["Code", "Date"])
    dates = sorted(feat["Date"].unique())
    ev = build_event_scores(disc if not disc.empty else None, news if not news.empty else None, dates, ctx.settings.params)
    if not ev.empty:
        for p, sub in ev.groupby(pd.to_datetime(ev["Date"]).dt.strftime("%Y-%m")):
            ctx.store.write_merge("mart", "event_scores_daily", sub, str(p), ["Code", "Date"])
    rg = build_regime(feat, ctx.settings.params, macro if not macro.empty else None)
    ctx.store.write_merge("mart", "regime_daily", rg, "all", ["Date"])
    _print({"oversold_rows": int(len(ov)), "oversold_active": int(ov["active"].sum()), "oversold_candidates": int(ov["oversold_candidate"].sum()),
            "event_score_rows": int(len(ev)), "regime_days": int(len(rg)), "latest_regime": rg.iloc[-1]["regime"] if len(rg) else None})


screen_app = typer.Typer(help="筛选引擎", no_args_is_help=True)
app.add_typer(screen_app, name="screen")


@screen_app.command("run")
def screen_run(as_of: Optional[str] = typer.Option(None, "--date", help="基准日（缺省：features 最新日）"),
               top: int = typer.Option(15, help="终端显示每篮子前 N")) -> None:
    """对某日运行硬过滤 + 六维评分 + 篮子排序，写入 scores_daily 并打印候选。"""
    from .screener.run import candidate_table, screen_day
    ctx = Ctx()
    d = to_date(as_of)
    if d is None:
        mx = ctx.store.query("SELECT max(Date) AS d FROM f", {"f": ("mart", "features_daily")})
        d = to_date(str(mx.iloc[0, 0]))
    run_id = ctx.ledger.start_run("screen", __version__, ctx.settings.params_version, note=str(d))
    scores = screen_day(ctx.store, ctx.settings.params, d, run_id)
    ctx.ledger.finish_run(run_id, "ok", note=f"rows={len(scores)}")
    if scores.empty:
        _print({"error": f"no features for {d}"})
        raise typer.Exit(1)
    cand = candidate_table(scores, ctx.settings.params)
    summary = {"run_id": run_id, "date": str(d), "universe": int(len(scores)), "eligible": int((~scores["excluded"]).sum()),
               "B1": int(scores["in_B1"].sum()), "B2": int(scores["in_B2"].sum()), "B3": int(scores["in_B3"].sum())}
    _print(summary)
    show = ["list_bucket", "list_rank", "Code", "CoName", "S33", "C", "mcap", "pbr", "per_fwd", "div_yield_fwd", "roe_avg", "score_value",
            "score_growth", "score_quality", "score_oversold", "score_trend", "list_score", "soft_flags"]
    if not cand.empty:
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            typer.echo(cand[cand["list_rank"] <= top][[c for c in show if c in cand.columns]].to_string(index=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
